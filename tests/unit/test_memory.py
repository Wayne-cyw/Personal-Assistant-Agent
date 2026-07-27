import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import app.agent.memory as memory_module
from app.agent.memory import (
    ToolEvent,
    _one_line_receipt,
    _parse_summary_json,
    _split_for_eviction,
    assemble_messages,
    load_memory,
    persist_turn,
    reload_and_reconcile,
    render_pinned_profile,
    run_eviction,
    run_reconciliation,
)
from app.agent.providers.base import LLMResponse, Usage
from app.agent.providers.fake import FakeProvider
from app.agent.session_lock import SessionBusyError, session_turn_lock
from app.config import settings
from app.db import session as db_session
from app.db.models import Base, SessionRow
from app.db.session import advance_summary, get_or_create_session, set_visitor_info


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine]:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture(autouse=True)
def _patch_session_factory(
    monkeypatch: pytest.MonkeyPatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """run_eviction opens its own DB session via get_session_factory() —
    point it at the test engine instead of the real (unconfigured) one.
    """
    monkeypatch.setattr(db_session, "_session_factory", session_factory)


def _summary_response(json_text: str) -> LLMResponse:
    usage = Usage(input_tokens=1, output_tokens=1)
    return LLMResponse(text=json_text, usage=usage, finish_reason="stop")


def _summary_json(context: str, **kwargs: object) -> str:
    import json

    payload = {"visitor_context": context, "open_questions": [], "commitments": [], "notes": []}
    payload.update(kwargs)
    return json.dumps(payload)


# --- render_pinned_profile / _render_summary --------------------------------


def test_render_pinned_profile_empty_when_nothing_captured() -> None:
    session = SessionRow(id="s1")
    assert render_pinned_profile(session) == ""


def test_render_pinned_profile_includes_name_linkedin_and_facts() -> None:
    session = SessionRow(
        id="s1",
        visitor_name="Priya",
        visitor_linkedin="https://linkedin.com/in/priya",
        pinned_facts_json=["likes Rust", "hiring for backend"],
    )
    rendered = render_pinned_profile(session)
    assert "Priya" in rendered
    assert "linkedin.com/in/priya" in rendered
    assert "likes Rust" in rendered
    assert "hiring for backend" in rendered


# --- assemble_messages: cache breakpoint placement --------------------------


def test_cache_breakpoint_set_on_pinned_and_summary_block_when_present() -> None:
    from app.agent.memory import Memory

    memory = Memory(pinned_profile_text="Visitor profile:\nName: Priya", summary_text="")
    messages = assemble_messages(memory, "SYSTEM", "hello")

    pinned_message = messages[1]
    assert pinned_message.content.startswith("Visitor profile:")
    assert pinned_message.cache_breakpoint is True
    # Nothing else carries the breakpoint marker.
    assert all(m.cache_breakpoint is False for m in messages if m is not pinned_message)


def test_no_cache_breakpoint_message_when_nothing_pinned_or_summarized() -> None:
    from app.agent.memory import Memory

    memory = Memory(pinned_profile_text="", summary_text="")
    messages = assemble_messages(memory, "SYSTEM", "hello")

    assert all(not m.cache_breakpoint for m in messages)
    assert len(messages) == 2  # just system + current message


# --- _split_for_eviction ------------------------------------------------


def test_split_for_eviction_keeps_nothing_evicted_under_low_water() -> None:
    from app.db.models import Message

    rows = [Message(id=i, session_id="s1", role="user", content="hi") for i in range(3)]
    evicted, kept = _split_for_eviction(rows)
    assert evicted == []
    assert kept == rows


def test_split_for_eviction_evicts_oldest_first_down_to_low_water(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.agent.tokens import count_tokens
    from app.db.models import Message

    one_row_tokens = count_tokens("x" * 40)
    monkeypatch.setattr(settings, "window_low_tokens", one_row_tokens)
    rows = [Message(id=i, session_id="s1", role="user", content="x" * 40) for i in range(5)]
    evicted, kept = _split_for_eviction(rows)
    assert [m.id for m in evicted] == [0, 1, 2, 3]
    assert [m.id for m in kept] == [4]


# --- _parse_summary_json -------------------------------------------------


def test_parse_summary_json_accepts_plain_json() -> None:
    result = _parse_summary_json(_summary_json("recruiter"))
    assert result is not None
    assert result.visitor_context == "recruiter"


def test_parse_summary_json_strips_markdown_code_fence() -> None:
    fenced = f"```json\n{_summary_json('recruiter')}\n```"
    result = _parse_summary_json(fenced)
    assert result is not None
    assert result.visitor_context == "recruiter"


def test_parse_summary_json_rejects_non_json() -> None:
    assert _parse_summary_json("not json at all") is None


def test_parse_summary_json_rejects_wrong_shape() -> None:
    assert _parse_summary_json('{"unexpected": "shape", "open_questions": "not-a-list"}') is None


# --- _one_line_receipt -----------------------------------------------------


def test_one_line_receipt_truncates_long_results() -> None:
    receipt = _one_line_receipt({"data": "x" * 500})
    assert len(receipt) <= 200
    assert receipt.endswith("…")


# --- persist_turn: high-water threshold ------------------------------------


async def test_persist_turn_under_high_water_returns_false(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        await get_or_create_session(db, "s1")
        needs_eviction = await persist_turn(db, "s1", "hello", "hi there", [])
    assert needs_eviction is False


async def test_persist_turn_over_high_water_returns_true(
    monkeypatch: pytest.MonkeyPatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    monkeypatch.setattr(settings, "window_high_tokens", 1)
    async with session_factory() as db:
        await get_or_create_session(db, "s1")
        needs_eviction = await persist_turn(db, "s1", "hello there", "hi, how can I help?", [])
    assert needs_eviction is True


async def test_persist_turn_writes_tool_receipts_but_skips_meta_tools(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        await get_or_create_session(db, "s1")
        await persist_turn(
            db,
            "s1",
            "what's today's date?",
            "it's today",
            [
                ToolEvent(name="get_current_date", result={"date": "2026-07-27"}),
                ToolEvent(name="save_visitor_info", result={"saved": {}}, persist_receipt=False),
            ],
        )
        session = await get_or_create_session(db, "s1")
        memory = await load_memory(db, session)

    tool_rows = [m for m in memory.window_messages if m.role == "tool"]
    assert len(tool_rows) == 1
    assert tool_rows[0].tool_name == "get_current_date"


# --- eviction (a)/(b)/(c): batched, summarizer scoped to newly evicted only ----


async def test_eviction_not_triggered_under_high_water_summary_stays_none(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        await get_or_create_session(db, "s1")
        await persist_turn(db, "s1", "hi", "hello!", [])
        session = await get_or_create_session(db, "s1")
    assert session.summary_json is None


async def test_eviction_folds_exactly_the_evicted_range_and_lands_at_or_under_low_water(
    monkeypatch: pytest.MonkeyPatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    monkeypatch.setattr(settings, "window_low_tokens", 5)
    fake = FakeProvider(responses=[_summary_response(_summary_json("first fold"))])

    async with session_factory() as db:
        await get_or_create_session(db, "s1")
        for i in range(6):
            await persist_turn(db, "s1", f"question {i}", f"answer {i}", [])

    await run_eviction("s1", fake)

    async with session_factory() as db:
        session = await get_or_create_session(db, "s1")
        assert session.summary_json == {
            "visitor_context": "first fold",
            "open_questions": [],
            "commitments": [],
            "notes": [],
        }
        assert session.summary_through_message_id is not None
        assert session.eviction_count == 1
        memory = await load_memory(db, session)
        remaining_text = "\n".join(m.content for m in memory.window_messages)
    from app.agent.tokens import count_tokens

    assert count_tokens(remaining_text) <= settings.window_low_tokens

    assert len(fake.calls) == 1
    summarizer_input = fake.calls[0][1].content  # system + user; user carries the turns
    assert "none" in summarizer_input  # no prior summary on the first fold


async def test_second_eviction_summarizer_receives_only_newly_evicted_turns(
    monkeypatch: pytest.MonkeyPatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    monkeypatch.setattr(settings, "window_low_tokens", 5)
    fake = FakeProvider(
        responses=[
            _summary_response(_summary_json("first batch")),
            _summary_response(_summary_json("second batch")),
        ]
    )

    async with session_factory() as db:
        await get_or_create_session(db, "s1")
        for i in range(6):
            await persist_turn(db, "s1", f"question {i}", f"answer {i}", [])
    await run_eviction("s1", fake)

    async with session_factory() as db:
        for i in range(6, 12):
            await persist_turn(db, "s1", f"question {i}", f"answer {i}", [])
    await run_eviction("s1", fake)

    assert len(fake.calls) == 2
    second_call_user_content = fake.calls[1][1].content
    # The turns folded into the *first* summary must not appear again in the
    # second call's input — each turn is folded at most once (4.3
    # incrementality guarantee).
    assert "question 0" not in second_call_user_content
    assert "first batch" in second_call_user_content  # existing summary is passed through
    assert "question 6" in second_call_user_content or "question 7" in second_call_user_content


async def test_concurrent_evictions_on_the_same_session_serialize_not_race(
    monkeypatch: pytest.MonkeyPatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Regression test for a race a code review caught: without the
    per-session lock, two run_eviction calls racing on the same session can
    each read a stale summary_through_message_id and commit
    advance_summary independently — the second's stale, smaller advance can
    overwrite the first's, regressing the boundary backward and causing
    already-summarized turns to be silently re-exposed in the live window
    (violating "each turn summarized at most once, ever"). With the lock,
    the second call only runs after the first has fully committed, sees the
    already-advanced boundary, and finds nothing left to evict.
    """
    monkeypatch.setattr(settings, "window_low_tokens", 5)
    fake = FakeProvider(
        responses=[
            _summary_response(_summary_json("first")),
            _summary_response(_summary_json("second")),
        ]
    )

    async with session_factory() as db:
        await get_or_create_session(db, "s1")
        for i in range(6):
            await persist_turn(db, "s1", f"question {i}", f"answer {i}", [])

    await asyncio.gather(run_eviction("s1", fake), run_eviction("s1", fake))

    # Only one eviction actually found anything to evict and called the
    # summarizer — the second, serialized behind the lock, saw the
    # already-advanced boundary and had nothing left to do.
    assert len(fake.calls) == 1

    async with session_factory() as db:
        session = await get_or_create_session(db, "s1")
    assert session.eviction_count == 1
    assert session.summary_json is not None
    assert session.summary_json["visitor_context"] == "first"


async def test_run_eviction_skips_gracefully_when_session_busy(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    @contextlib.asynccontextmanager
    async def _always_busy(session_id: str, **_kwargs: object) -> AsyncGenerator[None]:
        raise SessionBusyError(session_id)
        yield  # pragma: no cover — unreachable; makes this an async generator function

    monkeypatch.setattr(memory_module, "session_turn_lock", _always_busy)

    with caplog.at_level(logging.WARNING):
        await run_eviction("s1", FakeProvider(responses=[]))  # must not raise

    assert any("skipped" in r.getMessage() for r in caplog.records)


async def test_background_eviction_does_not_block_a_concurrent_live_turn(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Regression test for a second review pass's finding: run_eviction
    must use a lock key namespaced separately from
    app/api/chat.py's live-turn session_turn_lock(session_id) hold — a live
    turn never writes summary_json/summary_through_message_id, so it has
    nothing to race with a background eviction on, and must not be forced
    to wait behind one (that would violate 4.3's "memory bookkeeping adds
    zero user-facing latency" guarantee just as badly as the original
    synchronous-reconciliation bug did).
    """
    async with session_factory() as db:
        await get_or_create_session(db, "s1")
        await persist_turn(db, "s1", "hi", "hello!", [])

    async def hold_eviction_lock_for(seconds: float) -> None:
        async with session_turn_lock(memory_module._memory_lock_key("s1")):
            await asyncio.sleep(seconds)

    # A live turn's own lock (app/api/chat.py's, unnamespaced) must succeed
    # immediately even while a background eviction "holds" the memory-only
    # lock for the same session — proving the two don't contend.
    eviction_hold = asyncio.create_task(hold_eviction_lock_for(1.0))
    await asyncio.sleep(0.01)  # let the hold actually acquire first
    start = asyncio.get_running_loop().time()
    async with session_turn_lock("s1"):
        pass
    elapsed = asyncio.get_running_loop().time() - start

    assert elapsed < 0.5  # nowhere near the eviction hold's 1s duration
    eviction_hold.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await eviction_hold


# --- (d) prefix stability between evictions --------------------------------


async def test_message_list_is_prefix_stable_between_evictions(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        await get_or_create_session(db, "s1")
        await persist_turn(db, "s1", "turn 0", "reply 0", [])
        session = await get_or_create_session(db, "s1")
        memory_a = await load_memory(db, session)
        messages_a = assemble_messages(memory_a, "SYSTEM", "turn 1")

        await persist_turn(db, "s1", "turn 1", "reply 1", [])
        session = await get_or_create_session(db, "s1")
        memory_b = await load_memory(db, session)
        messages_b = assemble_messages(memory_b, "SYSTEM", "turn 2")

    # Everything request t saw, except its own trailing "current message"
    # (which becomes history for request t+1), must reappear byte-for-byte
    # at the start of request t+1's message list — the cache-hit property:
    # nothing before the first new turn is ever rewritten.
    prefix = messages_a[:-1]
    assert len(prefix) > 1  # sanity: this test would pass trivially if empty
    assert messages_b[: len(prefix)] == prefix


# --- (e) name recall survives eviction --------------------------------------


async def test_name_survives_eviction_via_pinned_profile(
    monkeypatch: pytest.MonkeyPatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    monkeypatch.setattr(settings, "window_low_tokens", 5)
    fake = FakeProvider(responses=[_summary_response(_summary_json("chatting"))])

    async with session_factory() as db:
        await get_or_create_session(db, "s1")
        await set_visitor_info(db, "s1", name="Priya")
        for i in range(6):
            await persist_turn(db, "s1", f"question {i}", f"answer {i}", [])
    await run_eviction("s1", fake)

    async with session_factory() as db:
        session = await get_or_create_session(db, "s1")
        memory = await load_memory(db, session)
        messages = assemble_messages(memory, "SYSTEM", "what's my name?")

    assert any("Priya" in m.content for m in messages)


# --- (f) malformed summarizer output degrades gracefully --------------------


async def test_malformed_summarizer_output_skips_gracefully_after_retry(
    monkeypatch: pytest.MonkeyPatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    monkeypatch.setattr(settings, "window_low_tokens", 5)
    fake = FakeProvider(
        responses=[_summary_response("not json"), _summary_response("still not json")]
    )
    async with session_factory() as db:
        await get_or_create_session(db, "s1")
        for i in range(6):
            await persist_turn(db, "s1", f"question {i}", f"answer {i}", [])

    await run_eviction("s1", fake)  # must not raise

    async with session_factory() as db:
        session = await get_or_create_session(db, "s1")
        # Skipped gracefully: no summary was written, and nothing was
        # evicted from the DB log (it's still there for the next attempt).
        assert session.summary_json is None
        assert session.summary_through_message_id is None
    assert len(fake.calls) == 2  # exactly one retry


# --- (h) reconciliation ------------------------------------------------------


async def test_reload_and_reconcile_overwrites_content_not_boundary(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        await get_or_create_session(db, "s1")
        for i in range(3):
            await persist_turn(db, "s1", f"question {i}", f"answer {i}", [])
    # Seed a stale/contradicted summary directly, as if an earlier eviction
    # had folded these turns with a wrong conclusion (no eviction has
    # actually run yet, so summary_through_message_id=3 is set by hand here
    # to simulate "a summary already covers these turns").
    async with session_factory() as db:
        from app.db.session import advance_summary

        await advance_summary(
            db,
            "s1",
            summary_json={
                "visitor_context": "stale/wrong",
                "open_questions": [],
                "commitments": [],
                "notes": [],
            },
            summary_through_message_id=3,
        )

    fake = FakeProvider(responses=[_summary_response(_summary_json("corrected context"))])
    async with session_factory() as db:
        result = await reload_and_reconcile(
            db, "s1", fake, trigger="model_detected", detail="visitor corrected role"
        )

    assert result is not None
    assert result.visitor_context == "corrected context"
    async with session_factory() as db:
        session = await get_or_create_session(db, "s1")
        assert session.summary_json["visitor_context"] == "corrected context"  # type: ignore[index]
        assert session.summary_through_message_id == 3  # unchanged


async def test_reload_and_reconcile_no_summary_yet_is_a_noop(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    fake = FakeProvider(responses=[])
    async with session_factory() as db:
        await get_or_create_session(db, "s1")
        result = await reload_and_reconcile(db, "s1", fake, trigger="model_detected")
    assert result is None
    assert fake.calls == []


# --- run_reconciliation: background-task entry point -------------------------


async def test_run_reconciliation_opens_own_session_and_updates_summary(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Issue #11 review fix: flag_summary_conflict (app/tools/registry.py)
    no longer calls reload_and_reconcile synchronously inside the turn — it
    only records intent, and app/api/chat.py schedules this function as a
    post-response background task instead, mirroring run_eviction.
    """
    async with session_factory() as db:
        await get_or_create_session(db, "s1")
        await advance_summary(
            db,
            "s1",
            summary_json={
                "visitor_context": "stale",
                "open_questions": [],
                "commitments": [],
                "notes": [],
            },
            summary_through_message_id=0,
        )
    fake = FakeProvider(responses=[_summary_response(_summary_json("corrected"))])

    await run_reconciliation("s1", fake, trigger="model_detected", detail="visitor corrected role")

    async with session_factory() as db:
        session = await get_or_create_session(db, "s1")
    assert session.summary_json is not None
    assert session.summary_json["visitor_context"] == "corrected"
    assert session.summary_through_message_id == 0  # unchanged


async def test_run_reconciliation_skips_gracefully_when_session_busy(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    @contextlib.asynccontextmanager
    async def _always_busy(session_id: str, **_kwargs: object) -> AsyncGenerator[None]:
        raise SessionBusyError(session_id)
        yield  # pragma: no cover — unreachable; makes this an async generator function

    monkeypatch.setattr(memory_module, "session_turn_lock", _always_busy)

    with caplog.at_level(logging.WARNING):
        # must not raise
        await run_reconciliation("s1", FakeProvider(responses=[]), trigger="model_detected")

    assert any("skipped" in r.getMessage() for r in caplog.records)
