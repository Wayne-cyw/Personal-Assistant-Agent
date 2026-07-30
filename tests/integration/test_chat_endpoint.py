from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.agent.intro import INTRO_MESSAGE
from app.agent.providers.base import (
    LLMResponse,
    Message,
    StreamEvent,
    ToolCall,
    ToolDef,
    UpstreamError,
    Usage,
)
from app.agent.providers.fake import FakeProvider
from app.api.chat import get_main_provider, get_summarizer_provider
from app.db.models import Base, SessionRow
from app.db.session import advance_summary, get_db
from app.main import app


class _OrderTrackingProvider:
    """Sleeps before returning, appending to a shared `order` list around
    the sleep — used to prove two concurrent requests actually serialize
    (the per-session lock, Issue #11) rather than interleave. FakeProvider
    resolves synchronously and can't create a race window on its own.
    """

    def __init__(self, order: list[str], seconds: float = 0.05) -> None:
        self._order = order
        self._seconds = seconds

    async def complete(
        self, messages: list[Message], tools: list[ToolDef], max_tokens: int
    ) -> LLMResponse:
        import asyncio

        self._order.append("start")
        await asyncio.sleep(self._seconds)
        self._order.append("end")
        return _response("ok")

    def complete_stream(
        self, messages: list[Message], tools: list[ToolDef], max_tokens: int
    ) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError  # unused by this test


class _RaisingProvider:
    """A provider that always raises, to exercise the 503/500 error paths."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def complete(
        self, messages: list[Message], tools: list[ToolDef], max_tokens: int
    ) -> LLMResponse:
        raise self._exc

    async def complete_stream(
        self, messages: list[Message], tools: list[ToolDef], max_tokens: int
    ) -> AsyncIterator[StreamEvent]:
        raise self._exc
        yield  # pragma: no cover — unreachable; makes this an async generator function


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine]:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def client(engine: AsyncEngine) -> AsyncGenerator[httpx.AsyncClient]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    try:
        # raise_app_exceptions=False: let the registered Exception handler
        # convert unhandled errors into the real HTTP response a deployed
        # server would send, instead of re-raising for pytest debugging.
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_main_provider, None)
        app.dependency_overrides.pop(get_summarizer_provider, None)


def _response(text: str) -> LLMResponse:
    usage = Usage(input_tokens=1, output_tokens=1)
    return LLMResponse(text=text, usage=usage, finish_reason="stop")


def _use_fake_provider(responses: list[LLMResponse]) -> FakeProvider:
    fake = FakeProvider(responses=responses)
    app.dependency_overrides[get_main_provider] = lambda: fake
    return fake


async def _get_session(engine: AsyncEngine, session_id: str) -> SessionRow | None:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as db:
        return await db.get(SessionRow, session_id)


async def _prime_past_turn_zero(client: httpx.AsyncClient, session_id: str) -> None:
    """Every session's first message gets the deterministic intro reply
    (Issue #9), not an LLM call — send and discard one throwaway turn-zero
    message so the rest of a test can exercise the real LLM-driven path.
    """
    response = await client.post(
        "/v1/chat", json={"session_id": session_id, "message": "(priming turn zero)"}
    )
    assert response.status_code == 200


async def test_two_turn_conversation_remembers_first_turn(client: httpx.AsyncClient) -> None:
    await _prime_past_turn_zero(client, "sess-1")
    fake = _use_fake_provider([_response("Nice to meet you, Sam."), _response("Your name is Sam.")])

    # "call me Sam" deliberately avoids the Issue #9 name-extraction patterns
    # ("my name is"/"I'm") so this test stays focused on history/memory, not
    # visitor-info capture (covered separately below).
    first = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "call me Sam"}
    )
    assert first.status_code == 200
    assert first.json() == {"reply": "Nice to meet you, Sam.", "type": "message", "data": None}

    second = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "what's my name?"}
    )
    assert second.status_code == 200
    assert second.json()["reply"] == "Your name is Sam."

    second_call_messages = fake.calls[1]
    contents = [m.content for m in second_call_messages]
    assert "call me Sam" in contents
    assert "Nice to meet you, Sam." in contents
    assert "what's my name?" in contents


async def test_fresh_session_first_reply_is_byte_identical_to_intro(
    client: httpx.AsyncClient,
) -> None:
    _use_fake_provider([_response("unused — turn zero must not call the LLM")])

    response = await client.post(
        "/v1/chat", json={"session_id": "sess-new", "message": "hello?"}
    )

    assert response.status_code == 200
    assert response.json() == {"reply": INTRO_MESSAGE, "type": "message", "data": None}


async def test_turn_zero_spends_zero_llm_tokens(client: httpx.AsyncClient) -> None:
    fake = _use_fake_provider([_response("unused")])

    await client.post("/v1/chat", json={"session_id": "sess-new", "message": "hello?"})

    assert fake.calls == []


async def test_turn_zero_question_is_answered_on_next_send(client: httpx.AsyncClient) -> None:
    """Issue #9: if the caller's first payload already contains a question,
    the prefix is still returned for that call; the question is answered on
    the next send, and the first message remains in history for that reply.
    """
    fake = _use_fake_provider([_response("Yes, they know Rust.")])

    first = await client.post(
        "/v1/chat", json={"session_id": "sess-new", "message": "does the owner know Rust?"}
    )
    assert first.status_code == 200
    assert first.json()["reply"] == INTRO_MESSAGE

    second = await client.post(
        "/v1/chat", json={"session_id": "sess-new", "message": "well?"}
    )
    assert second.status_code == 200
    assert second.json()["reply"] == "Yes, they know Rust."

    contents = [m.content for m in fake.calls[0]]
    assert "does the owner know Rust?" in contents
    assert "well?" in contents


async def test_name_volunteered_on_turn_zero_is_captured_silently(
    client: httpx.AsyncClient, engine: AsyncEngine
) -> None:
    """Turn zero's reply must stay byte-identical to intro.md even if a name
    was volunteered in that same first message (Issue #9) — capture happens,
    acknowledgment does not.
    """
    _use_fake_provider([_response("unused — turn zero must not call the LLM")])

    response = await client.post(
        "/v1/chat", json={"session_id": "sess-new", "message": "hi, my name is Priya"}
    )

    assert response.json()["reply"] == INTRO_MESSAGE  # no appended acknowledgment
    session = await _get_session(engine, "sess-new")
    assert session is not None
    assert session.visitor_name == "Priya"


def _tool_call_response(name: str, arguments: dict[str, object]) -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=[ToolCall(id="call_1", name=name, arguments=arguments)],
        usage=Usage(input_tokens=1, output_tokens=1),
        finish_reason="tool_calls",
    )


async def test_name_volunteered_after_turn_zero_is_captured_via_tool_call(
    client: httpx.AsyncClient, engine: AsyncEngine
) -> None:
    """Issue #11: post-turn-zero visitor-info capture is tool-based
    (save_visitor_info), not the regex extractor — the model's own final
    text is returned verbatim, with no code-appended acknowledgment suffix
    (the model composes its own acknowledgment, having just seen the tool
    result).
    """
    await _prime_past_turn_zero(client, "sess-1")
    fake = _use_fake_provider(
        [
            _tool_call_response("save_visitor_info", {"name": "Priya"}),
            _response("Nice to meet you, Priya!"),
        ]
    )

    response = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "my name is Priya"}
    )
    assert response.json()["reply"] == "Nice to meet you, Priya!"

    session = await _get_session(engine, "sess-1")
    assert session is not None
    assert session.visitor_name == "Priya"
    assert fake.calls  # sanity: the LLM path was actually exercised


async def test_linkedin_captured_via_tool_call(
    client: httpx.AsyncClient, engine: AsyncEngine
) -> None:
    await _prime_past_turn_zero(client, "sess-1")
    _use_fake_provider(
        [
            _tool_call_response(
                "save_visitor_info", {"linkedin": "https://www.linkedin.com/in/priya-example"}
            ),
            _response("Got it."),
        ]
    )

    response = await client.post(
        "/v1/chat",
        json={
            "session_id": "sess-1",
            "message": "here's my linkedin: https://www.linkedin.com/in/priya-example",
        },
    )

    assert response.json()["reply"] == "Got it."
    session = await _get_session(engine, "sess-1")
    assert session is not None
    assert session.visitor_linkedin == "https://www.linkedin.com/in/priya-example"


async def test_uses_real_system_prompt_not_a_placeholder(client: httpx.AsyncClient) -> None:
    from app.agent.prompts import SYSTEM_PROMPT

    await _prime_past_turn_zero(client, "sess-1")
    fake = _use_fake_provider([_response("hi")])

    await client.post("/v1/chat", json={"session_id": "sess-1", "message": "hello"})

    system_message = fake.calls[0][0]
    assert system_message.role == "system"
    assert system_message.content == SYSTEM_PROMPT


async def test_window_retains_all_turns_under_the_token_budget(client: httpx.AsyncClient) -> None:
    """Issue #11: the window is token-budgeted (WINDOW_HIGH_TOKENS), not a
    fixed turn count — a short conversation like this one stays entirely
    under the default budget, so nothing is evicted and every turn remains
    in context.
    """
    await _prime_past_turn_zero(client, "sess-1")
    fake = _use_fake_provider([_response(f"reply {i}") for i in range(11)])

    for i in range(11):
        await client.post("/v1/chat", json={"session_id": "sess-1", "message": f"turn {i}"})

    last_call_messages = fake.calls[-1]
    contents = [m.content for m in last_call_messages]
    assert "turn 0" in contents  # still in the 10-turn window as of turn 10 (0-indexed)
    assert "reply 0" in contents


async def test_eviction_runs_as_background_task_and_updates_summary(
    client: httpx.AsyncClient, engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end proof that the FastAPI BackgroundTasks wiring (Issue #11)
    actually runs: memory.py's eviction/summarization behavior itself is
    covered thoroughly in tests/unit/test_memory.py — this only exercises
    that app/api/chat.py schedules and the background task actually
    executes through the real HTTP path.
    """
    from app.config import settings
    from app.db import session as db_session

    monkeypatch.setattr(settings, "window_high_tokens", 10)
    monkeypatch.setattr(settings, "window_low_tokens", 5)
    # run_eviction (Issue #11) opens its own DB session via
    # get_session_factory() since it runs as a background task, after the
    # request-scoped session is gone — point it at the test engine instead
    # of the real (unconfigured) one.
    monkeypatch.setattr(
        db_session, "_session_factory", async_sessionmaker(engine, expire_on_commit=False)
    )

    await _prime_past_turn_zero(client, "sess-1")
    _use_fake_provider([_response(f"reply {i}") for i in range(6)])
    summary_json = (
        '{"visitor_context": "chatting", "open_questions": [], '
        '"commitments": [], "notes": []}'
    )
    summary_usage = Usage(input_tokens=1, output_tokens=1)
    fake_summarizer = FakeProvider(
        responses=[LLMResponse(text=summary_json, usage=summary_usage, finish_reason="stop")]
    )
    app.dependency_overrides[get_summarizer_provider] = lambda: fake_summarizer

    for i in range(6):
        await client.post("/v1/chat", json={"session_id": "sess-1", "message": f"turn {i}"})

    session = await _get_session(engine, "sess-1")
    assert session is not None
    assert session.summary_json is not None
    assert session.summary_json["visitor_context"] == "chatting"
    assert fake_summarizer.calls  # the background task actually ran


async def test_flag_summary_conflict_reconciles_as_background_task_not_synchronously(
    client: httpx.AsyncClient, engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end proof that app/api/chat.py's post-response scheduling for
    flag_summary_conflict actually works: the summary is corrected after a
    real HTTP round trip through the endpoint, with the tool-call turn's
    reply being the model's own final text (not anything reconciliation-
    related). This does NOT by itself prove no *synchronous* summarizer
    call happened first — httpx's ASGI transport runs FastAPI
    BackgroundTasks within the same client.post() call either way, so a
    synchronous-call regression here would look identical from the outside.
    That property (flag_summary_conflict never touches the summarizer
    inline) is what tests/unit/test_tool_registry.py's
    test_flag_summary_conflict_does_not_touch_the_summarizer directly
    proves, by asserting zero calls immediately after execute_tool.
    """
    from app.db import session as db_session

    monkeypatch.setattr(
        db_session, "_session_factory", async_sessionmaker(engine, expire_on_commit=False)
    )

    await _prime_past_turn_zero(client, "sess-1")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as db:
        await advance_summary(
            db,
            "sess-1",
            summary_json={
                "visitor_context": "stale",
                "open_questions": [],
                "commitments": [],
                "notes": [],
            },
            summary_through_message_id=1,
        )

    _use_fake_provider(
        [
            _tool_call_response(
                "flag_summary_conflict",
                {"explanation": "visitor said they're no longer interested"},
            ),
            _response("Got it, noted."),
        ]
    )
    corrected_json = (
        '{"visitor_context": "corrected", "open_questions": [], '
        '"commitments": [], "notes": []}'
    )
    summary_usage = Usage(input_tokens=1, output_tokens=1)
    fake_summarizer = FakeProvider(
        responses=[LLMResponse(text=corrected_json, usage=summary_usage, finish_reason="stop")]
    )
    app.dependency_overrides[get_summarizer_provider] = lambda: fake_summarizer

    response = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "actually never mind"}
    )

    # The reply is the model's own final text, not anything
    # reconciliation-related — see the docstring above for what this test
    # does and doesn't prove about synchronicity.
    assert response.json()["reply"] == "Got it, noted."

    session = await _get_session(engine, "sess-1")
    assert session is not None
    assert session.summary_json is not None
    assert session.summary_json["visitor_context"] == "corrected"
    assert session.summary_through_message_id == 1  # unchanged — content corrected, not coverage
    assert fake_summarizer.calls  # the background task actually ran


async def test_concurrent_same_session_requests_serialize_through_the_endpoint(
    client: httpx.AsyncClient,
) -> None:
    """Proves app/api/chat.py actually applies session_turn_lock end to end
    — the lock primitive itself, and its 429-on-timeout behavior, are
    unit-tested directly in tests/unit/test_session_lock.py.
    """
    import asyncio

    await _prime_past_turn_zero(client, "sess-1")
    order: list[str] = []
    app.dependency_overrides[get_main_provider] = lambda: _OrderTrackingProvider(order)

    responses = await asyncio.gather(
        client.post("/v1/chat", json={"session_id": "sess-1", "message": "first"}),
        client.post("/v1/chat", json={"session_id": "sess-1", "message": "second"}),
    )

    assert all(r.status_code == 200 for r in responses)
    # If the lock weren't applied, both "start"s would appear before either
    # "end" (interleaved concurrent execution) instead of one turn fully
    # completing before the other begins.
    assert order == ["start", "end", "start", "end"]


async def test_session_token_budget_exceeded_returns_wrap_up_message_with_zero_llm_calls(
    client: httpx.AsyncClient, engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import settings
    from app.db.session import add_token_budget_used

    monkeypatch.setattr(settings, "session_token_budget", 10)
    await _prime_past_turn_zero(client, "sess-1")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as db:
        await add_token_budget_used(db, "sess-1", 10)  # already at the budget

    fake = _use_fake_provider([_response("unused — budget already exhausted")])

    response = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "one more question"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["reply"] == (
        f"This conversation has reached its limit — email the owner directly at "
        f"{settings.owner_contact_email}."
    )
    assert fake.calls == []  # zero LLM calls, per the acceptance criteria

    # The DB log stays complete even on the wrap-up path (4.3).
    session = await _get_session(engine, "sess-1")
    assert session is not None
    assert session.token_budget_used == 10  # unchanged — no LLM call was made to add to it


async def test_token_budget_used_matches_combined_loop_and_summarizer_usage(
    client: httpx.AsyncClient, engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance criterion (Issue #12): token_budget_used matches
    provider-reported usage, loop + summarizer combined.
    """
    from app.config import settings
    from app.db import session as db_session

    monkeypatch.setattr(settings, "window_high_tokens", 10)
    monkeypatch.setattr(settings, "window_low_tokens", 5)
    monkeypatch.setattr(
        db_session, "_session_factory", async_sessionmaker(engine, expire_on_commit=False)
    )

    await _prime_past_turn_zero(client, "sess-1")
    # 100 input (50 cached) + 20 output => (100-50) + 50*0.1 + 20 = 75 per turn.
    main_usage = Usage(input_tokens=100, output_tokens=20, cached_input_tokens=50)
    _use_fake_provider(
        [LLMResponse(text=f"reply {i}", usage=main_usage, finish_reason="stop") for i in range(6)]
    )
    # 40 input (uncached) + 10 output => 50, one summarizer call.
    summary_usage = Usage(input_tokens=40, output_tokens=10)
    summary_json = (
        '{"visitor_context": "chatting", "open_questions": [], '
        '"commitments": [], "notes": []}'
    )
    fake_summarizer = FakeProvider(
        responses=[LLMResponse(text=summary_json, usage=summary_usage, finish_reason="stop")]
    )
    app.dependency_overrides[get_summarizer_provider] = lambda: fake_summarizer

    for i in range(6):
        await client.post("/v1/chat", json={"session_id": "sess-1", "message": f"turn {i}"})

    session = await _get_session(engine, "sess-1")
    assert session is not None
    assert session.token_budget_used == 6 * 75 + 50


async def test_empty_session_id_returns_invalid_request_envelope(client: httpx.AsyncClient) -> None:
    _use_fake_provider([_response("unused")])

    response = await client.post("/v1/chat", json={"session_id": "", "message": "hello"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


async def test_user_message_persisted_even_when_provider_fails(client: httpx.AsyncClient) -> None:
    """4.3: the DB log is complete — a message that triggers an upstream
    failure must still be recorded, even though no reply exists for it.
    """
    await _prime_past_turn_zero(client, "sess-1")
    app.dependency_overrides[get_main_provider] = lambda: _RaisingProvider(
        UpstreamError(status_code=503, error_type="APIStatusError", message="boom")
    )

    failed = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "this triggers a failure"}
    )
    assert failed.status_code == 503

    fake = _use_fake_provider([_response("ok now")])
    recovered = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "second attempt"}
    )
    assert recovered.status_code == 200

    contents = [m.content for m in fake.calls[0]]
    assert "this triggers a failure" in contents
    assert "second attempt" in contents


async def test_empty_message_returns_invalid_request_envelope(client: httpx.AsyncClient) -> None:
    _use_fake_provider([_response("unused")])

    response = await client.post("/v1/chat", json={"session_id": "sess-1", "message": ""})

    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_request", "message": "The request was invalid."}
    }


async def test_missing_field_returns_invalid_request_envelope(client: httpx.AsyncClient) -> None:
    _use_fake_provider([_response("unused")])

    response = await client.post("/v1/chat", json={"session_id": "sess-1"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


async def test_upstream_error_returns_503_envelope(client: httpx.AsyncClient) -> None:
    await _prime_past_turn_zero(client, "sess-1")
    app.dependency_overrides[get_main_provider] = lambda: _RaisingProvider(
        UpstreamError(status_code=503, error_type="APIStatusError", message="boom")
    )

    response = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "hello"}
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "upstream_unavailable",
            "message": "The service is temporarily unavailable. Please try again.",
        }
    }


async def test_unhandled_exception_returns_500_envelope_without_leaking_details(
    client: httpx.AsyncClient,
) -> None:
    await _prime_past_turn_zero(client, "sess-1")
    app.dependency_overrides[get_main_provider] = lambda: _RaisingProvider(
        RuntimeError("sensitive internal detail that must never reach the client")
    )

    response = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "hello"}
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "An internal error occurred."}
    }
    assert "sensitive internal detail" not in response.text


async def test_health_endpoint(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok", "llm": "unchecked"}


async def test_openapi_schema_includes_chat_contract(client: httpx.AsyncClient) -> None:
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/v1/chat" in paths
    assert "post" in paths["/v1/chat"]
