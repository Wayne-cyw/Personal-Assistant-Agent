import asyncio
import subprocess
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.db import session as db_session
from app.db.models import Base
from app.db.session import (
    add_pinned_fact,
    add_token_budget_used,
    advance_summary,
    append_message,
    get_engine,
    get_or_create_session,
    messages_after,
    messages_up_to,
    overwrite_summary_content,
    recent_messages,
    set_visitor_info,
)


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine]:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = create_async_engine(db_url)
    db_session._register_sqlite_pragmas(engine)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def test_sqlite_pragmas_are_in_effect(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        result = await conn.execute(text("PRAGMA journal_mode"))
        assert result.scalar() == "wal"
        result = await conn.execute(text("PRAGMA foreign_keys"))
        assert result.scalar() == 1


async def test_get_or_create_session_creates_then_reuses(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        created = await get_or_create_session(db, "sess-1")
        assert created.id == "sess-1"

    async with session_factory() as db:
        fetched = await get_or_create_session(db, "sess-1")
        assert fetched.id == "sess-1"
        assert fetched.created_at == created.created_at


async def test_messages_do_not_leak_across_sessions(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        await get_or_create_session(db, "sess-a")
        await get_or_create_session(db, "sess-b")
        await append_message(db, "sess-a", "user", "hello from a")
        await append_message(db, "sess-b", "user", "hello from b")

    async with session_factory() as db:
        a_messages = await recent_messages(db, "sess-a", n=10)
        b_messages = await recent_messages(db, "sess-b", n=10)

    assert [m.content for m in a_messages] == ["hello from a"]
    assert [m.content for m in b_messages] == ["hello from b"]


async def test_recent_messages_returns_chronological_order_limited(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        await get_or_create_session(db, "sess-1")
        for i in range(5):
            await append_message(db, "sess-1", "user", f"message {i}")

    async with session_factory() as db:
        messages = await recent_messages(db, "sess-1", n=3)

    assert [m.content for m in messages] == ["message 2", "message 3", "message 4"]


async def test_concurrent_slow_write_does_not_block_concurrent_read(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A slow write in one task must not stall a concurrent read in another —
    the event loop must not be blocked by sync DB calls (4.7 rule 1).
    """

    async def slow_write() -> None:
        async with session_factory() as db:
            await get_or_create_session(db, "sess-slow")
            await asyncio.sleep(0.2)
            await append_message(db, "sess-slow", "user", "slow message")

    async def fast_read() -> float:
        start = time.monotonic()
        async with session_factory() as db:
            await get_or_create_session(db, "sess-fast")
            await recent_messages(db, "sess-fast", n=1)
        return time.monotonic() - start

    _, read_duration = await asyncio.gather(slow_write(), fast_read())
    assert read_duration < 0.1


async def test_postgres_pool_kwargs_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Postgres-only pool settings (4.7 rule 4) must reach create_async_engine
    unchanged; verified by spying on the call rather than introspecting the
    Pool object's private attributes. No live Postgres needed: constructing
    an asyncpg engine does not connect.
    """
    captured: dict[str, object] = {}

    def _spy(url: str, **kwargs: object) -> AsyncEngine:
        captured.update(kwargs)
        return create_async_engine(url, **kwargs)

    monkeypatch.setattr(db_session, "create_async_engine", _spy)
    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://user:pw@localhost/db")
    await db_session.dispose_engine()
    try:
        get_engine()
    finally:
        await db_session.dispose_engine()

    assert captured == {
        "pool_size": 5,
        "max_overflow": 5,
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }


async def test_sqlite_gets_no_postgres_pool_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _spy(url: str, **kwargs: object) -> AsyncEngine:
        captured.update(kwargs)
        return create_async_engine(url, **kwargs)

    monkeypatch.setattr(db_session, "create_async_engine", _spy)
    monkeypatch.setattr(settings, "database_url", "sqlite+aiosqlite:///:memory:")
    await db_session.dispose_engine()
    try:
        get_engine()
    finally:
        await db_session.dispose_engine()

    assert captured == {}


async def test_add_pinned_fact_appends_dedupes_and_caps(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        await get_or_create_session(db, "sess-1")
        assert await add_pinned_fact(db, "sess-1", "likes Rust", max_facts=2) is True
        # Case-insensitive duplicate of an existing fact is rejected.
        assert await add_pinned_fact(db, "sess-1", "Likes Rust", max_facts=2) is False
        assert await add_pinned_fact(db, "sess-1", "based in NYC", max_facts=2) is True
        # Cap reached: a third distinct fact is rejected.
        assert await add_pinned_fact(db, "sess-1", "hiring for backend", max_facts=2) is False

    async with session_factory() as db:
        row = await get_or_create_session(db, "sess-1")
        assert row.pinned_facts_json == ["likes Rust", "based in NYC"]


async def test_add_pinned_fact_unknown_session_raises(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        with pytest.raises(ValueError, match="unknown-session"):
            await add_pinned_fact(db, "unknown-session", "fact", max_facts=5)


async def test_messages_up_to_and_after_split_on_boundary(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        await get_or_create_session(db, "sess-1")
        ids = []
        for i in range(5):
            msg = await append_message(db, "sess-1", "user", f"message {i}")
            ids.append(msg.id)

    async with session_factory() as db:
        up_to = await messages_up_to(db, "sess-1", ids[2])
        after = await messages_after(db, "sess-1", ids[2])

    assert [m.content for m in up_to] == ["message 0", "message 1", "message 2"]
    assert [m.content for m in after] == ["message 3", "message 4"]


async def test_messages_after_none_returns_everything(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        await get_or_create_session(db, "sess-1")
        await append_message(db, "sess-1", "user", "only message")

    async with session_factory() as db:
        after = await messages_after(db, "sess-1", None)

    assert [m.content for m in after] == ["only message"]


async def test_advance_summary_sets_content_and_boundary(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        await get_or_create_session(db, "sess-1")
        await advance_summary(
            db,
            "sess-1",
            summary_json={"visitor_context": "recruiter"},
            summary_through_message_id=7,
        )

    async with session_factory() as db:
        row = await get_or_create_session(db, "sess-1")
        assert row.summary_json == {"visitor_context": "recruiter"}
        assert row.summary_through_message_id == 7


async def test_overwrite_summary_content_leaves_boundary_unchanged(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        await get_or_create_session(db, "sess-1")
        await advance_summary(
            db,
            "sess-1",
            summary_json={"visitor_context": "stale"},
            summary_through_message_id=7,
        )
        await overwrite_summary_content(db, "sess-1", summary_json={"visitor_context": "corrected"})

    async with session_factory() as db:
        row = await get_or_create_session(db, "sess-1")
        assert row.summary_json == {"visitor_context": "corrected"}
        assert row.summary_through_message_id == 7  # unchanged


async def test_set_visitor_info_overwrites_unconditionally(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Regression test: the tool-based save_visitor_info path (Issue #11)
    relies on set_visitor_info applying a new value even when one is already
    set — unlike Issue #9's regex-based turn-zero capture, which gates
    capture-once in the *caller* before ever reaching this function.
    """
    async with session_factory() as db:
        await get_or_create_session(db, "sess-1")
        await set_visitor_info(db, "sess-1", name="Priya")
        await set_visitor_info(db, "sess-1", name="Priya Patel")

    async with session_factory() as db:
        row = await get_or_create_session(db, "sess-1")
        assert row.visitor_name == "Priya Patel"


async def test_add_token_budget_used_accumulates_across_calls(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        await get_or_create_session(db, "sess-1")
        first_total = await add_token_budget_used(db, "sess-1", 100)
        second_total = await add_token_budget_used(db, "sess-1", 50)

    assert first_total == 100
    assert second_total == 150
    async with session_factory() as db:
        row = await get_or_create_session(db, "sess-1")
        assert row.token_budget_used == 150


async def test_add_token_budget_used_unknown_session_raises(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        with pytest.raises(ValueError, match="unknown-session"):
            await add_token_budget_used(db, "unknown-session", 10)


def test_no_sync_db_access_outside_session_module() -> None:
    """Grep-audit (Issue #3 acceptance criteria): no synchronous SQLAlchemy
    engine, no `sqlite3` import, and no dialect-specific SQL (PRAGMA / ON
    CONFLICT) anywhere under app/ except db/session.py.
    """
    repo_root = Path(__file__).parents[2]
    result = subprocess.run(
        ["grep", "-rlnE", "--include=*.py", r"sqlite3|create_engine\(|PRAGMA|ON CONFLICT", "app/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    matches = [line for line in result.stdout.splitlines() if line and line != "app/db/session.py"]
    assert matches == []
