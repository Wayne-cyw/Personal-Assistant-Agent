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

from app.db import session as db_session
from app.db.models import Base
from app.db.session import append_message, get_or_create_session, recent_messages


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


def test_no_sync_db_access_outside_session_module() -> None:
    """Grep-audit (Issue #3 acceptance criteria): no synchronous SQLAlchemy
    engine, no `sqlite3` import, anywhere under app/ except db/session.py.
    """
    repo_root = Path(__file__).parents[2]
    result = subprocess.run(
        ["grep", "-rlnE", "--include=*.py", r"sqlite3|create_engine\(", "app/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    matches = [line for line in result.stdout.splitlines() if line and line != "app/db/session.py"]
    assert matches == []
