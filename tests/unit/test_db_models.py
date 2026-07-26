from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db import session as db_session
from app.db.models import Base, SessionRow

EXPECTED_TABLES = {
    "sessions",
    "messages",
    "booking_states",
    "bookings",
    "kb_chunks",
    "rate_limits",
}


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


async def test_all_six_tables_created(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        table_names = await conn.run_sync(
            lambda sync_conn: set(inspect(sync_conn).get_table_names())
        )
    assert EXPECTED_TABLES <= table_names


async def test_expected_indexes_created(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        indexes = await conn.run_sync(
            lambda sync_conn: {
                (table, ix["name"])
                for table in EXPECTED_TABLES
                for ix in inspect(sync_conn).get_indexes(table)
            }
        )
    index_names = {name for _table, name in indexes}
    assert "ix_messages_session_id_id" in index_names
    assert "ix_messages_created_at" in index_names
    assert "ix_sessions_last_seen_at" in index_names
    assert "ix_rate_limits_window_start" in index_names


async def test_naive_datetime_rejected_on_insert(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        row = SessionRow(id="sess-naive", created_at=datetime.now())  # noqa: DTZ005 (deliberately naive)
        db.add(row)
        with pytest.raises(ValueError, match="timezone-aware"):
            await db.commit()


async def test_aware_datetime_accepted_on_insert(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        row = SessionRow(id="sess-aware")  # uses the _utcnow() default, tz-aware
        db.add(row)
        await db.commit()  # must not raise


async def test_json_columns_round_trip(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as db:
        row = SessionRow(id="sess-json", pinned_facts_json=[{"fact": "likes rust"}])
        db.add(row)
        await db.commit()

    async with session_factory() as db:
        fetched = await db.get(SessionRow, "sess-json")
        assert fetched is not None
        assert fetched.pinned_facts_json == [{"fact": "likes rust"}]


async def test_embedding_stored_as_bytes(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    from app.db.models import KBChunk

    async with session_factory() as db:
        chunk = KBChunk(
            source_file="bio.md", heading="Intro", content="hi", embedding=b"\x00\x01\x02"
        )
        db.add(chunk)
        await db.commit()

    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT embedding FROM kb_chunks WHERE source_file = 'bio.md'")
        )
        stored = result.scalar()
        assert stored == b"\x00\x01\x02"
