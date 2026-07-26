import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import DateTime, String, Table, inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import session as db_session
from app.db.models import Base, BookingState, SessionRow

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


class _RenamedDateTimeColumnModel(Base):
    """Test-only model regression-testing the naive-datetime guard against a
    *DateTime* column whose Python attribute name differs from its DB column
    name (BookingState/Booking rename `timezone` to `timezone_name`, but
    that column is a String, so it never exercised this code path). Before
    the fix, the guard indexed InstanceState.attrs (keyed by the Python
    attribute name) using the Core column's DB name, raising KeyError on any
    write to a renamed DateTime column.
    """

    __tablename__ = "_test_renamed_datetime_column_model"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    happened_at: Mapped[datetime] = mapped_column("occurred_at", DateTime(timezone=True))


_renamed_datetime_table = cast(Table, _RenamedDateTimeColumnModel.__table__)

# Deliberately detached from Base.metadata's create_all sweep: every other
# fixture/test in this file (and the live-Postgres parity test) calls
# `Base.metadata.create_all`, which would otherwise create/drop this stray
# test-only table against every engine, including a real external Postgres.
# The one test below creates/drops it explicitly instead.
Base.metadata.remove(_renamed_datetime_table)


async def test_naive_datetime_guard_handles_renamed_datetime_column(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(_renamed_datetime_table.create)
    try:
        async with session_factory() as db:
            db.add(_RenamedDateTimeColumnModel(id="x", happened_at=datetime.now(UTC)))
            await db.commit()  # must not raise KeyError (regression for the renamed-column bug)

        async with session_factory() as db:
            naive_row = _RenamedDateTimeColumnModel(id="y", happened_at=datetime.now())  # noqa: DTZ005
            db.add(naive_row)
            with pytest.raises(ValueError, match="timezone-aware"):
                await db.commit()
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(_renamed_datetime_table.drop)


async def test_naive_datetime_guard_handles_renamed_string_column(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """BookingState.timezone_name maps to DB column "timezone" — confirms an
    ordinary write through a renamed *non-datetime* column still works.
    """
    async with session_factory() as db:
        db.add(SessionRow(id="sess-1"))
        await db.flush()
        db.add(BookingState(session_id="sess-1", step="proposing", timezone_name="UTC"))
        await db.commit()  # must not raise


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


@pytest.mark.skipif(
    not os.environ.get("POSTGRES_TEST_URL"),
    reason="set POSTGRES_TEST_URL (see README) to run against a local Postgres",
)
async def test_tables_land_as_postgres_native_types() -> None:
    """Acceptance criteria (Issue #3): on Postgres, portable types render as
    the expected native types — JSONB, BYTEA, TIMESTAMPTZ. Skipped unless
    POSTGRES_TEST_URL is set (no Postgres available by default; the CI
    parity job lands in Issue #31).
    """
    pg_engine = create_async_engine(os.environ["POSTGRES_TEST_URL"])
    try:
        async with pg_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        def _inspect(sync_conn: Connection) -> dict[str, str]:
            insp = inspect(sync_conn)
            columns = {c["name"]: str(c["type"]) for c in insp.get_columns("sessions")}
            columns.update(
                {f"kb_chunks.{c['name']}": str(c["type"]) for c in insp.get_columns("kb_chunks")}
            )
            return columns

        async with pg_engine.connect() as conn:
            types = await conn.run_sync(_inspect)

        assert "JSONB" in types["pinned_facts_json"]
        assert "TIMESTAMP" in types["created_at"] and "WITH TIME ZONE" in types["created_at"]
        assert "BYTEA" in types["kb_chunks.embedding"]

        async with pg_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    finally:
        await pg_engine.dispose()
