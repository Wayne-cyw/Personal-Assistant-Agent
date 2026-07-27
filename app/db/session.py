"""Engine creation from DATABASE_URL, table creation on startup, and
request-scoped session helpers.

This is the one module allowed to know about dialect differences (Engineering
Guide 4.7 rule 3): Postgres pooling options and SQLite pragmas both live here,
gated on `engine.dialect.name`. Everything else in the codebase talks to the
DB only through the repository functions below, using portable SQL.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.db.models import Base, Message, SessionRow

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Create (once) and return the process-wide async engine.

    Postgres-only pool kwargs and SQLite-only pragmas are applied based on
    the resolved dialect name, checked once here at construction time.
    """
    global _engine
    if _engine is None:
        engine_kwargs: dict[str, object] = {}
        if settings.database_url.startswith("postgresql"):
            engine_kwargs.update(
                pool_size=5,
                max_overflow=5,
                pool_pre_ping=True,
                pool_recycle=300,
            )
        engine = create_async_engine(settings.database_url, **engine_kwargs)
        if engine.dialect.name == "sqlite":
            _register_sqlite_pragmas(engine)
        _engine = engine
    return _engine


def _register_sqlite_pragmas(engine: AsyncEngine) -> None:
    """Dev/test-side only (4.7 rule 5): WAL, synchronous, busy_timeout, FKs.

    SQLAlchemy's `connect` event is DBAPI-level, so it attaches to the sync
    engine `AsyncEngine` wraps — this is the one legitimate touch of a "sync"
    object in this module; it configures the raw DBAPI connection, it does
    not perform sync I/O on the request path.
    """
    from sqlalchemy import event

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection: object, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request."""
    async with get_session_factory()() as session:
        yield session


async def init_db(retries: int = 5, base_delay: float = 0.5) -> None:
    """Create all tables, retrying with backoff (4.7 rule 8) so a serverless
    Postgres waking up or a brief DB outage during deploy doesn't crash-loop
    the app.
    """
    engine = get_engine()
    for attempt in range(retries):
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            return
        except (OSError, SQLAlchemyError):
            if attempt == retries - 1:
                raise
            await asyncio.sleep(base_delay * 2**attempt)


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


# --- Repository functions -------------------------------------------------
# Portable SQL only (4.7 rule 3): no dialect-specific SQL below this line.


async def get_or_create_session(db: AsyncSession, session_id: str) -> SessionRow:
    row = await db.get(SessionRow, session_id)
    if row is not None:
        row.last_seen_at = datetime.now(UTC)
        await db.commit()
        return row
    row = SessionRow(id=session_id)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def append_message(
    db: AsyncSession,
    session_id: str,
    role: str,
    content: str,
    *,
    response_type: str | None = None,
    turn_tag: str | None = None,
    tool_name: str | None = None,
    tool_payload: dict[str, object] | None = None,
) -> Message:
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
        response_type=response_type,
        turn_tag=turn_tag,
        tool_name=tool_name,
        tool_payload_json=tool_payload,
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


async def recent_messages(db: AsyncSession, session_id: str, n: int) -> list[Message]:
    """The last `n` messages for `session_id`, oldest first (chronological
    order — the order a caller assembling LLM context expects).
    """
    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.id.desc())
        .limit(n)
    )
    result = await db.execute(stmt)
    return list(reversed(result.scalars().all()))


async def set_visitor_info(
    db: AsyncSession,
    session_id: str,
    *,
    name: str | None = None,
    linkedin: str | None = None,
) -> SessionRow:
    """Set visitor_name/visitor_linkedin on the session row (Issue #9).
    Only overwrites a field when a non-None value is passed — callers decide
    whether a field is already set before calling this (e.g. capture-once
    semantics), this function does not check.
    """
    row = await db.get(SessionRow, session_id)
    assert row is not None, f"set_visitor_info called for unknown session_id={session_id!r}"
    if name is not None:
        row.visitor_name = name
    if linkedin is not None:
        row.visitor_linkedin = linkedin
    await db.commit()
    await db.refresh(row)
    return row
