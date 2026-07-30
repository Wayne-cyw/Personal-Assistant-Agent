"""Engine creation from DATABASE_URL, table creation on startup, and
request-scoped session helpers.

This is the one module allowed to know about dialect differences (Engineering
Guide 4.7 rule 3): Postgres pooling options and SQLite pragmas both live here,
gated on `engine.dialect.name`. Everything else in the codebase talks to the
DB only through the repository functions below, using portable SQL.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.booking.state import BookingState
from app.booking.state import Step as BookingStep
from app.config import settings
from app.db.models import Base, Booking, Message, SessionRow
from app.db.models import BookingState as BookingStateRow

logger = logging.getLogger(__name__)

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
    Only overwrites a field when a non-None value is passed. Unlike Issue
    #9's regex-based turn-zero capture (which the caller gates to
    capture-once, since a false positive there is unrecoverable), the
    tool-based `save_visitor_info` path (Issue #11) calls this unconditionally
    on every volunteered value — a deliberate model tool call is a visitor
    correction, not a guess, so last-write-wins is the correct semantics
    there.
    """
    row = await db.get(SessionRow, session_id)
    if row is None:
        raise ValueError(f"set_visitor_info called for unknown session_id={session_id!r}")
    if name is not None:
        row.visitor_name = name
    if linkedin is not None:
        row.visitor_linkedin = linkedin
    await db.commit()
    await db.refresh(row)
    return row


async def add_pinned_fact(
    db: AsyncSession, session_id: str, fact: str, *, max_facts: int
) -> bool:
    """Append `fact` to the session's pinned_facts_json (Issue #11),
    deduplicated case-insensitively and capped at `max_facts`. Returns
    whether the fact was actually added (False if it was a duplicate or the
    cap was already reached) so the caller (the save_visitor_info tool) can
    report the outcome back to the model.
    """
    row = await db.get(SessionRow, session_id)
    if row is None:
        raise ValueError(f"add_pinned_fact called for unknown session_id={session_id!r}")
    facts: list[str] = [str(existing) for existing in (row.pinned_facts_json or [])]
    if any(existing.strip().lower() == fact.strip().lower() for existing in facts):
        return False
    if len(facts) >= max_facts:
        return False
    facts.append(fact)
    row.pinned_facts_json = cast(list[object], facts)
    await db.commit()
    return True


async def messages_up_to(
    db: AsyncSession, session_id: str, through_message_id: int
) -> list[Message]:
    """All messages for `session_id` with id <= through_message_id, oldest
    first — the exact raw range a summary claims to cover (Issue #11), used
    by `reload_and_reconcile` to regenerate that range from source data.
    """
    stmt = (
        select(Message)
        .where(Message.session_id == session_id, Message.id <= through_message_id)
        .order_by(Message.id.asc())
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def messages_after(
    db: AsyncSession, session_id: str, after_message_id: int | None
) -> list[Message]:
    """All messages for `session_id` with id > after_message_id (or every
    message, if None), oldest first — the live rolling window (Issue #11):
    everything not yet folded into the summary.
    """
    stmt = select(Message).where(Message.session_id == session_id)
    if after_message_id is not None:
        stmt = stmt.where(Message.id > after_message_id)
    stmt = stmt.order_by(Message.id.asc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def advance_summary(
    db: AsyncSession,
    session_id: str,
    *,
    summary_json: dict[str, object],
    summary_through_message_id: int,
) -> None:
    """Normal eviction path (Issue #11): store the newly-merged summary and
    advance the coverage boundary to the last evicted message id.
    """
    row = await db.get(SessionRow, session_id)
    if row is None:
        raise ValueError(f"advance_summary called for unknown session_id={session_id!r}")
    row.summary_json = summary_json
    row.summary_through_message_id = summary_through_message_id
    await db.commit()


async def add_token_budget_used(db: AsyncSession, session_id: str, tokens: int) -> int:
    """Add `tokens` (a cost-weighted count — app/agent/tokens.py's
    effective_tokens) to sessions.token_budget_used (Issue #12), returning
    the new total. Called from every real LLM call site that spends
    against a session's budget: the main loop (app/api/chat.py, under its
    own session_turn_lock(session_id)) and every summarizer call
    (app/agent/memory.py's run_eviction/run_reconciliation, background
    tasks running under a deliberately *different* lock namespace,
    session_turn_lock(f"mem:{{session_id}}") — see _memory_lock_key's
    docstring for why). Those two call sites are not serialized against
    each other, so this is an atomic single-statement `SET x = x + n`
    UPDATE (4.7 rule 3's pattern for exactly this class of counter) rather
    than a read-modify-write in Python — the latter would lose an
    increment whenever a live turn's usage and a background summarizer's
    usage get charged concurrently for the same session.
    """
    stmt = (
        update(SessionRow)
        .where(SessionRow.id == session_id)
        .values(token_budget_used=SessionRow.token_budget_used + tokens)
        .returning(SessionRow.token_budget_used)
    )
    result = await db.execute(stmt)
    row = result.first()
    if row is None:
        raise ValueError(f"add_token_budget_used called for unknown session_id={session_id!r}")
    await db.commit()
    return int(row[0])


async def increment_eviction_count(db: AsyncSession, session_id: str) -> int:
    """Bump sessions.eviction_count by one after a real eviction (Issue
    #11), returning the new value — drives the SUMMARY_AUDIT_INTERVAL
    scheduling check in app/agent/memory.py's run_eviction. A dedicated
    function rather than a direct ORM mutation in memory.py, per this
    module's own rule that every DB write goes through a named repo
    function here. Atomic `SET x = x + 1`, same as add_token_budget_used —
    every current caller already runs under the "mem:" lock namespace, so
    this isn't currently exposed to a cross-lock-domain race, but there's
    no reason for this counter to be less safe than that one.
    """
    stmt = (
        update(SessionRow)
        .where(SessionRow.id == session_id)
        .values(eviction_count=SessionRow.eviction_count + 1)
        .returning(SessionRow.eviction_count)
    )
    result = await db.execute(stmt)
    row = result.first()
    if row is None:
        raise ValueError(f"increment_eviction_count called for unknown session_id={session_id!r}")
    await db.commit()
    return int(row[0])


async def overwrite_summary_content(
    db: AsyncSession, session_id: str, *, summary_json: dict[str, object]
) -> None:
    """Reconciliation path (Issue #11, `reload_and_reconcile`): correct the
    summary's *content* without touching summary_through_message_id — no new
    coverage was added, the existing coverage was regenerated from source.
    """
    row = await db.get(SessionRow, session_id)
    if row is None:
        raise ValueError(f"overwrite_summary_content called for unknown session_id={session_id!r}")
    row.summary_json = summary_json
    await db.commit()


async def load_booking_state(db: AsyncSession, session_id: str) -> BookingState:
    """Load the booking state machine's state for `session_id` (Issue
    #18). No row yet (the common case — most sessions never book) returns
    a fresh, unpersisted default (step=idle) rather than raising; the row
    only starts existing once save_booking_state is first called for this
    session.
    """
    row = await db.get(BookingStateRow, session_id)
    if row is None:
        return BookingState(session_id=session_id)
    return BookingState(
        session_id=row.session_id,
        step=BookingStep(row.step),
        timezone_name=row.timezone_name,
        proposed_slots_json=cast(list[dict[str, object]] | None, row.proposed_slots_json),
        selected_slot_json=row.selected_slot_json,
        hold_expires_at=_reattach_utc(row.hold_expires_at),
        proposal_rounds=row.proposal_rounds,
        contact_name=row.contact_name,
        contact_email=row.contact_email,
        excluded_slots_json=cast(list[dict[str, object]] | None, row.excluded_slots_json),
        confirmation_declines=row.confirmation_declines,
    )


def _reattach_utc(value: datetime | None) -> datetime | None:
    """SQLite has no native timezone-aware storage — a tz-aware datetime
    written via save_booking_state (the ORM's before_insert/before_update
    guard in app/db/models.py rejects writing a naive one, so it's always
    UTC going in) comes back naive on a plain reload, on this dialect only
    (4.7 rule 2's documented SQLite quirk). Silently reattaching UTC here
    matters more for hold_expires_at than most datetime fields in this
    codebase: it's the one field a future soft-hold expiry check
    (app/booking/holds.py, Issue #21) will compare against
    datetime.now(UTC) — comparing aware to naive raises TypeError, and
    that comparison would work on Postgres (TIMESTAMPTZ preserves tzinfo)
    while silently breaking in SQLite dev/test, exactly the dialect-parity
    trap 4.7 warns about.
    """
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


async def save_booking_state(db: AsyncSession, state: BookingState) -> None:
    """Persist `state` (Issue #18) — insert-or-update, since a session's
    booking_states row may not exist yet the first time the flow starts.
    """
    row = await db.get(BookingStateRow, state.session_id)
    if row is None:
        row = BookingStateRow(session_id=state.session_id, step=state.step.value)
        db.add(row)
    else:
        row.step = state.step.value
    row.timezone_name = state.timezone_name
    row.proposed_slots_json = cast("list[object] | None", state.proposed_slots_json)
    row.selected_slot_json = state.selected_slot_json
    row.hold_expires_at = state.hold_expires_at
    row.proposal_rounds = state.proposal_rounds
    row.contact_name = state.contact_name
    row.contact_email = state.contact_email
    row.excluded_slots_json = cast("list[object] | None", state.excluded_slots_json)
    row.confirmation_declines = state.confirmation_declines
    await db.commit()


async def active_holds(
    db: AsyncSession, *, exclude_session_id: str
) -> list[tuple[datetime, datetime]]:
    """Every *other* session's currently-held slot (Issue #21, 4.5's
    slot_selected step) — soft holds are DB-only, never written to the
    calendar, so calendar_find_slots (app/tools/registry.py) has to treat
    them as additional busy time itself when proposing to a different
    session. Expiry is checked lazily as part of the query
    (`hold_expires_at > now`) rather than via any separate cleanup job — an
    expired hold simply stops being returned, no row is deleted or reset.
    """
    now = datetime.now(UTC)
    stmt = select(BookingStateRow).where(
        BookingStateRow.session_id != exclude_session_id,
        BookingStateRow.hold_expires_at.is_not(None),
        BookingStateRow.hold_expires_at > now,
        BookingStateRow.selected_slot_json.is_not(None),
    )
    result = await db.execute(stmt)
    holds: list[tuple[datetime, datetime]] = []
    for row in result.scalars():
        slot = row.selected_slot_json
        try:
            if not isinstance(slot, dict):
                raise TypeError(f"selected_slot_json was not a dict: {type(slot).__name__}")
            holds.append(
                (
                    datetime.fromisoformat(str(slot["start_iso"])),
                    datetime.fromisoformat(str(slot["end_iso"])),
                )
            )
        except (KeyError, ValueError, TypeError):
            # A single corrupted row must not fail this query for every
            # session — active_holds() runs on every calendar_find_slots
            # call, so an unhandled exception here would break slot
            # proposals process-wide, not just for the offending session.
            logger.error(
                "active_holds: session %s has a malformed selected_slot_json, skipping: %r",
                row.session_id,
                slot,
            )
    return holds


async def create_booking(
    db: AsyncSession,
    *,
    session_id: str,
    slot_start_iso: str,
    slot_end_iso: str,
    timezone_name: str,
    contact_name: str,
    contact_email: str,
    gcal_event_id: str,
) -> Booking:
    """Insert the `bookings` row for a newly-created tentative event (Issue
    #22) — the durable, permanent record, independent of `booking_states`
    (which tracks the in-progress flow and can be overwritten/reset;
    `bookings` never is). `status` starts "tentative", matching the real
    Google Calendar event's own status (app/tools/calendar.py's
    `create_event`); nothing in v1 ever transitions it to "cancelled" —
    that's out of scope here (rescheduling/cancellation is by email, per
    the issue text).
    """
    booking = Booking(
        session_id=session_id,
        slot_start_iso=slot_start_iso,
        slot_end_iso=slot_end_iso,
        timezone_name=timezone_name,
        contact_name=contact_name,
        contact_email=contact_email,
        gcal_event_id=gcal_event_id,
        status="tentative",
    )
    db.add(booking)
    await db.commit()
    try:
        await db.refresh(booking)
    except Exception:
        # The commit above already succeeded and durably created the row —
        # `id` was already populated on `booking` at flush time (standard
        # SQLAlchemy behavior for an autoincrement PK, independent of this
        # refresh) and `created_at` was set client-side by the column's
        # Python default before the insert, so a refresh failure here
        # doesn't leave `booking` missing anything a caller actually reads.
        # Letting it propagate instead would make create_booking() look
        # like it failed even though the row is committed — which matters
        # to app/tools/registry.py's calendar_create_booking: it decides
        # whether to attempt a compensating delete of an orphaned bookings
        # row based on whether this function returned a Booking at all.
        logger.warning(
            "create_booking: row committed for session %s but refresh failed — continuing "
            "with the pre-commit object",
            session_id,
        )
    return booking
