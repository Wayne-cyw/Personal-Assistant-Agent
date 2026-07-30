"""SQLAlchemy models (Engineering Guide 4.7).

Dialect-neutral: every column type here means the same thing on SQLite (dev/
tests) and Postgres (prod) per 4.7 rule 2. All dialect-specific behavior
(engine options, pragmas, pooling) lives in session.py, not here.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    event,
    inspect,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Mapper, mapped_column

# Portable JSON: generic JSON everywhere, rendered as JSONB on Postgres.
PortableJSON = JSON().with_variant(JSONB(), "postgresql")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


@event.listens_for(Base, "before_insert", propagate=True)
@event.listens_for(Base, "before_update", propagate=True)
def _reject_naive_datetimes(mapper: Mapper[object], _connection: object, target: object) -> None:
    """Naive datetimes are banned (4.7 rule 2): SQLite silently tolerates
    them but Postgres comparisons will bite. SQLAlchemy's DateTime(timezone=
    True) does not enforce this on its own, so it's enforced here instead.

    Only attributes *changed in this flush* are checked. SQLite has no
    native timezone-aware storage, so a previously-written aware datetime
    can come back naive after a plain reload; re-validating unchanged
    columns on every update would reject that harmless round-trip instead
    of catching an actual newly-written naive value.

    Scope: this is an ORM unit-of-work event, so it only fires for writes
    that go through the ORM (session.add/flush) — exactly what every
    function in this module uses. It does NOT fire for Core-level
    `session.execute(insert(...)/update(...))` statements. 4.7 rule 3
    mandates the future rate-limit counter update use exactly such a Core
    upsert statement (single round trip, insert-else-update, with a
    RETURNING clause) for atomicity — whoever implements that write is
    responsible for constructing `window_start` as timezone-aware by hand,
    since this guard will not catch a mistake there.
    """
    state = inspect(target)
    assert state is not None
    for prop in mapper.column_attrs:
        column = prop.columns[0]
        if not (isinstance(column.type, DateTime) and column.type.timezone):
            continue
        if not state.attrs[prop.key].history.has_changes():
            continue
        value = getattr(target, prop.key)
        if isinstance(value, datetime) and value.tzinfo is None:
            raise ValueError(
                f"{type(target).__name__}.{prop.key} must be a "
                "timezone-aware datetime, got a naive one"
            )


class SessionRow(Base):
    __tablename__ = "sessions"
    __table_args__ = (Index("ix_sessions_last_seen_at", "last_seen_at"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    visitor_name: Mapped[str | None] = mapped_column(Text, default=None)
    visitor_linkedin: Mapped[str | None] = mapped_column(Text, default=None)
    pinned_facts_json: Mapped[list[object] | None] = mapped_column(PortableJSON, default=None)
    summary_json: Mapped[dict[str, object] | None] = mapped_column(PortableJSON, default=None)
    summary_through_message_id: Mapped[int | None] = mapped_column(Integer, default=None)
    # Counts batched high-water-mark evictions (Issue #11), not individual
    # turns — drives the SUMMARY_AUDIT_INTERVAL drift-audit canary (4.3).
    eviction_count: Mapped[int] = mapped_column(Integer, default=0)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    token_budget_used: Mapped[int] = mapped_column(Integer, default=0)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_session_id_id", "session_id", "id"),
        Index("ix_messages_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    role: Mapped[str] = mapped_column(String)  # user|assistant|tool
    content: Mapped[str] = mapped_column(Text)
    # response_type: message|refusal|booking_*
    response_type: Mapped[str | None] = mapped_column(String, default=None)
    # turn_tag: on_topic|off_topic|refused|abusive
    turn_tag: Mapped[str | None] = mapped_column(String, default=None)
    tool_name: Mapped[str | None] = mapped_column(String, default=None)
    tool_payload_json: Mapped[dict[str, object] | None] = mapped_column(PortableJSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class BookingState(Base):
    __tablename__ = "booking_states"

    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), primary_key=True)
    step: Mapped[str] = mapped_column(String)
    timezone_name: Mapped[str | None] = mapped_column("timezone", String, default=None)
    proposed_slots_json: Mapped[list[object] | None] = mapped_column(PortableJSON, default=None)
    selected_slot_json: Mapped[dict[str, object] | None] = mapped_column(PortableJSON, default=None)
    hold_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    proposal_rounds: Mapped[int] = mapped_column(Integer, default=0)
    # Every slot offered and then superseded across the whole negotiation
    # (Issue #20 review fix) — see app.booking.state.BookingState's own
    # field docstring for why this can't just be re-derived from
    # proposed_slots_json (which only ever holds the current round).
    excluded_slots_json: Mapped[list[object] | None] = mapped_column(PortableJSON, default=None)
    contact_name: Mapped[str | None] = mapped_column(Text, default=None)
    contact_email: Mapped[str | None] = mapped_column(Text, default=None)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Booking(Base):
    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    slot_start_iso: Mapped[str] = mapped_column(String)
    slot_end_iso: Mapped[str] = mapped_column(String)
    timezone_name: Mapped[str] = mapped_column("timezone", String)
    contact_name: Mapped[str] = mapped_column(Text)
    contact_email: Mapped[str] = mapped_column(Text)
    gcal_event_id: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)  # tentative|cancelled
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class KBChunk(Base):
    __tablename__ = "kb_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_file: Mapped[str] = mapped_column(String)
    heading: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class RateLimit(Base):
    __tablename__ = "rate_limits"
    __table_args__ = (Index("ix_rate_limits_window_start", "window_start"),)

    # key: "sess:{id}" | "ip:{addr}" | "book:{id}"
    key: Mapped[str] = mapped_column(String, primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    count: Mapped[int] = mapped_column(Integer, default=0)
