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
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Portable JSON: generic JSON everywhere, rendered as JSONB on Postgres.
PortableJSON = JSON().with_variant(JSONB(), "postgresql")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


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
    contact_name: Mapped[str | None] = mapped_column(Text, default=None)
    contact_email: Mapped[str | None] = mapped_column(Text, default=None)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Booking(Base):
    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    slot_start_iso: Mapped[str] = mapped_column(String)
    slot_end_iso: Mapped[str] = mapped_column(String)
    timezone_name: Mapped[str] = mapped_column("timezone", String)
    contact_name: Mapped[str] = mapped_column(Text)
    contact_email: Mapped[str] = mapped_column(Text)
    gcal_event_id: Mapped[str | None] = mapped_column(String, default=None)
    status: Mapped[str] = mapped_column(String)  # tentative|cancelled
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class KBChunk(Base):
    __tablename__ = "kb_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_file: Mapped[str] = mapped_column(String)
    heading: Mapped[str | None] = mapped_column(String, default=None)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class RateLimit(Base):
    __tablename__ = "rate_limits"
    __table_args__ = (Index("ix_rate_limits_window_start", "window_start"),)

    # key: "sess:{id}" | "ip:{addr}" | "book:{id}"
    key: Mapped[str] = mapped_column(String, primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    count: Mapped[int] = mapped_column(Integer, default=0)
