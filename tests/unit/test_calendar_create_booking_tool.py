"""calendar_create_booking tool handler (Issue #22): the confirmation ->
free/busy re-check -> event creation finale, including the race-condition
guard (a lot can happen between proposing a slot and the visitor confirming
it).
"""

from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.agent.providers.base import ToolCall
from app.agent.providers.fake import FakeProvider
from app.booking.state import BookingState, Step
from app.config import settings
from app.db.models import Base
from app.db.session import get_or_create_session, load_booking_state, save_booking_state
from app.tools.calendar import Attendee, BusyInterval, CalendarError
from app.tools.context import ToolContext
from app.tools.fake_calendar import FakeCalendar
from app.tools.registry import execute_tool, tool_defs_for_step

SID = "sess-1"
SLOT = {
    "slot_id": "s1",
    "start_iso": "2026-08-03T09:00:00-04:00",
    "end_iso": "2026-08-03T09:30:00-04:00",
    "label": "Mon Aug 3, 9:00-9:30 AM EDT",
}


class _CreateEventFailingCalendar:
    """get_free_busy always reports free; create_event always fails --
    isolates the "creation_failed" path from the "taken at re-check" path,
    which FakeCalendar's single fail_with flag can't do (it fails every
    method uniformly).
    """

    async def get_free_busy(self, _start: datetime, _end: datetime) -> list[BusyInterval]:
        return []

    async def create_event(
        self, start: datetime, end: datetime, attendee: Attendee, description: str
    ) -> str:
        raise CalendarError("event creation failed")

    async def delete_event(self, event_id: str) -> None:
        raise NotImplementedError


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine]:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db(engine: AsyncEngine) -> AsyncGenerator[AsyncSession]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        await get_or_create_session(session, SID)
        yield session


def _confirmed_state(**overrides: object) -> BookingState:
    defaults: dict[str, object] = dict(
        session_id=SID,
        step=Step.CONFIRMED,
        timezone_name="America/Toronto",
        selected_slot_json=SLOT,
        contact_name="Priya Patel",
        contact_email="priya@example.com",
    )
    defaults.update(overrides)
    return BookingState(**defaults)  # type: ignore[arg-type]


def _context(db: AsyncSession, *, calendar_client: object, affirmative: bool = True) -> ToolContext:
    return ToolContext(
        db=db,
        session_id=SID,
        summarizer_provider=FakeProvider(responses=[]),
        calendar_client=calendar_client,  # type: ignore[arg-type]
        confirmation_is_affirmative=affirmative,
    )


def _call() -> ToolCall:
    return ToolCall(id="call_1", name="calendar_create_booking", arguments={})


# --- tool gating --------------------------------------------------------------


def test_calendar_create_booking_offered_only_from_confirmed() -> None:
    for step in Step:
        names = [t.name for t in tool_defs_for_step(step)]
        if step is Step.CONFIRMED:
            assert "calendar_create_booking" in names
        else:
            assert "calendar_create_booking" not in names


# --- happy path ----------------------------------------------------------------


async def test_happy_path_creates_event_and_booking_row(db: AsyncSession) -> None:
    await save_booking_state(db, _confirmed_state())
    calendar = FakeCalendar()
    context = _context(db, calendar_client=calendar)

    result = await execute_tool(_call(), context)

    assert "error" not in result
    assert "taken" not in result
    assert result["slot"] == SLOT
    assert result["timezone"] == "America/Toronto"
    assert settings.owner_contact_email in str(result["next_steps"])
    assert isinstance(result["booking_id"], int)

    assert len(calendar.created_events) == 1
    created = next(iter(calendar.created_events.values()))
    assert created.attendee == Attendee(name="Priya Patel", email="priya@example.com")

    state = await load_booking_state(db, SID)
    assert state.step is Step.BOOKING_CREATED


async def test_happy_path_schedules_an_owner_notification(db: AsyncSession) -> None:
    await save_booking_state(db, _confirmed_state())
    context = _context(db, calendar_client=FakeCalendar())

    await execute_tool(_call(), context)

    assert len(context.pending_owner_notifications) == 1
    subject, body = context.pending_owner_notifications[0]
    assert "booking" in subject.lower()
    assert "Priya Patel" in body
    assert "priya@example.com" in body


# --- affirmative gating (defense in depth) ------------------------------------


async def test_refuses_when_not_affirmative(db: AsyncSession) -> None:
    await save_booking_state(db, _confirmed_state())
    calendar = FakeCalendar()
    context = _context(db, calendar_client=calendar, affirmative=False)

    result = await execute_tool(_call(), context)

    assert result["error"] == "not_confirmed"
    assert calendar.created_events == {}
    state = await load_booking_state(db, SID)
    assert state.step is Step.CONFIRMED  # unchanged


@pytest.mark.parametrize(
    "step",
    [
        Step.IDLE,
        Step.INTENT_DETECTED,
        Step.SLOTS_PROPOSED,
        Step.SLOT_SELECTED,
        Step.CONTACT_INFO_COLLECTED,
    ],
)
async def test_refuses_outside_confirmed(db: AsyncSession, step: Step) -> None:
    await save_booking_state(db, BookingState(session_id=SID, step=step))
    context = _context(db, calendar_client=FakeCalendar())

    result = await execute_tool(_call(), context)

    assert "error" in result
    state = await load_booking_state(db, SID)
    assert state.step is step  # unchanged


# --- race-condition guard ------------------------------------------------------


async def test_slot_taken_at_recheck_bounces_back_to_slots_proposed(db: AsyncSession) -> None:
    await save_booking_state(
        db,
        _confirmed_state(
            proposed_slots_json=[SLOT, {"slot_id": "s2", "start_iso": "x", "end_iso": "y"}],
            proposal_rounds=1,
        ),
    )
    calendar = FakeCalendar(
        busy=[
            BusyInterval(
                start=datetime.fromisoformat(str(SLOT["start_iso"])),
                end=datetime.fromisoformat(str(SLOT["end_iso"])),
            )
        ]
    )
    context = _context(db, calendar_client=calendar)

    result = await execute_tool(_call(), context)

    assert result["taken"] is True
    assert calendar.created_events == {}
    state = await load_booking_state(db, SID)
    assert state.step is Step.SLOTS_PROPOSED
    assert state.selected_slot_json is None
    assert state.proposed_slots_json == [{"slot_id": "s2", "start_iso": "x", "end_iso": "y"}]


async def test_event_creation_failure_returns_structured_error(db: AsyncSession) -> None:
    await save_booking_state(db, _confirmed_state())
    context = _context(db, calendar_client=_CreateEventFailingCalendar())

    result = await execute_tool(_call(), context)

    assert result["error"] == "creation_failed"
    state = await load_booking_state(db, SID)
    assert state.step is Step.CONFIRMED  # unchanged; visitor can retry


async def test_db_write_failure_after_event_creation_deletes_the_orphaned_event(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: a review pass found that if create_booking (the DB
    write) fails *after* calendar_client.create_event already succeeded, the
    handler left a real orphaned event on the calendar with no bookings row
    and booking_state stuck at confirmed -- a retry would then see the
    orphaned event as busy at the free/busy re-check and wrongly tell the
    visitor their own just-created slot was taken by someone else. The fix
    wraps the DB write in try/except and attempts a compensating
    delete_event() call. FakeCalendar.delete_event() pops the event out of
    created_events, so asserting created_events == {} after the failure
    proves the compensating delete actually ran (not just that an exception
    was swallowed).
    """
    await save_booking_state(db, _confirmed_state())
    calendar = FakeCalendar()
    context = _context(db, calendar_client=calendar)

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("db write failed")

    monkeypatch.setattr("app.tools.registry.create_booking", _raise)

    result = await execute_tool(_call(), context)

    assert result["error"] == "creation_failed"
    assert calendar.created_events == {}
    state = await load_booking_state(db, SID)
    assert state.step is Step.CONFIRMED  # unchanged; visitor can retry
