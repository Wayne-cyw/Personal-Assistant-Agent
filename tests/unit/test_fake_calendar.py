from datetime import UTC, datetime

import pytest

from app.tools.calendar import Attendee, BusyInterval, CalendarError
from app.tools.fake_calendar import FakeCalendar


def _dt(hour: int) -> datetime:
    return datetime(2026, 8, 3, hour, 0, tzinfo=UTC)


async def test_get_free_busy_returns_only_overlapping_intervals() -> None:
    fake = FakeCalendar(
        busy=[
            BusyInterval(start=_dt(9), end=_dt(10)),  # overlaps the query window
            BusyInterval(start=_dt(20), end=_dt(21)),  # does not
        ]
    )

    result = await fake.get_free_busy(_dt(8), _dt(12))

    assert result == [BusyInterval(start=_dt(9), end=_dt(10))]


async def test_get_free_busy_excludes_touching_but_non_overlapping_intervals() -> None:
    """Boundary case: a busy interval that ends exactly when the query
    window starts (or starts exactly when it ends) does not overlap —
    half-open interval semantics, matching how calendar slots are never
    double-booked at the exact boundary second.
    """
    fake = FakeCalendar(
        busy=[
            BusyInterval(start=_dt(7), end=_dt(9)),  # ends exactly at query start
            BusyInterval(start=_dt(12), end=_dt(14)),  # starts exactly at query end
        ]
    )

    result = await fake.get_free_busy(_dt(9), _dt(12))

    assert result == []


async def test_create_event_returns_unique_ids_and_tracks_state() -> None:
    fake = FakeCalendar()
    attendee = Attendee(name="Priya", email="priya@example.com")

    first_id = await fake.create_event(_dt(9), _dt(10), attendee, "First call")
    second_id = await fake.create_event(_dt(11), _dt(12), attendee, "Second call")

    assert first_id != second_id
    assert fake.created_events[first_id].description == "First call"
    assert fake.created_events[second_id].description == "Second call"


async def test_create_event_adds_to_busy_list() -> None:
    fake = FakeCalendar()
    attendee = Attendee(name="Priya", email="priya@example.com")

    await fake.create_event(_dt(9), _dt(10), attendee, "Call")

    assert fake.busy == [BusyInterval(start=_dt(9), end=_dt(10))]


async def test_delete_event_removes_from_created_events_and_busy_list() -> None:
    fake = FakeCalendar()
    attendee = Attendee(name="Priya", email="priya@example.com")
    event_id = await fake.create_event(_dt(9), _dt(10), attendee, "Call")

    await fake.delete_event(event_id)

    assert event_id not in fake.created_events
    assert fake.busy == []


async def test_delete_event_unknown_id_is_a_harmless_noop() -> None:
    fake = FakeCalendar()
    await fake.delete_event("does-not-exist")  # must not raise


async def test_fail_with_raises_on_every_method() -> None:
    err = CalendarError("simulated failure")
    fake = FakeCalendar(fail_with=err)
    attendee = Attendee(name="Priya", email="priya@example.com")

    with pytest.raises(CalendarError):
        await fake.get_free_busy(_dt(9), _dt(10))
    with pytest.raises(CalendarError):
        await fake.create_event(_dt(9), _dt(10), attendee, "Call")
    with pytest.raises(CalendarError):
        await fake.delete_event("any-id")
