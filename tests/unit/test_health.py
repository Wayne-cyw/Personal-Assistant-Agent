import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

import app.api.health as health_module
from app.api.health import _check_availability_policy, _check_calendar
from app.booking.slots import AvailabilityPolicy, AvailabilityPolicyError
from app.tools.calendar import BusyInterval, CalendarError
from app.tools.fake_calendar import FakeCalendar

# tests/conftest.py's autouse _reset_calendar_health_cache fixture resets
# the module-global cache/lock before every test in the suite.


async def test_check_calendar_ok_when_free_busy_succeeds() -> None:
    fake = FakeCalendar()
    assert await _check_calendar(fake) == "ok"


async def test_check_calendar_error_when_free_busy_raises() -> None:
    fake = FakeCalendar(fail_with=CalendarError("auth failed"))
    assert await _check_calendar(fake) == "error"


async def test_check_calendar_result_is_cached_within_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ok = FakeCalendar()
    result1 = await _check_calendar(fake_ok)

    # Swap in a failing client — a fresh, uncached check would report
    # "error", but the cached "ok" from moments ago should still be
    # returned since no time (simulated) has passed.
    fake_failing = FakeCalendar(fail_with=CalendarError("auth failed"))
    result2 = await _check_calendar(fake_failing)

    assert result1 == "ok"
    assert result2 == "ok"  # cached, not re-checked


async def test_check_calendar_re_checks_after_cache_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ok = FakeCalendar()
    await _check_calendar(fake_ok)

    # Simulate the cache having gone stale by moving its recorded
    # timestamp far enough into the past.
    monkeypatch.setattr(health_module, "_cached_at", 0.0)

    fake_failing = FakeCalendar(fail_with=CalendarError("auth failed"))
    result = await _check_calendar(fake_failing)

    assert result == "error"


class _CountingSlowCalendar:
    """Records how many times get_free_busy was actually called, with a
    real await in between so concurrent callers can interleave.
    """

    def __init__(self) -> None:
        self.call_count = 0

    async def get_free_busy(self, _start: datetime, _end: datetime) -> list[BusyInterval]:
        self.call_count += 1
        await asyncio.sleep(0.05)
        return []

    async def create_event(self, *args: object, **kwargs: object) -> str:
        raise NotImplementedError

    async def delete_event(self, event_id: str) -> None:
        raise NotImplementedError


def test_check_availability_policy_ok_when_policy_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = AvailabilityPolicy(
        timezone=ZoneInfo("America/New_York"),
        timezone_name="America/New_York",
        work_start_day=0,
        work_end_day=4,
        work_start_time=datetime(2026, 1, 1, 9, 0).time(),
        work_end_time=datetime(2026, 1, 1, 17, 0).time(),
        meeting_length=timedelta(minutes=30),
        buffer=timedelta(minutes=15),
        min_notice=timedelta(hours=24),
    )
    monkeypatch.setattr(health_module, "get_availability_policy", lambda: policy)

    assert _check_availability_policy() == "ok"


def test_check_availability_policy_not_configured_when_still_a_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> AvailabilityPolicy:
        raise AvailabilityPolicyError("still a template")

    monkeypatch.setattr(health_module, "get_availability_policy", _raise)

    assert _check_availability_policy() == "not_configured"


async def test_concurrent_checks_on_a_stale_cache_only_call_the_api_once() -> None:
    """Regression test: two /health requests arriving as the cache goes
    stale must not both fire a real Calendar API call — the point of
    caching in the first place. Without the lock in _check_calendar, both
    concurrent calls observe the stale cache before either writes it back.
    """
    fake = _CountingSlowCalendar()

    results = await asyncio.gather(_check_calendar(fake), _check_calendar(fake))  # type: ignore[arg-type]

    assert results == ["ok", "ok"]
    assert fake.call_count == 1
