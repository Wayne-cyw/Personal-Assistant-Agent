import pytest

import app.api.health as health_module
from app.api.health import _check_calendar, _reset_calendar_health_cache
from app.tools.calendar import CalendarError
from app.tools.fake_calendar import FakeCalendar


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    _reset_calendar_health_cache()
    yield
    _reset_calendar_health_cache()


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
