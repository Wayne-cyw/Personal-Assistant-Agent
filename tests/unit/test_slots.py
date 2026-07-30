import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.booking.slots import (
    AvailabilityPolicy,
    AvailabilityPolicyError,
    CoarseWindow,
    DayPart,
    Slot,
    generate_slots,
    get_availability_policy,
    parse_availability_policy,
    resolve_window,
    widen_window,
)
from app.tools.calendar import BusyInterval
from app.tools.fake_calendar import FakeCalendar

TZ = ZoneInfo("America/Toronto")


def _policy(**overrides: object) -> AvailabilityPolicy:
    defaults: dict[str, object] = dict(
        timezone=TZ,
        timezone_name="America/Toronto",
        work_start_day=0,  # Monday
        work_end_day=4,  # Friday
        work_start_time=datetime(2000, 1, 1, 9, 0).time(),
        work_end_time=datetime(2000, 1, 1, 17, 0).time(),
        meeting_length=timedelta(minutes=30),
        buffer=timedelta(minutes=15),
        min_notice=timedelta(hours=24),
    )
    defaults.update(overrides)
    return AvailabilityPolicy(**defaults)  # type: ignore[arg-type]


def _write_policy_file(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "availability_policy.md"
    path.write_text(content, encoding="utf-8")
    return path


_VALID_CONTENT = """## Availability

- **Timezone:** America/New_York
- **Working hours:** Monday-Friday, 09:00-17:00
- **Meeting length:** 30 minutes
- **Buffer:** 15 minutes
- **Minimum notice:** 24 hours
"""


# --- parse_availability_policy ----------------------------------------------


def test_parse_valid_policy(tmp_path: Path) -> None:
    path = _write_policy_file(tmp_path, _VALID_CONTENT)
    policy = parse_availability_policy(path)

    assert policy.timezone_name == "America/New_York"
    assert policy.work_start_day == 0  # Monday
    assert policy.work_end_day == 4  # Friday
    assert policy.work_start_time == datetime(2000, 1, 1, 9, 0).time()
    assert policy.work_end_time == datetime(2000, 1, 1, 17, 0).time()
    assert policy.meeting_length == timedelta(minutes=30)
    assert policy.buffer == timedelta(minutes=15)
    assert policy.min_notice == timedelta(hours=24)


def test_parse_missing_file_raises() -> None:
    with pytest.raises(AvailabilityPolicyError, match="does not exist"):
        parse_availability_policy(Path("/nonexistent/availability_policy.md"))


def test_parse_missing_required_field_raises(tmp_path: Path) -> None:
    content = """## Availability

- **Timezone:** America/New_York
- **Working hours:** Monday-Friday, 09:00-17:00
- **Meeting length:** 30 minutes
"""
    path = _write_policy_file(tmp_path, content)
    with pytest.raises(AvailabilityPolicyError, match="buffer"):
        parse_availability_policy(path)


def test_parse_unfilled_placeholder_raises(tmp_path: Path) -> None:
    content = """## Availability

- **Timezone:** [e.g. America/New_York]
- **Working hours:** Monday-Friday, 09:00-17:00
- **Meeting length:** 30 minutes
- **Buffer:** 15 minutes
- **Minimum notice:** 24 hours
"""
    path = _write_policy_file(tmp_path, content)
    with pytest.raises(AvailabilityPolicyError, match="placeholder"):
        parse_availability_policy(path)


def test_parse_invalid_timezone_raises(tmp_path: Path) -> None:
    content = _VALID_CONTENT.replace("America/New_York", "Not/A/Real/Zone")
    path = _write_policy_file(tmp_path, content)
    with pytest.raises(AvailabilityPolicyError, match="not a valid IANA"):
        parse_availability_policy(path)


def test_parse_malformed_working_hours_raises(tmp_path: Path) -> None:
    content = _VALID_CONTENT.replace("Monday-Friday, 09:00-17:00", "sometime weekdays")
    path = _write_policy_file(tmp_path, content)
    with pytest.raises(AvailabilityPolicyError, match="Working hours"):
        parse_availability_policy(path)


def test_parse_working_hours_start_after_end_raises(tmp_path: Path) -> None:
    content = _VALID_CONTENT.replace("09:00-17:00", "17:00-09:00")
    path = _write_policy_file(tmp_path, content)
    with pytest.raises(AvailabilityPolicyError, match="start must be before end"):
        parse_availability_policy(path)


def test_parse_malformed_duration_raises(tmp_path: Path) -> None:
    content = _VALID_CONTENT.replace("30 minutes", "half an hour")
    path = _write_policy_file(tmp_path, content)
    with pytest.raises(AvailabilityPolicyError, match="Meeting length"):
        parse_availability_policy(path)


def test_parse_duration_units() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        content = _VALID_CONTENT.replace("15 minutes", "1 day").replace("24 hours", "2 hours")
        path = _write_policy_file(Path(tmp), content)
        policy = parse_availability_policy(path)
    assert policy.buffer == timedelta(days=1)
    assert policy.min_notice == timedelta(hours=2)


def test_is_workday() -> None:
    policy = _policy()
    assert policy.is_workday(0) is True  # Monday
    assert policy.is_workday(4) is True  # Friday
    assert policy.is_workday(5) is False  # Saturday
    assert policy.is_workday(6) is False  # Sunday


# --- get_availability_policy (lazy singleton) --------------------------------


def test_get_availability_policy_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import app.booking.slots as slots_module

    monkeypatch.setattr(slots_module, "_policy", None)
    path = _write_policy_file(tmp_path, _VALID_CONTENT)
    monkeypatch.setattr(slots_module, "_POLICY_PATH", path)

    first = get_availability_policy()
    # Mutate the underlying file — a cached call must not re-read it.
    path.write_text(_VALID_CONTENT.replace("America/New_York", "Europe/London"), encoding="utf-8")
    second = get_availability_policy()

    assert first is second
    assert second.timezone_name == "America/New_York"


# --- CoarseWindow / resolve_window / widen_window ----------------------------


def test_coarse_window_rejects_date_to_before_date_from() -> None:
    with pytest.raises(ValueError, match="date_to"):
        CoarseWindow(date_from=date(2026, 8, 10), date_to=date(2026, 8, 1))


def test_coarse_window_allows_equal_dates() -> None:
    window = CoarseWindow(date_from=date(2026, 8, 10), date_to=date(2026, 8, 10))
    assert window.date_from == window.date_to


def test_resolve_window_spans_full_days_in_policy_timezone() -> None:
    policy = _policy()
    coarse = CoarseWindow(date_from=date(2026, 8, 3), date_to=date(2026, 8, 7))
    resolved = resolve_window(coarse, policy)

    assert resolved.start == datetime(2026, 8, 3, 0, 0, tzinfo=TZ)
    assert resolved.end.date() == date(2026, 8, 7)
    assert resolved.end.tzinfo is not None
    assert resolved.day_part is None


def test_resolve_window_preserves_day_part() -> None:
    policy = _policy()
    coarse = CoarseWindow(
        date_from=date(2026, 8, 3), date_to=date(2026, 8, 7), day_part=DayPart.MORNING
    )
    resolved = resolve_window(coarse, policy)
    assert resolved.day_part is DayPart.MORNING


def test_widen_window_extends_end_by_default_seven_days() -> None:
    policy = _policy()
    resolved = resolve_window(
        CoarseWindow(date_from=date(2026, 8, 3), date_to=date(2026, 8, 7)), policy
    )
    widened = widen_window(resolved)

    assert widened.start == resolved.start
    assert widened.end == resolved.end + timedelta(days=7)


def test_widen_window_preserves_day_part() -> None:
    policy = _policy()
    coarse = CoarseWindow(
        date_from=date(2026, 8, 3), date_to=date(2026, 8, 7), day_part=DayPart.EVENING
    )
    resolved = resolve_window(coarse, policy)
    widened = widen_window(resolved)
    assert widened.day_part is DayPart.EVENING


# --- generate_slots ----------------------------------------------------------


def _now_for_window(window_start: datetime, hours_before: float = 48) -> datetime:
    return window_start - timedelta(hours=hours_before)


async def test_generate_slots_empty_window_returns_empty_list() -> None:
    policy = _policy()
    # A window entirely inside the min-notice period from `now`.
    resolved = resolve_window(
        CoarseWindow(date_from=date(2026, 8, 3), date_to=date(2026, 8, 3)), policy
    )
    now = datetime(2026, 8, 2, 23, 59, tzinfo=TZ)  # min_notice (24h) pushes past the window
    fake = FakeCalendar()

    slots = await generate_slots(fake, resolved, policy, "America/Toronto", now=now)
    assert slots == []


async def test_generate_slots_no_busy_time_fills_the_workday() -> None:
    policy = _policy()
    # Monday Aug 3, 2026.
    resolved = resolve_window(
        CoarseWindow(date_from=date(2026, 8, 3), date_to=date(2026, 8, 3)), policy
    )
    now = _now_for_window(resolved.start)
    fake = FakeCalendar()

    slots = await generate_slots(fake, resolved, policy, "America/Toronto", now=now)

    assert 0 < len(slots) <= 5
    for slot in slots:
        start = datetime.fromisoformat(slot.start_iso)
        assert start.astimezone(TZ).time() >= policy.work_start_time
        end = datetime.fromisoformat(slot.end_iso)
        assert end.astimezone(TZ).time() <= policy.work_end_time


async def test_generate_slots_subtracts_busy_time() -> None:
    # buffer=0 here so this test isolates busy-time subtraction specifically
    # from buffer padding, which has its own dedicated test below.
    policy = _policy(buffer=timedelta(0))
    resolved = resolve_window(
        CoarseWindow(date_from=date(2026, 8, 3), date_to=date(2026, 8, 3)), policy
    )
    now = _now_for_window(resolved.start)
    # Busy for the entire workday except the last 30 minutes.
    fake = FakeCalendar(
        busy=[
            BusyInterval(
                start=datetime(2026, 8, 3, 9, 0, tzinfo=TZ),
                end=datetime(2026, 8, 3, 16, 30, tzinfo=TZ),
            )
        ]
    )

    slots = await generate_slots(fake, resolved, policy, "America/Toronto", now=now)

    assert len(slots) == 1
    start = datetime.fromisoformat(slots[0].start_iso)
    assert start == datetime(2026, 8, 3, 16, 30, tzinfo=TZ)


async def test_generate_slots_applies_buffer_around_busy_time() -> None:
    policy = _policy(buffer=timedelta(minutes=15))
    resolved = resolve_window(
        CoarseWindow(date_from=date(2026, 8, 3), date_to=date(2026, 8, 3)), policy
    )
    now = _now_for_window(resolved.start)
    # A short 30-minute meeting at 12:00 — with a 15-minute buffer on each
    # side, the slots immediately before/after (11:30 and 12:30) must also
    # be excluded, not just the meeting's own exact time.
    fake = FakeCalendar(
        busy=[
            BusyInterval(
                start=datetime(2026, 8, 3, 12, 0, tzinfo=TZ),
                end=datetime(2026, 8, 3, 12, 30, tzinfo=TZ),
            )
        ]
    )

    slots = await generate_slots(fake, resolved, policy, "America/Toronto", now=now)

    starts = {datetime.fromisoformat(s.start_iso) for s in slots}
    assert datetime(2026, 8, 3, 11, 30, tzinfo=TZ) not in starts
    assert datetime(2026, 8, 3, 12, 0, tzinfo=TZ) not in starts
    assert datetime(2026, 8, 3, 12, 30, tzinfo=TZ) not in starts


async def test_generate_slots_respects_minimum_notice() -> None:
    policy = _policy(min_notice=timedelta(hours=24))
    resolved = resolve_window(
        CoarseWindow(date_from=date(2026, 8, 3), date_to=date(2026, 8, 4)), policy
    )
    # "Now" is Monday 10:00 — 24h notice pushes the earliest allowed slot
    # to Tuesday 10:00, so no Monday slots should appear.
    now = datetime(2026, 8, 3, 10, 0, tzinfo=TZ)
    fake = FakeCalendar()

    slots = await generate_slots(fake, resolved, policy, "America/Toronto", now=now)

    for slot in slots:
        start = datetime.fromisoformat(slot.start_iso)
        assert start >= now + policy.min_notice


async def test_generate_slots_excludes_non_workdays() -> None:
    policy = _policy()
    # Sat Aug 8 - Sun Aug 9, 2026 — a weekend, entirely outside Mon-Fri.
    resolved = resolve_window(
        CoarseWindow(date_from=date(2026, 8, 8), date_to=date(2026, 8, 9)), policy
    )
    now = _now_for_window(resolved.start)
    fake = FakeCalendar()

    slots = await generate_slots(fake, resolved, policy, "America/Toronto", now=now)
    assert slots == []


async def test_generate_slots_respects_exclusion_list() -> None:
    policy = _policy()
    resolved = resolve_window(
        CoarseWindow(date_from=date(2026, 8, 3), date_to=date(2026, 8, 3)), policy
    )
    now = _now_for_window(resolved.start)
    fake = FakeCalendar()
    burned = (
        datetime(2026, 8, 3, 9, 0, tzinfo=TZ),
        datetime(2026, 8, 3, 9, 30, tzinfo=TZ),
    )

    slots = await generate_slots(
        fake, resolved, policy, "America/Toronto", now=now, exclude=[burned]
    )

    starts = {datetime.fromisoformat(s.start_iso) for s in slots}
    assert datetime(2026, 8, 3, 9, 0, tzinfo=TZ) not in starts


async def test_generate_slots_day_part_filters_to_afternoon_only() -> None:
    policy = _policy()
    coarse = CoarseWindow(
        date_from=date(2026, 8, 3), date_to=date(2026, 8, 3), day_part=DayPart.AFTERNOON
    )
    resolved = resolve_window(coarse, policy)
    now = _now_for_window(resolved.start)
    fake = FakeCalendar()

    slots = await generate_slots(fake, resolved, policy, "America/Toronto", now=now)

    for slot in slots:
        start = datetime.fromisoformat(slot.start_iso).astimezone(TZ)
        assert start.hour >= 12


async def test_generate_slots_picks_at_most_five_spread_across_a_long_window() -> None:
    policy = _policy()
    resolved = resolve_window(
        CoarseWindow(date_from=date(2026, 8, 3), date_to=date(2026, 8, 14)), policy
    )
    now = _now_for_window(resolved.start)
    fake = FakeCalendar()

    slots = await generate_slots(fake, resolved, policy, "America/Toronto", now=now)

    assert len(slots) == 5
    days = {datetime.fromisoformat(s.start_iso).astimezone(TZ).date() for s in slots}
    # Spread across more than one day, not clustered entirely on day one.
    assert len(days) > 1


async def test_generate_slots_labels_are_correct_for_toronto_caller() -> None:
    # buffer=0: isolates label formatting from buffer-padding's effect on
    # which candidate survives (already covered by a dedicated test).
    policy = _policy(buffer=timedelta(0))
    resolved = resolve_window(
        CoarseWindow(date_from=date(2026, 7, 28), date_to=date(2026, 7, 28)), policy
    )
    now = _now_for_window(resolved.start)
    fake = FakeCalendar(
        busy=[
            BusyInterval(
                start=datetime(2026, 7, 28, 9, 0, tzinfo=TZ),
                end=datetime(2026, 7, 28, 14, 0, tzinfo=TZ),
            )
        ]
    )

    slots = await generate_slots(fake, resolved, policy, "America/Toronto", now=now)

    assert slots[0].label == "Tue Jul 28, 2:00–2:30 PM EDT"


async def test_generate_slots_labels_are_correct_for_london_caller() -> None:
    """Same underlying availability as the Toronto test, but rendered for
    a caller in Europe/London — a different clock time and abbreviation
    for the identical real-world instant (Issue #19 acceptance criteria).
    """
    policy = _policy(buffer=timedelta(0))
    resolved = resolve_window(
        CoarseWindow(date_from=date(2026, 7, 28), date_to=date(2026, 7, 28)), policy
    )
    now = _now_for_window(resolved.start)
    fake = FakeCalendar(
        busy=[
            BusyInterval(
                start=datetime(2026, 7, 28, 9, 0, tzinfo=TZ),
                end=datetime(2026, 7, 28, 14, 0, tzinfo=TZ),
            )
        ]
    )

    slots = await generate_slots(fake, resolved, policy, "Europe/London", now=now)

    # 2:00 PM EDT (UTC-4) == 7:00 PM BST (UTC+1) in late July.
    assert slots[0].label == "Tue Jul 28, 7:00–7:30 PM BST"


async def test_generate_slots_correct_across_spring_forward_dst_boundary() -> None:
    """America/Toronto springs forward on 2026-03-08. Slots generated for
    the workday immediately after must still land at 09:00-17:00 *local*
    time, not shift by an hour due to naive UTC-offset arithmetic.
    """
    policy = _policy()
    # Monday 2026-03-09, the first workday after the Sunday transition.
    resolved = resolve_window(
        CoarseWindow(date_from=date(2026, 3, 9), date_to=date(2026, 3, 9)), policy
    )
    now = _now_for_window(resolved.start)
    fake = FakeCalendar()

    slots = await generate_slots(fake, resolved, policy, "America/Toronto", now=now)

    first_start = datetime.fromisoformat(slots[0].start_iso).astimezone(TZ)
    assert first_start == datetime(2026, 3, 9, 9, 0, tzinfo=TZ)
    assert first_start.tzname() == "EDT"  # already in daylight time


async def test_generate_slots_correct_across_fall_back_dst_boundary() -> None:
    """America/Toronto falls back on 2026-11-01."""
    policy = _policy()
    # Monday 2026-11-02, the first workday after the Sunday transition.
    resolved = resolve_window(
        CoarseWindow(date_from=date(2026, 11, 2), date_to=date(2026, 11, 2)), policy
    )
    now = _now_for_window(resolved.start)
    fake = FakeCalendar()

    slots = await generate_slots(fake, resolved, policy, "America/Toronto", now=now)

    first_start = datetime.fromisoformat(slots[0].start_iso).astimezone(TZ)
    assert first_start == datetime(2026, 11, 2, 9, 0, tzinfo=TZ)
    assert first_start.tzname() == "EST"  # already back in standard time


def test_slot_model_fields() -> None:
    slot = Slot(
        slot_id="abc",
        start_iso="2026-08-03T09:00:00-04:00",
        end_iso="2026-08-03T09:30:00-04:00",
        label="Mon Aug 3, 9:00–9:30 AM EDT",
    )
    assert slot.slot_id == "abc"
