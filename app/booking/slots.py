"""Free/busy to structured slots, availability policy, and natural-language
window resolution (Engineering Guide §4.5, Issue #19).

Natural-language window parsing itself does NOT happen here — the LLM
extracts a *coarse* structured window (`CoarseWindow`: date_from, date_to,
an optional day_part) as tool-call arguments (Issue #20's
calendar_find_slots registration); this module only validates that and
converts it into a concrete, tz-aware query, then does the actual time
math in code. No dateutil-style free-text parsing of raw user input — the
LLM does language, code does time math, all tz-aware, stored UTC.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, model_validator

from app.tools.calendar import CalendarClient

_POLICY_PATH = Path(__file__).resolve().parents[2] / "knowledge" / "availability_policy.md"

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Up to 5 slots per proposal (4.5's slots_proposed row: "returns 3-5
# structured slots") — the lower end of that range isn't a separate target
# to hit; it falls out naturally from however many real openings exist
# (including zero, the empty-window acceptance case).
_TARGET_SLOT_COUNT = 5


class DayPart(StrEnum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"


_DAY_PART_HOURS: dict[DayPart, tuple[time, time]] = {
    DayPart.MORNING: (time(8, 0), time(12, 0)),
    DayPart.AFTERNOON: (time(12, 0), time(17, 0)),
    DayPart.EVENING: (time(17, 0), time(20, 0)),
}


class AvailabilityPolicyError(Exception):
    """Raised for a missing, malformed, or still-templated
    availability_policy.md — see that file's own comment block for the
    exact expected format.
    """


@dataclass
class AvailabilityPolicy:
    timezone: ZoneInfo
    timezone_name: str
    work_start_day: int  # 0=Monday .. 6=Sunday
    work_end_day: int
    work_start_time: time
    work_end_time: time
    meeting_length: timedelta
    buffer: timedelta
    min_notice: timedelta

    def is_workday(self, weekday: int) -> bool:
        return self.work_start_day <= weekday <= self.work_end_day


_LABEL_LINE = re.compile(r"^-\s*\*\*(?P<label>[^*]+):\*\*\s*(?P<value>.+)$")
_DURATION = re.compile(
    r"^(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>minute|min|hour|hr|day)s?$", re.IGNORECASE
)
_WORKING_HOURS = re.compile(
    r"^(?P<start_day>[A-Za-z]+)-(?P<end_day>[A-Za-z]+),\s*"
    r"(?P<start_hour>\d{1,2}):(?P<start_min>\d{2})-(?P<end_hour>\d{1,2}):(?P<end_min>\d{2})$"
)

_REQUIRED_FIELDS = ("timezone", "working hours", "meeting length", "buffer", "minimum notice")


def _parse_duration(raw: str, *, field: str) -> timedelta:
    match = _DURATION.match(raw.strip())
    if match is None:
        raise AvailabilityPolicyError(f"{field}: could not parse duration {raw!r}")
    amount = float(match.group("amount"))
    unit = match.group("unit").lower()
    if unit in ("minute", "min"):
        return timedelta(minutes=amount)
    if unit in ("hour", "hr"):
        return timedelta(hours=amount)
    return timedelta(days=amount)


def _parse_working_hours(raw: str) -> tuple[int, int, time, time]:
    match = _WORKING_HOURS.match(raw.strip())
    if match is None:
        raise AvailabilityPolicyError(f"Working hours: could not parse {raw!r}")
    try:
        start_day = _WEEKDAYS.index(match.group("start_day").lower())
        end_day = _WEEKDAYS.index(match.group("end_day").lower())
    except ValueError:
        raise AvailabilityPolicyError(f"Working hours: unrecognized day name in {raw!r}") from None
    start_t = time(int(match.group("start_hour")), int(match.group("start_min")))
    end_t = time(int(match.group("end_hour")), int(match.group("end_min")))
    if start_t >= end_t:
        raise AvailabilityPolicyError(f"Working hours: start must be before end in {raw!r}")
    return start_day, end_day, start_t, end_t


def parse_availability_policy(path: Path) -> AvailabilityPolicy:
    """Parse `- **Label:** value` lines out of an availability_policy.md
    file. Raises AvailabilityPolicyError on a missing file, a missing
    required label, or a value that still looks like an unfilled `[...]`
    template placeholder — deliberately fails loudly rather than silently
    computing availability from garbage data.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise AvailabilityPolicyError(f"{path} does not exist") from None

    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = _LABEL_LINE.match(line.strip())
        if match is None:
            continue
        fields[match.group("label").strip().lower()] = match.group("value").strip()

    for label in _REQUIRED_FIELDS:
        if label not in fields:
            raise AvailabilityPolicyError(f"{path.name} is missing a {label!r} line")
        if fields[label].startswith("["):
            raise AvailabilityPolicyError(
                f"{path.name}'s {label!r} line is still an unfilled template placeholder"
            )

    timezone_name = fields["timezone"]
    try:
        timezone = ZoneInfo(timezone_name)
    except Exception as exc:
        raise AvailabilityPolicyError(
            f"Timezone: {timezone_name!r} is not a valid IANA timezone name"
        ) from exc

    start_day, end_day, work_start, work_end = _parse_working_hours(fields["working hours"])

    return AvailabilityPolicy(
        timezone=timezone,
        timezone_name=timezone_name,
        work_start_day=start_day,
        work_end_day=end_day,
        work_start_time=work_start,
        work_end_time=work_end,
        meeting_length=_parse_duration(fields["meeting length"], field="Meeting length"),
        buffer=_parse_duration(fields["buffer"], field="Buffer"),
        min_notice=_parse_duration(fields["minimum notice"], field="Minimum notice"),
    )


_policy: AvailabilityPolicy | None = None


def get_availability_policy() -> AvailabilityPolicy:
    """Lazily parsed and cached — deliberately NOT evaluated at import
    time (unlike e.g. app/agent/intro.py's INTRO_MESSAGE): most of the app
    has nothing to do with booking and shouldn't fail to even import over
    a booking-only config file, and eager evaluation would break test
    collection for the whole suite until knowledge/availability_policy.md
    is actually filled in with real values (Issue #13).
    """
    global _policy
    if _policy is None:
        _policy = parse_availability_policy(_POLICY_PATH)
    return _policy


class CoarseWindow(BaseModel):
    """The LLM's tool-call arguments for calendar_find_slots (Issue #20) —
    the LLM extracts *coarse* structure ("next week", "afternoons"), never
    a concrete datetime; resolve_window() below does the actual time math
    in code (4.2: tool inputs/results are structured, never inferred from
    prose).
    """

    date_from: date
    date_to: date
    day_part: DayPart | None = None

    @model_validator(mode="after")
    def _date_to_not_before_date_from(self) -> CoarseWindow:
        if self.date_to < self.date_from:
            raise ValueError("date_to must not be before date_from")
        return self


@dataclass
class ResolvedWindow:
    start: datetime
    end: datetime
    day_part: DayPart | None


def resolve_window(coarse: CoarseWindow, policy: AvailabilityPolicy) -> ResolvedWindow:
    """Convert the LLM's coarse date range into a concrete, tz-aware query
    range grounded in the *owner's* timezone (the one working hours are
    defined in per availability_policy.md) — the caller's own display
    timezone is a separate, later concern (generate_slots' label
    rendering only).
    """
    start = datetime.combine(coarse.date_from, time.min, tzinfo=policy.timezone)
    end = datetime.combine(coarse.date_to, time.max, tzinfo=policy.timezone)
    return ResolvedWindow(start=start, end=end, day_part=coarse.day_part)


def widen_window(resolved: ResolvedWindow, *, days: int = 7) -> ResolvedWindow:
    """The negotiation fallback's "widen the window once" strategy (4.5)."""
    return ResolvedWindow(
        start=resolved.start, end=resolved.end + timedelta(days=days), day_part=resolved.day_part
    )


class Slot(BaseModel):
    slot_id: str
    start_iso: str
    end_iso: str
    label: str


def _overlaps_any(start: datetime, end: datetime, blocked: list[tuple[datetime, datetime]]) -> bool:
    return any(start < b_end and end > b_start for b_start, b_end in blocked)


def _step(cursor: datetime, delta: timedelta) -> datetime:
    """Advance `cursor` by exactly `delta` of absolute (UTC) time, then
    re-express the result in `cursor`'s original zone.

    Plain `cursor + delta` on a zoneinfo-aware datetime performs *wall-
    clock* arithmetic, not absolute-time arithmetic — a well-known
    zoneinfo gotcha. Near a DST spring-forward, that can silently produce
    a slot whose `end` precedes its `start` in real time: e.g. adding 30
    minutes to a 02:30 local start (a wall-clock time that doesn't exist
    on the transition date, since clocks skip 02:00 -> 03:00) resolves
    using the *pre*-transition UTC offset, while the resulting 03:00
    already resolves using the *post*-transition offset — so `end` ends up
    30 minutes *before* `start`. Converting through UTC sidesteps this
    entirely: UTC has no DST, so the arithmetic is always unambiguous, and
    every generated slot is guaranteed exactly `delta` long in real time.

    This dodges the acute "inverted slot" bug but not every DST edge case
    in general — day_start/day_end below are still constructed via
    datetime.combine(), which could in principle land on a non-existent
    local time if a policy's working hours started or ended exactly within
    a transition gap. Not addressed here: no realistic booking policy has
    working hours starting at 2 AM, and handling that fully general case
    is out of proportion to this issue's scope.
    """
    return (cursor.astimezone(UTC) + delta).astimezone(cursor.tzinfo)


def _discretize(
    window_start: datetime,
    window_end: datetime,
    policy: AvailabilityPolicy,
    day_part: DayPart | None,
    blocked: list[tuple[datetime, datetime]],
) -> list[datetime]:
    candidates: list[datetime] = []
    day = window_start.astimezone(policy.timezone).date()
    end_date = window_end.astimezone(policy.timezone).date()

    while day <= end_date:
        if policy.is_workday(day.weekday()):
            day_start_time, day_end_time = policy.work_start_time, policy.work_end_time
            if day_part is not None:
                part_start, part_end = _DAY_PART_HOURS[day_part]
                day_start_time = max(day_start_time, part_start)
                day_end_time = min(day_end_time, part_end)

            if day_start_time < day_end_time:
                day_start = datetime.combine(day, day_start_time, tzinfo=policy.timezone)
                day_end = datetime.combine(day, day_end_time, tzinfo=policy.timezone)
                cursor = max(day_start, window_start)
                capped_end = min(day_end, window_end)
                while True:
                    slot_end = _step(cursor, policy.meeting_length)
                    if slot_end > capped_end:
                        break
                    if not _overlaps_any(cursor, slot_end, blocked):
                        candidates.append(cursor)
                    cursor = slot_end

        day += timedelta(days=1)

    return candidates


def _spread_pick(candidates: list[datetime], count: int) -> list[datetime]:
    """Pick up to `count` candidates spread evenly across the list, rather
    than just the earliest `count` — so a long window doesn't collapse
    into five options all on the first available day.
    """
    if len(candidates) <= count:
        return candidates
    step = len(candidates) / count
    indices = sorted({min(round(i * step), len(candidates) - 1) for i in range(count)})
    chosen = [candidates[i] for i in indices]
    if len(chosen) < count:
        remaining = [c for i, c in enumerate(candidates) if i not in indices]
        chosen.extend(remaining[: count - len(chosen)])
    return chosen[:count]


def _format_time_12h(value: datetime, *, include_period: bool) -> str:
    hour12 = value.strftime("%I").lstrip("0") or "12"
    minute = value.strftime("%M")
    if not include_period:
        return f"{hour12}:{minute}"
    return f"{hour12}:{minute} {value.strftime('%p')}"


def _format_label(start: datetime, end: datetime, caller_tz: ZoneInfo) -> str:
    start_local = start.astimezone(caller_tz)
    end_local = end.astimezone(caller_tz)
    date_part = f"{start_local.strftime('%a %b')} {start_local.day}"
    same_period = start_local.strftime("%p") == end_local.strftime("%p")
    start_time = _format_time_12h(start_local, include_period=not same_period)
    end_time = _format_time_12h(end_local, include_period=True)
    tz_abbrev = end_local.tzname() or ""
    return f"{date_part}, {start_time}–{end_time} {tz_abbrev}".strip()


async def generate_slots(
    calendar_client: CalendarClient,
    resolved: ResolvedWindow,
    policy: AvailabilityPolicy,
    caller_timezone: str,
    *,
    now: datetime,
    exclude: list[tuple[datetime, datetime]] | None = None,
) -> list[Slot]:
    """Free/busy -> subtract busy + buffers + min-notice -> discretize into
    meeting-length slots -> pick up to 5 spread across the window ->
    `{slot_id, start_iso, end_iso, label}` with labels rendered in the
    caller's timezone.
    """
    if now.tzinfo is None:
        raise ValueError("now must be a timezone-aware datetime, got a naive one")

    # min_notice is a real-elapsed-time guarantee ("at least 24 hours from
    # now"), not a wall-clock offset, so this needs the same UTC-safe
    # stepping as meeting durations — plain `+` here could under-deliver
    # the promised notice by an hour across a spring-forward transition.
    earliest_allowed = _step(now, policy.min_notice)
    query_start = max(resolved.start, earliest_allowed)
    if query_start >= resolved.end:
        return []

    busy = await calendar_client.get_free_busy(query_start, resolved.end)
    # Padding each busy interval by the buffer on both sides, and treating
    # excluded (burned, from a prior proposal round) ranges the same as
    # busy time, are both just more "blocked" ranges to the discretizer.
    blocked: list[tuple[datetime, datetime]] = [
        (b.start - policy.buffer, b.end + policy.buffer) for b in busy
    ]
    blocked.extend(exclude or [])

    candidates = _discretize(query_start, resolved.end, policy, resolved.day_part, blocked)
    chosen = _spread_pick(candidates, _TARGET_SLOT_COUNT)

    caller_tz = ZoneInfo(caller_timezone)
    slots = []
    for start in chosen:
        end = _step(start, policy.meeting_length)
        slots.append(
            Slot(
                slot_id=str(uuid.uuid4()),
                start_iso=start.isoformat(),
                end_iso=end.isoformat(),
                label=_format_label(start, end, caller_tz),
            )
        )
    return slots
