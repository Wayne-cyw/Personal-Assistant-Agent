"""In-memory calendar test double (Issue #17) — implements CalendarClient's
Protocol without ever making a network call, mirroring
app/agent/providers/fake.py's FakeProvider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.tools.calendar import Attendee, BusyInterval, CalendarError


@dataclass
class _CreatedEvent:
    start: datetime
    end: datetime
    attendee: Attendee
    description: str
    status: str = "tentative"


@dataclass
class FakeCalendar:
    busy: list[BusyInterval] = field(default_factory=list)
    created_events: dict[str, _CreatedEvent] = field(default_factory=dict)
    # Set to raise this on every call instead of succeeding — for testing
    # CalendarError-handling paths (e.g. the /health degraded-status check).
    fail_with: CalendarError | None = None
    _next_id: int = 0

    async def get_free_busy(self, start: datetime, end: datetime) -> list[BusyInterval]:
        if self.fail_with is not None:
            raise self.fail_with
        return [b for b in self.busy if b.start < end and b.end > start]

    async def create_event(
        self, start: datetime, end: datetime, attendee: Attendee, description: str
    ) -> str:
        if self.fail_with is not None:
            raise self.fail_with
        self._next_id += 1
        event_id = f"fake-event-{self._next_id}"
        self.created_events[event_id] = _CreatedEvent(
            start=start, end=end, attendee=attendee, description=description
        )
        self.busy.append(BusyInterval(start=start, end=end))
        return event_id

    async def delete_event(self, event_id: str) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        event = self.created_events.pop(event_id, None)
        if event is None:
            return
        self.busy = [b for b in self.busy if not (b.start == event.start and b.end == event.end)]
