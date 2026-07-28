"""Smoke test for the Google Calendar wrapper (Issue #17 acceptance
criteria): prints real free/busy for the next 7 days, then creates and
deletes a test tentative event. Requires real GOOGLE_CLIENT_ID/
GOOGLE_CLIENT_SECRET/GOOGLE_REFRESH_TOKEN (see README's "Google Calendar
setup" section) — not run as part of `pytest`.

Usage: python scripts/gcal_smoke.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from app.tools.calendar import Attendee, get_calendar_client


async def main() -> None:
    client = get_calendar_client()
    now = datetime.now(UTC)
    window_end = now + timedelta(days=7)

    busy = await client.get_free_busy(now, window_end)
    print(f"Busy intervals over the next 7 days ({len(busy)}):")
    for interval in busy:
        print(f"  {interval.start.isoformat()} - {interval.end.isoformat()}")

    print("\nCreating a test tentative event...")
    start = now + timedelta(hours=1)
    end = start + timedelta(minutes=30)
    event_id = await client.create_event(
        start,
        end,
        Attendee(name="Smoke Test", email="smoke-test@example.com"),
        "scripts/gcal_smoke.py test event — safe to ignore if you see it briefly appear.",
    )
    print(f"Created event: {event_id}")

    print("Deleting it...")
    await client.delete_event(event_id)
    print("Deleted. Smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
