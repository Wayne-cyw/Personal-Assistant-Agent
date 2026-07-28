"""GET /health handler.

`db`/`llm` stay static for now. `calendar` is really checked (Issue #17):
a cheap free/busy ping against a 1-minute window, cached for
`_CACHE_SECONDS` so a revoked/expired refresh token is caught within a
bounded window (the issue's acceptance criteria: within 5 minutes) without
pinging Google on every single health check.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends

from app.tools.calendar import CalendarClient, CalendarError, get_calendar_client

router = APIRouter()

_CACHE_SECONDS = 300  # 5 minutes

# Module-global cache: there is exactly one calendar being checked, not one
# per session/request, so a single cached value (not a dict keyed by
# anything) is the right shape here.
_cached_status: str | None = None
_cached_at: float = 0.0


def get_health_calendar_client() -> CalendarClient:
    """FastAPI dependency wrapping get_calendar_client() — overridable in
    tests with a FakeCalendar, unlike a direct module-level call.
    """
    return get_calendar_client()


def _reset_calendar_health_cache() -> None:
    """Test-only helper — the cache is module-global, so tests that swap in
    a different calendar client between calls need a way to bypass a
    stale cached result from a previous test/call.
    """
    global _cached_status, _cached_at
    _cached_status = None
    _cached_at = 0.0


async def _check_calendar(client: CalendarClient) -> str:
    global _cached_status, _cached_at
    now = time.monotonic()
    if _cached_status is not None and (now - _cached_at) < _CACHE_SECONDS:
        return _cached_status

    ping_start = datetime.now(UTC)
    try:
        await client.get_free_busy(ping_start, ping_start + timedelta(minutes=1))
        status = "ok"
    except CalendarError:
        status = "error"

    _cached_status = status
    _cached_at = now
    return status


@router.get("/health")
async def health(
    calendar_client: CalendarClient = Depends(get_health_calendar_client),
) -> dict[str, str]:
    calendar_status = await _check_calendar(calendar_client)
    return {"status": "ok", "db": "ok", "llm": "unchecked", "calendar": calendar_status}
