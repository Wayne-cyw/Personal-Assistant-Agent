"""GET /health handler.

`db`/`llm` stay static for now. `calendar` is really checked (Issue #17):
a cheap free/busy ping against a 1-minute window, cached for
`_CACHE_SECONDS` so a revoked/expired refresh token is caught within a
bounded window (the issue's acceptance criteria: within 5 minutes) without
pinging Google on every single health check.
"""

from __future__ import annotations

import asyncio
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
# Serializes the check-then-refresh below across concurrent /health
# requests. Without this, two requests arriving as the cache goes stale
# can both observe a stale value, both await a real Google API call before
# either writes the cache back, and both fire a redundant outbound
# request — defeating the whole point of caching (rare polling still
# wouldn't corrupt anything, since it's a last-write-wins overwrite of two
# presumably-equal results, but it does mean the "without pinging Google on
# every single health check" guarantee doesn't actually hold under any
# overlapping health-check traffic, e.g. a host's own liveness probe
# overlapping a monitoring service's).
_check_lock = asyncio.Lock()


def get_health_calendar_client() -> CalendarClient:
    """FastAPI dependency wrapping get_calendar_client() — overridable in
    tests with a FakeCalendar, unlike a direct module-level call.
    """
    return get_calendar_client()


def _reset_calendar_health_cache() -> None:
    """Test-only helper — the cache is module-global, so tests that swap in
    a different calendar client between calls need a way to bypass a
    stale cached result from a previous test/call. Also rebuilds
    `_check_lock`: an `asyncio.Lock` binds to whatever event loop is
    running the first time it's awaited, and pytest-asyncio gives each
    test its own event loop (function scope) — reusing the same Lock
    object across tests raises "bound to a different event loop" the
    moment a second test's loop tries to acquire it. Real deployments never
    hit this (the process has exactly one event loop for its whole
    lifetime, Engineering Guide 4.3); it's purely a test-isolation
    artifact, the same one app/agent/session_lock.py's tests hit.
    """
    global _cached_status, _cached_at, _check_lock
    _cached_status = None
    _cached_at = 0.0
    _check_lock = asyncio.Lock()


def _cache_is_fresh() -> bool:
    return _cached_status is not None and (time.monotonic() - _cached_at) < _CACHE_SECONDS


async def _check_calendar(client: CalendarClient) -> str:
    global _cached_status, _cached_at

    if _cache_is_fresh():
        assert _cached_status is not None
        return _cached_status

    async with _check_lock:
        # Re-check inside the lock: a request that was waiting on the lock
        # may find the cache already refreshed by whichever request got
        # there first, and can skip the real API call entirely.
        if _cache_is_fresh():
            assert _cached_status is not None
            return _cached_status

        ping_start = datetime.now(UTC)
        try:
            await client.get_free_busy(ping_start, ping_start + timedelta(minutes=1))
            status = "ok"
        except CalendarError:
            status = "error"

        _cached_status = status
        _cached_at = time.monotonic()
        return status


@router.get("/health")
async def health(
    calendar_client: CalendarClient = Depends(get_health_calendar_client),
) -> dict[str, str]:
    calendar_status = await _check_calendar(calendar_client)
    return {"status": "ok", "db": "ok", "llm": "unchecked", "calendar": calendar_status}
