"""Per-session concurrency guard (Engineering Guide 4.3, Issue #11).

Two in-flight requests on the same session_id — an impatient double-send, a
client retry — race on the booking state machine (Issue #18) and on
window/summary bookkeeping (app/agent/memory.py). Turn processing is
therefore serialized per session_id: a concurrent request waits briefly and,
if still blocked, gets the standard 429 envelope (app/api/errors.py).

**This lock is only correct in a single process** — it's a plain in-process
`asyncio.Lock`, so it provides no guarantee across multiple uvicorn workers
or horizontal scaling. 4.3's rule for v1: **the app runs as exactly one
worker, everywhere, always.** Issue #34's deploy config is where that's
actually enforced; the startup check in app/main.py is a lighter in-app
guard against the same mistake.

Known limitation: `_locks` grows by one entry per distinct session_id ever
seen and is never pruned, for the lifetime of the process. Not addressed
here — out of this issue's scope, and at this traffic scale (single
long-running worker per 4.3) the memory cost of one asyncio.Lock per
session is not a practical concern before a process restart.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

_LOCK_WAIT_SECONDS = 5.0

_locks: dict[str, asyncio.Lock] = {}


class SessionBusyError(Exception):
    """Raised when a session's turn lock can't be acquired within the wait
    window — app/api/errors.py maps this to the standard 429 rate_limited
    envelope.
    """


def _get_lock(session_id: str) -> asyncio.Lock:
    lock = _locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[session_id] = lock
    return lock


@asynccontextmanager
async def session_turn_lock(
    session_id: str, *, wait_seconds: float = _LOCK_WAIT_SECONDS
) -> AsyncIterator[None]:
    lock = _get_lock(session_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=wait_seconds)
    except TimeoutError:
        raise SessionBusyError(session_id) from None
    try:
        yield
    finally:
        lock.release()
