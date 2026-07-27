import asyncio
import contextlib

import pytest

from app.agent.session_lock import SessionBusyError, session_turn_lock

# tests/conftest.py's autouse _reset_session_locks fixture clears
# app.agent.session_lock._locks before every test in the suite — see its
# docstring for why that's needed even within this file alone.


async def test_lock_is_reentrant_free_sequential_acquire_release() -> None:
    async with session_turn_lock("s1"):
        pass
    async with session_turn_lock("s1"):
        pass  # must not deadlock — the first context fully released


async def test_different_sessions_do_not_block_each_other() -> None:
    order: list[str] = []

    async def hold(session_id: str, seconds: float) -> None:
        async with session_turn_lock(session_id):
            order.append(f"{session_id}-start")
            await asyncio.sleep(seconds)
            order.append(f"{session_id}-end")

    await asyncio.gather(hold("s1", 0.05), hold("s2", 0.05))
    # Both started before either finished — they ran concurrently, not
    # serialized against each other (only same-session turns serialize).
    assert order.index("s1-start") < order.index("s2-end")
    assert order.index("s2-start") < order.index("s1-end")


async def test_concurrent_same_session_requests_serialize() -> None:
    order: list[str] = []

    async def hold(tag: str, seconds: float) -> None:
        async with session_turn_lock("s1"):
            order.append(f"{tag}-start")
            await asyncio.sleep(seconds)
            order.append(f"{tag}-end")

    await asyncio.gather(hold("a", 0.05), hold("b", 0.01))
    # Whichever ran first must fully finish before the second starts.
    assert order in (
        ["a-start", "a-end", "b-start", "b-end"],
        ["b-start", "b-end", "a-start", "a-end"],
    )


async def test_blocked_request_raises_session_busy_after_wait_timeout() -> None:
    async def hold_forever() -> None:
        async with session_turn_lock("s1"):
            await asyncio.sleep(1)

    holder = asyncio.create_task(hold_forever())
    await asyncio.sleep(0.01)  # let the holder acquire first

    with pytest.raises(SessionBusyError):
        async with session_turn_lock("s1", wait_seconds=0.05):
            pass  # pragma: no cover — never reached

    holder.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await holder


async def test_lock_released_after_exception_inside_the_block() -> None:
    with contextlib.suppress(ValueError):
        async with session_turn_lock("s1"):
            raise ValueError("boom")

    # If the lock weren't released on exception, this would time out.
    async with session_turn_lock("s1", wait_seconds=0.1):
        pass
