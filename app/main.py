"""FastAPI app instantiation, middleware wiring, and route registration."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.config import settings
from app.db.session import init_db
from app.middleware.logging import LoggingMiddleware

logger = logging.getLogger(__name__)


def _check_single_worker() -> None:
    """Engineering Guide 4.3: the per-session lock (app/agent/session_lock.py)
    is only correct with exactly one worker process — enforced for real in
    Issue #34's deploy config; this is the lighter in-app guard 4.3 also
    calls for. WEB_CONCURRENCY (the common gunicorn/uvicorn-worker-manager
    env var, read via settings.web_concurrency — Issue #2's rule that only
    app/config.py reads the process environment directly) is the best
    available signal: a bare `uvicorn --workers N` invocation has no other
    way to tell a single process how many siblings it has, so a deployment
    that sets --workers without also setting this env var will not be
    caught here.
    """
    workers = settings.web_concurrency
    if workers is None or workers <= 1:
        return

    message = (
        f"WEB_CONCURRENCY={workers} — this app's per-session concurrency lock "
        "(app/agent/session_lock.py) is an in-process asyncio.Lock and is only "
        "correct with exactly one worker (Engineering Guide 4.3). Running "
        "multiple workers silently reintroduces booking-state/memory races."
    )
    if settings.enforce_single_worker:
        raise RuntimeError(message)
    logger.error(message)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    _check_single_worker()
    await init_db()
    yield


app = FastAPI(title="Personal AI Assistant Agent", lifespan=lifespan)

register_exception_handlers(app)
# Registration order relative to add_middleware has no effect on request
# handling — Starlette's build_middleware_stack() routes each handler by
# exception type regardless of call order (see app/middleware/logging.py's
# module docstring for why LoggingMiddleware still logs the correct status
# for every path, including the generic-Exception one bound to
# ServerErrorMiddleware, which always sits outside any add_middleware layer).
app.add_middleware(LoggingMiddleware)
app.include_router(chat_router)
app.include_router(health_router)
