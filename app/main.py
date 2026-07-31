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
from app.db.session import get_session_factory, init_db
from app.middleware.logging import LoggingMiddleware
from app.middleware.rate_limit import RateLimitMiddleware

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
    # RateLimitMiddleware (Issue #24) is raw ASGI middleware, outside
    # FastAPI's dependency-injection tree, so it can't be pointed at a
    # test's isolated DB via app.dependency_overrides the way get_db is —
    # it reads this app.state attribute instead (a test's client fixture
    # sets the same attribute to its own test session_factory).
    _app.state.db_session_factory = get_session_factory()
    yield


app = FastAPI(title="Personal AI Assistant Agent", lifespan=lifespan)

register_exception_handlers(app)
# Registration order relative to *exception-handler* routing has no effect
# — Starlette's build_middleware_stack() routes each handler by exception
# type regardless of add_middleware call order (see app/middleware/
# logging.py's module docstring for why LoggingMiddleware still logs the
# correct status for every path, including the generic-Exception one bound
# to ServerErrorMiddleware, which always sits outside any add_middleware
# layer). Order *between the two middlewares below* does matter, though, and
# is easy to get backwards (a review pass caught this exact mistake here):
# Starlette's add_middleware() *prepends* to its internal list
# (`self.user_middleware.insert(0, ...)`), and build_middleware_stack()
# then wraps by iterating that list in reverse — so the *last*-added
# middleware ends up OUTERMOST, not innermost. Registering RateLimitMiddleware
# after LoggingMiddleware (as an earlier version of this file did) would put
# RateLimitMiddleware outermost instead, so every 429 it short-circuits
# before the request reaches routing would never reach LoggingMiddleware at
# all and would go unlogged. Registering RateLimitMiddleware *first* (below)
# makes LoggingMiddleware the last-added, and therefore outermost, layer —
# verified directly against this app object, not just reasoned about.
app.add_middleware(RateLimitMiddleware)
app.add_middleware(LoggingMiddleware)
app.include_router(chat_router)
app.include_router(health_router)
