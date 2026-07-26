"""FastAPI app instantiation, middleware wiring, and route registration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.db.session import init_db
from app.middleware.logging import LoggingMiddleware


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    yield


app = FastAPI(title="Personal AI Assistant Agent", lifespan=lifespan)

register_exception_handlers(app)
# Added after exception handlers: middleware wraps outside FastAPI's
# exception-handling layer, so call_next() here returns the final response
# (including a 422/503/500 already converted by the handlers above) — the
# log line always reflects the real status code sent to the client.
app.add_middleware(LoggingMiddleware)
app.include_router(chat_router)
app.include_router(health_router)
