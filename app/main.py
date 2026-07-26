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
# Registration order relative to add_middleware has no effect on request
# handling — Starlette's build_middleware_stack() routes each handler by
# exception type regardless of call order (see app/middleware/logging.py's
# module docstring for why LoggingMiddleware still logs the correct status
# for every path, including the generic-Exception one bound to
# ServerErrorMiddleware, which always sits outside any add_middleware layer).
app.add_middleware(LoggingMiddleware)
app.include_router(chat_router)
app.include_router(health_router)
