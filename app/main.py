"""FastAPI app instantiation, middleware wiring, and route registration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.db.session import init_db


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    yield


app = FastAPI(title="Personal AI Assistant Agent", lifespan=lifespan)

register_exception_handlers(app)
app.include_router(chat_router)
app.include_router(health_router)
