"""POST /v1/chat handler.

Walking skeleton (Issue #5): placeholder system prompt + last-10-turns
history + the user's message, sent straight to the LLM, both turns
persisted. No memory system (#11), no classifier (#25), no orchestration
loop with tools (#10), no turn-zero prefix (#9) yet — those upgrade this
handler in later issues without changing the envelope shape.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.providers import get_provider
from app.agent.providers.base import LLMProvider
from app.agent.providers.base import Message as LLMMessage
from app.config import settings
from app.db.session import append_message, get_db, get_or_create_session, recent_messages

logger = logging.getLogger(__name__)

router = APIRouter()

_HISTORY_TURNS = 10
_HISTORY_ROWS = _HISTORY_TURNS * 2  # each turn is one user row + one assistant row

_PLACEHOLDER_SYSTEM_PROMPT = (
    "You are a helpful assistant. This is a placeholder system prompt; "
    "Issue #8 replaces it with the real persona and scope rules."
)


class ResponseType(StrEnum):
    MESSAGE = "message"
    REFUSAL = "refusal"
    BOOKING_PROPOSAL = "booking_proposal"
    BOOKING_CONFIRMATION_REQUEST = "booking_confirmation_request"
    BOOKING_CONFIRMED = "booking_confirmed"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=2000)
    timezone: str | None = None


class ChatResponse(BaseModel):
    reply: str
    type: ResponseType
    data: dict[str, object] | None = None


def get_main_provider() -> LLMProvider:
    """FastAPI dependency wrapping get_provider("main") — overridable in
    tests with a FakeProvider, unlike a direct module-level call.
    """
    return get_provider("main")


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    provider: LLMProvider = Depends(get_main_provider),
) -> ChatResponse:
    # Read by the logging middleware (Issue #6) after this handler returns.
    http_request.state.session_id = request.session_id

    await get_or_create_session(db, request.session_id)
    history = await recent_messages(db, request.session_id, n=_HISTORY_ROWS)

    messages = [LLMMessage(role="system", content=_PLACEHOLDER_SYSTEM_PROMPT)]
    messages.extend(
        LLMMessage(role=m.role, content=m.content)  # type: ignore[arg-type]
        for m in history
        if m.role in ("user", "assistant")  # tool-role replay needs pairing logic; Issue #10
    )
    messages.append(LLMMessage(role="user", content=request.message))

    # The user's message is persisted before the provider call, per 4.3's
    # "the DB log is complete" invariant — even a message that triggers an
    # upstream failure (rate limit, outage) must remain in the audit log.
    await append_message(db, request.session_id, "user", request.message)

    response = await provider.complete(
        messages=messages, tools=[], max_tokens=settings.max_tokens_per_turn
    )
    http_request.state.llm_tokens_in = response.usage.input_tokens
    http_request.state.llm_tokens_out = response.usage.output_tokens

    await append_message(db, request.session_id, "assistant", response.text)

    return ChatResponse(reply=response.text, type=ResponseType.MESSAGE, data=None)
