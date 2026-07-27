"""POST /v1/chat handler.

Walking skeleton (Issue #5, real system prompt wired in Issue #8, turn-zero
prefix + visitor-info capture wired in Issue #9): a deterministic,
zero-LLM-token intro on a brand-new session, then SYSTEM_PROMPT +
last-10-turns history + the user's message on every later turn. No memory
system (#11), no classifier (#25), no orchestration loop with tools (#10)
yet — those upgrade this handler in later issues without changing the
envelope shape.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.intro import INTRO_MESSAGE
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.providers import get_provider
from app.agent.providers.base import LLMProvider
from app.agent.providers.base import Message as LLMMessage
from app.agent.visitor_info import extract_linkedin_url, extract_name
from app.config import settings
from app.db.models import SessionRow
from app.db.session import (
    append_message,
    get_db,
    get_or_create_session,
    recent_messages,
    set_visitor_info,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_HISTORY_TURNS = 10
_HISTORY_ROWS = _HISTORY_TURNS * 2  # each turn is one user row + one assistant row


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


async def _maybe_capture_visitor_info(
    db: AsyncSession, session: SessionRow, message: str
) -> tuple[str | None, str | None]:
    """Extract + persist a volunteered name/LinkedIn (Issue #9), capturing
    each at most once per session — a field already set is never
    overwritten. Returns whichever were newly captured this turn (None if
    already set or not present in this message), so the caller can decide
    whether to acknowledge.
    """
    newly_name = extract_name(message) if session.visitor_name is None else None
    newly_linkedin = extract_linkedin_url(message) if session.visitor_linkedin is None else None
    if newly_name or newly_linkedin:
        await set_visitor_info(db, session.id, name=newly_name, linkedin=newly_linkedin)
    return newly_name, newly_linkedin


def _acknowledgment_suffix(name: str | None, linkedin: str | None) -> str | None:
    if name and linkedin:
        return f"(Thanks for sharing your name and LinkedIn, {name}!)"
    if name:
        return f"(Thanks for sharing your name, {name}!)"
    if linkedin:
        return "(Thanks for sharing your LinkedIn!)"
    return None


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
    provider: LLMProvider = Depends(get_main_provider),
) -> ChatResponse:
    # Read by the logging middleware (Issue #6) after this handler returns.
    http_request.state.session_id = request.session_id

    session = await get_or_create_session(db, request.session_id)
    is_turn_zero = len(await recent_messages(db, request.session_id, n=1)) == 0

    # Extraction runs every turn, including turn zero (Issue #9: "this turn
    # or any later one") — but turn zero's *reply* stays byte-identical to
    # intro.md regardless, so any capture here is acknowledged silently.
    newly_name, newly_linkedin = await _maybe_capture_visitor_info(db, session, request.message)

    if is_turn_zero:
        await append_message(db, request.session_id, "user", request.message)
        await append_message(db, request.session_id, "assistant", INTRO_MESSAGE)
        return ChatResponse(reply=INTRO_MESSAGE, type=ResponseType.MESSAGE, data=None)

    history = await recent_messages(db, request.session_id, n=_HISTORY_ROWS)

    messages = [LLMMessage(role="system", content=SYSTEM_PROMPT)]
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

    reply_text = response.text
    ack = _acknowledgment_suffix(newly_name, newly_linkedin)
    if ack:
        reply_text = f"{reply_text}\n\n{ack}"

    await append_message(db, request.session_id, "assistant", reply_text)

    return ChatResponse(reply=reply_text, type=ResponseType.MESSAGE, data=None)
