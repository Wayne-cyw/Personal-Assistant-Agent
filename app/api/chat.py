"""POST /v1/chat handler.

Walking skeleton (Issue #5), real system prompt (Issue #8), turn-zero prefix
(Issue #9), real orchestration loop with tool-call execution (Issue #10),
and the full conversation memory system (Issue #11): a deterministic,
zero-LLM-token intro on a brand-new session, then the memory-assembled
context (pinned profile + summary + token-budgeted window + current
message) run through the tool-calling agent loop on every later turn. Turn
processing is serialized per session_id (4.3's concurrency guard). No
classifier (#25) yet — that upgrades this handler in a later issue without
changing the envelope shape.

Post-turn-zero visitor-info capture is now tool-based (save_visitor_info,
app/tools/registry.py), replacing Issue #9's regex extractor for every turn
except turn zero itself — turn zero spends zero LLM tokens by design, so a
tool call is structurally impossible there, and the regex extractor remains
the only option for that one turn (see `_maybe_capture_visitor_info_turn_
zero`).
"""

from __future__ import annotations

import logging
from enum import StrEnum

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.intro import INTRO_MESSAGE
from app.agent.loop import run_agent
from app.agent.memory import (
    assemble_messages,
    load_memory,
    persist_turn,
    run_eviction,
    run_reconciliation,
)
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.providers import get_provider
from app.agent.providers.base import LLMProvider
from app.agent.session_lock import session_turn_lock
from app.agent.tokens import effective_tokens
from app.agent.visitor_info import extract_linkedin_url, extract_name
from app.config import settings
from app.db.models import SessionRow
from app.db.session import (
    add_token_budget_used,
    append_message,
    get_db,
    get_or_create_session,
    recent_messages,
    set_visitor_info,
)
from app.tools.context import ToolContext

logger = logging.getLogger(__name__)

router = APIRouter()


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


def get_summarizer_provider() -> LLMProvider:
    """Mirrors get_main_provider — a separate call site/dependency (4.3's
    role-isolation rule), independently overridable in tests.
    """
    return get_provider("summarizer")


async def _maybe_capture_visitor_info_turn_zero(
    db: AsyncSession, session: SessionRow, message: str
) -> None:
    """Turn zero's regex-based fallback (Issue #9): the only turn with zero
    LLM calls by design, so the tool-based save_visitor_info path (Issue
    #11, used for every later turn) is structurally unavailable here.
    Capture-once — a field already set is never overwritten, since a false
    positive from this conservative-but-imperfect regex has no correction
    path on this turn (the reply must stay byte-identical to intro.md, so
    nothing can be acknowledged or corrected here regardless).
    """
    newly_name = extract_name(message) if session.visitor_name is None else None
    newly_linkedin = extract_linkedin_url(message) if session.visitor_linkedin is None else None
    if newly_name or newly_linkedin:
        await set_visitor_info(db, session.id, name=newly_name, linkedin=newly_linkedin)


def _session_budget_exceeded_message() -> str:
    return (
        "This conversation has reached its limit — email the owner directly at "
        f"{settings.owner_contact_email}."
    )


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    provider: LLMProvider = Depends(get_main_provider),
    summarizer_provider: LLMProvider = Depends(get_summarizer_provider),
) -> ChatResponse:
    # Read by the logging middleware (Issue #6) after this handler returns.
    http_request.state.session_id = request.session_id

    async with session_turn_lock(request.session_id):
        session = await get_or_create_session(db, request.session_id)
        is_turn_zero = len(await recent_messages(db, request.session_id, n=1)) == 0

        if is_turn_zero:
            # Capture happens silently on turn zero even though this turn's
            # reply stays byte-identical to intro.md regardless (Issue #9).
            await _maybe_capture_visitor_info_turn_zero(db, session, request.message)
            await append_message(db, request.session_id, "user", request.message)
            await append_message(db, request.session_id, "assistant", INTRO_MESSAGE)
            return ChatResponse(reply=INTRO_MESSAGE, type=ResponseType.MESSAGE, data=None)

        # Cost-control guardrail (Engineering Guide 4.6 layer 4, Issue #12):
        # a session that has already spent its lifetime budget gets a
        # static wrap-up message and zero LLM calls — but the message
        # itself is still logged, per 4.3's "DB log is complete" invariant.
        # Deliberately permanent for the rest of the session's life once
        # crossed (token_budget_used only ever increases, and there's no
        # reset path) — the plain reading of "lifetime budget" in the issue
        # body, and simpler than trying to distinguish "trivial" turns that
        # might still deserve a real reply from ones that don't.
        if session.token_budget_used >= settings.session_token_budget:
            await append_message(db, request.session_id, "user", request.message)
            reply_text = _session_budget_exceeded_message()
            await append_message(db, request.session_id, "assistant", reply_text)
            return ChatResponse(reply=reply_text, type=ResponseType.MESSAGE, data=None)

        memory = await load_memory(db, session)
        messages = assemble_messages(memory, SYSTEM_PROMPT, request.message)

        # The user's message is persisted before the provider call, per
        # 4.3's "the DB log is complete" invariant — even a message that
        # triggers an upstream failure (rate limit, outage) must remain in
        # the audit log. persist_turn (below) is told not to re-append it.
        await append_message(db, request.session_id, "user", request.message)

        tool_context = ToolContext(
            db=db, session_id=request.session_id, summarizer_provider=summarizer_provider
        )
        # CHAT_MAX_OUTPUT_TOKENS is the tighter, chat-reply-specific cap
        # (pairs with #8's brevity rule); MAX_TOKENS_PER_TURN remains the
        # outer ceiling enforced on every provider call (Issue #12) — the
        # summarizer's own call is clamped the same way in
        # app/agent/memory.py's _summarize.
        max_tokens = min(settings.chat_max_output_tokens, settings.max_tokens_per_turn)
        result = await run_agent(
            messages,
            provider,
            max_tokens,
            max_iterations=settings.max_iterations,
            tool_context=tool_context,
        )
        http_request.state.llm_tokens_in = result.input_tokens
        http_request.state.llm_tokens_out = result.output_tokens
        await add_token_budget_used(
            db,
            request.session_id,
            effective_tokens(
                input_tokens=result.input_tokens,
                cached_input_tokens=result.cached_input_tokens,
                output_tokens=result.output_tokens,
            ),
        )

        needs_eviction = await persist_turn(
            db,
            request.session_id,
            request.message,
            result.text,
            result.tool_events,
            user_already_persisted=True,
        )
        if needs_eviction:
            background_tasks.add_task(run_eviction, request.session_id, summarizer_provider)

        # flag_summary_conflict (app/tools/registry.py) only records intent
        # on tool_context during the loop above — the actual reconciliation
        # call runs here as a background task, same as eviction, so it
        # never adds a summarizer round trip to this response.
        for explanation in tool_context.pending_reconciliations:
            background_tasks.add_task(
                run_reconciliation,
                request.session_id,
                summarizer_provider,
                trigger="model_detected",
                detail=explanation,
            )

        return ChatResponse(reply=result.text, type=ResponseType.MESSAGE, data=None)
