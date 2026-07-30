"""POST /v1/chat handler.

Walking skeleton (Issue #5), real system prompt (Issue #8), turn-zero prefix
(Issue #9), real orchestration loop with tool-call execution (Issue #10),
the full conversation memory system (Issue #11), and the booking flow's
intent/proposal/selection half (Issue #20): a deterministic, zero-LLM-token
intro on a brand-new session, then the memory-assembled context (pinned
profile + summary + token-budgeted window + booking guidance + current
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

Slot selection is matched deterministically in code (app/booking/selection.py),
*before* the agent loop runs, not inferred by the LLM (4.5) — a match fires
the slot_selected transition immediately, so the rest of the turn (tool
gating, booking guidance) already reflects the new step.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.intro import INTRO_MESSAGE
from app.agent.loop import run_agent
from app.agent.memory import (
    ToolEvent,
    assemble_messages,
    load_memory,
    persist_turn,
    run_eviction,
    run_reconciliation,
)
from app.agent.prompts import SYSTEM_PROMPT, render_booking_guidance
from app.agent.providers import get_provider
from app.agent.providers.base import LLMProvider
from app.agent.session_lock import session_turn_lock
from app.agent.tokens import effective_tokens
from app.agent.visitor_info import extract_linkedin_url, extract_name
from app.booking.confirmation import classify_confirmation_reply
from app.booking.selection import match_selection
from app.booking.state import Event, EventKind, Step, transition
from app.config import settings
from app.db.models import SessionRow
from app.db.session import (
    active_holds,
    add_token_budget_used,
    append_message,
    get_db,
    get_or_create_session,
    load_booking_state,
    recent_messages,
    save_booking_state,
    set_visitor_info,
)
from app.notify import get_notifier
from app.tools.calendar import CalendarClient, get_calendar_client
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


def get_chat_calendar_client() -> CalendarClient:
    """FastAPI dependency wrapping get_calendar_client() — mirrors
    app/api/health.py's get_health_calendar_client (Issue #17), overridable
    in tests with a FakeCalendar.
    """
    return get_calendar_client()


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


async def _slot_is_held_by_another_session(
    db: AsyncSession, slot: dict[str, object], *, exclude_session_id: str
) -> bool:
    """Guards the moment a hold is actually granted (Issue #21 review
    finding), not just proposal time — active_holds() is also consulted in
    app/tools/registry.py's calendar_find_slots, but a slot could still be
    proposed to two sessions before either selects, so the grant itself
    needs its own re-check.
    """
    held = await active_holds(db, exclude_session_id=exclude_session_id)
    start = datetime.fromisoformat(str(slot["start_iso"]))
    end = datetime.fromisoformat(str(slot["end_iso"]))
    return any(held_start < end and held_end > start for held_start, held_end in held)


_BOOKING_TOOL_NAMES = ("calendar_find_slots", "provide_contact_info", "calendar_create_booking")


def _booking_response(
    tool_events: list[ToolEvent],
) -> tuple[ResponseType, dict[str, object] | None]:
    """A turn whose tool activity produced a fresh slot proposal maps to
    `type: "booking_proposal"` with the slots in `data` (Issue #20, 4.8's
    documented shape: `{"slots": [...], "round": N}`); one that produced a
    confirmation summary maps to `type: "booking_confirmation_request"`
    (Issue #21, 4.8's `{"slot", "timezone", "name", "email"}` shape,
    already built exactly that way by provide_contact_info); one that
    actually created the event maps to `type: "booking_confirmed"` (Issue
    #22, 4.8's `{"booking_id", "slot", "timezone", "next_steps"}` shape,
    already built exactly that way by calendar_create_booking). Only the
    most recent booking-tool call in the turn is considered — the
    negotiation cap's email-fallback result, an invalid-email rejection,
    and a slot-taken-at-recheck bounce-back all have no dedicated response
    type in 4.8, so they surface as an ordinary "message" reply, with the
    model's own prose (informed by the tool result's "message" field)
    explaining it.
    """
    for event in reversed(tool_events):
        if event.name == "calendar_find_slots" and "slots" in event.result:
            return ResponseType.BOOKING_PROPOSAL, {
                "slots": event.result["slots"],
                "round": event.result["round"],
            }
        if event.name == "provide_contact_info" and "confirmation_summary" in event.result:
            summary = event.result["confirmation_summary"]
            assert isinstance(summary, dict)
            return ResponseType.BOOKING_CONFIRMATION_REQUEST, summary
        if event.name == "calendar_create_booking" and "booking_id" in event.result:
            return ResponseType.BOOKING_CONFIRMED, {
                "booking_id": event.result["booking_id"],
                "slot": event.result["slot"],
                "timezone": event.result["timezone"],
                "next_steps": event.result["next_steps"],
            }
        if event.name in _BOOKING_TOOL_NAMES:
            break  # most recent booking-tool call was an error/fallback, not a success
    return ResponseType.MESSAGE, None


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    provider: LLMProvider = Depends(get_main_provider),
    summarizer_provider: LLMProvider = Depends(get_summarizer_provider),
    calendar_client: CalendarClient = Depends(get_chat_calendar_client),
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

        # Deterministic slot selection (4.5): the visitor's raw reply is
        # matched against the stored proposal list in code, *before* the
        # agent loop runs — never inferred by the LLM. A match fires
        # slot_selected immediately, so tool gating and booking guidance
        # below already reflect the new step for this same turn. No match
        # leaves state untouched; the booking guidance for slots_proposed
        # tells the model to ask for clarification, which doesn't consume a
        # negotiation round (only a real calendar_find_slots call does).
        booking_state = await load_booking_state(db, request.session_id)
        if booking_state.step is Step.SLOTS_PROPOSED:
            matched_slot = match_selection(
                request.message, booking_state.proposed_slots_json or []
            )
            if matched_slot is not None and not await _slot_is_held_by_another_session(
                db, matched_slot, exclude_session_id=request.session_id
            ):
                # DB-only soft hold (Issue #21, 4.5) — never written to the
                # calendar. calendar_find_slots (app/tools/registry.py)
                # treats other sessions' unexpired holds as busy via
                # app/db/session.py's active_holds(), but that's only
                # checked at *proposal* time — two sessions could still be
                # offered the same slot before either selects it, so this
                # re-checks right before actually granting the hold too
                # (review finding: without this, both could reach
                # slot_selected/confirmed for the identical slot). Still a
                # narrow, best-effort soft-hold guarantee, not a hard one —
                # the real guarantee against a genuine double-booking is
                # Issue #22's free/busy re-check right before the actual
                # Google Calendar event gets created.
                hold_expires_at = datetime.now(UTC) + timedelta(minutes=settings.hold_minutes)
                booking_state = transition(
                    booking_state,
                    Event(
                        kind=EventKind.SLOT_SELECTED,
                        slot=matched_slot,
                        hold_expires_at=hold_expires_at,
                    ),
                )
                await save_booking_state(db, booking_state)

        # Deterministic confirmation-reply classification (Issue #22, 4.5:
        # "Only an affirmative reply advances") — never left to the LLM's
        # own judgment, mirroring slot selection above. A clear "no"
        # re-enters the negotiation immediately (counts as a round, per
        # CONFIRMATION_DECLINED); "affirmative" only flags the tool as
        # legal to call this turn — the actual booking-creation side
        # effects still only ever happen inside calendar_create_booking's
        # own handler. "unclear" (a question, hesitation) leaves both state
        # and the flag untouched, so the model just answers and keeps
        # waiting, per the confirmed-step guidance below.
        confirmation_is_affirmative = False
        if booking_state.step is Step.CONFIRMED:
            classification = classify_confirmation_reply(request.message)
            if classification == "negative":
                booking_state = transition(
                    booking_state, Event(kind=EventKind.CONFIRMATION_DECLINED)
                )
                await save_booking_state(db, booking_state)
            elif classification == "affirmative":
                confirmation_is_affirmative = True

        memory = await load_memory(
            db,
            session,
            booking_timezone=booking_state.timezone_name,
            booking_contact_email=booking_state.contact_email,
        )
        messages = assemble_messages(
            memory,
            SYSTEM_PROMPT,
            request.message,
            booking_guidance=render_booking_guidance(booking_state.step),
        )

        # The user's message is persisted before the provider call, per
        # 4.3's "the DB log is complete" invariant — even a message that
        # triggers an upstream failure (rate limit, outage) must remain in
        # the audit log. persist_turn (below) is told not to re-append it.
        await append_message(db, request.session_id, "user", request.message)

        tool_context = ToolContext(
            db=db,
            session_id=request.session_id,
            summarizer_provider=summarizer_provider,
            calendar_client=calendar_client,
            caller_timezone=request.timezone,
            confirmation_is_affirmative=confirmation_is_affirmative,
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

        response_type, response_data = _booking_response(result.tool_events)

        needs_eviction = await persist_turn(
            db,
            request.session_id,
            request.message,
            result.text,
            result.tool_events,
            user_already_persisted=True,
            response_type=response_type.value,
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

        # calendar_create_booking (app/tools/registry.py) only records
        # intent on tool_context during the loop above too — same rationale
        # as reconciliation: a notification failure must never affect this
        # turn's reply, and the booking it's reporting on is already
        # durably persisted regardless (Issue #22).
        for subject, body in tool_context.pending_owner_notifications:
            background_tasks.add_task(get_notifier().notify, subject, body)

        return ChatResponse(reply=result.text, type=response_type, data=response_data)
