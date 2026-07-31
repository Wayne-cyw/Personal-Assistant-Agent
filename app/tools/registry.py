"""Tool definitions, argument validation, and execute_tool dispatch
(Engineering Guide 4.2, Issue #10; save_visitor_info/flag_summary_conflict
added in Issue #11; calendar_find_slots added in Issue #20).

`get_current_date` is a trivial proof tool — real tools (RAG search,
calendar) register here the same way in later issues (#15).
"""

from __future__ import annotations

import logging
import re
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field, ValidationError

from app.agent.providers.base import ToolCall, ToolDef
from app.agent.visitor_info import extract_linkedin_url
from app.booking.slots import (
    AvailabilityPolicyError,
    CoarseWindow,
    generate_slots,
    get_availability_policy,
    resolve_window,
    widen_window,
)
from app.booking.state import (
    BookingState,
    Event,
    EventKind,
    Step,
    allowed_tools_for,
    can_abandon,
    next_proposal_event,
    transition,
)
from app.config import settings
from app.db.session import (
    active_holds,
    add_pinned_fact,
    check_and_increment_rate_limit,
    create_booking,
    flag_session,
    load_booking_state,
    save_booking_state,
    set_visitor_info,
)
from app.safety.pii import redact
from app.tools.calendar import Attendee, CalendarError
from app.tools.context import ToolContext

logger = logging.getLogger(__name__)

_NAME_MAX_LENGTH = 100
_FACT_MAX_LENGTH = 200


class GetCurrentDateArgs(BaseModel):
    """No arguments — the tool takes none."""


async def _get_current_date(_args: BaseModel, _context: ToolContext) -> dict[str, object]:
    now = datetime.now(UTC)
    return {"date": now.date().isoformat(), "iso": now.isoformat()}


class SaveVisitorInfoArgs(BaseModel):
    name: str | None = None
    linkedin: str | None = None
    fact: str | None = None


def _sanitize_short_text(value: str, max_length: int) -> str:
    # Replace (not delete) non-printable characters before collapsing
    # whitespace and clamping — a pinned-profile field is rendered verbatim
    # into every future prompt (Engineering Guide 4.6: visitor-volunteered
    # strings are data, and a multi-line/oddly-formatted value shouldn't be
    # able to smuggle extra structure into that block). Replacing with a
    # space rather than deleting matters: deleting a newline glues the
    # words on either side of it together ("no\nthing" -> "nothing"),
    # silently changing the text's meaning instead of just its formatting.
    cleaned = "".join(ch if ch.isprintable() else " " for ch in value)
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_length]


async def _save_visitor_info(args: BaseModel, context: ToolContext) -> dict[str, object]:
    assert isinstance(args, SaveVisitorInfoArgs)
    saved: dict[str, str] = {}
    rejected: dict[str, str] = {}

    name: str | None = None
    if args.name is not None:
        cleaned_name = _sanitize_short_text(args.name, _NAME_MAX_LENGTH)
        if cleaned_name:
            name = cleaned_name
            saved["name"] = cleaned_name
        else:
            rejected["name"] = "empty after sanitization"

    linkedin: str | None = None
    if args.linkedin is not None:
        matched = extract_linkedin_url(args.linkedin)
        if matched is not None:
            linkedin = matched
            saved["linkedin"] = matched
        else:
            rejected["linkedin"] = "not a recognizable linkedin.com/in/... URL"

    if name is not None or linkedin is not None:
        await set_visitor_info(context.db, context.session_id, name=name, linkedin=linkedin)

    if args.fact is not None:
        cleaned_fact = _sanitize_short_text(args.fact, _FACT_MAX_LENGTH)
        if not cleaned_fact:
            rejected["fact"] = "empty after sanitization"
        else:
            added = await add_pinned_fact(
                context.db, context.session_id, cleaned_fact, max_facts=settings.pinned_facts_max
            )
            if added:
                saved["fact"] = cleaned_fact
            else:
                rejected["fact"] = "duplicate, or the pinned-facts limit is already reached"

    return {"saved": saved, "rejected": rejected}


class FlagSummaryConflictArgs(BaseModel):
    explanation: str


async def _flag_summary_conflict(args: BaseModel, context: ToolContext) -> dict[str, object]:
    """Records the conflict for app/api/chat.py to schedule as a background
    reconciliation (app/agent/memory.py's run_reconciliation) rather than
    running reload_and_reconcile here — that makes a real summarizer LLM
    call, and reconciliation must not add user-facing latency to the turn
    any more than eviction does (Engineering Guide 4.3).
    """
    assert isinstance(args, FlagSummaryConflictArgs)
    context.pending_reconciliations.append(args.explanation)
    return {"acknowledged": True}


async def _calendar_find_slots(args: BaseModel, context: ToolContext) -> dict[str, object]:
    """Detects booking intent, ensures the timezone is known, and proposes
    slots — all three of 4.5's intent_detected/slots_proposed row behaviors,
    since there is no classifier yet (Issue #25) and state transitions only
    ever happen inside execute_tool/handler code, never inferred from the
    LLM's prose (4.5). This call itself is what signals "the visitor wants
    to book" (see app/booking/state.py's _ALLOWED_TOOLS comment on why idle
    is gated to this tool).
    """
    assert isinstance(args, CoarseWindow)

    if context.calendar_find_slots_used_this_turn:
        # A negotiation round is meant to correspond to one visitor turn
        # (4.5) — refuse a second call within the same turn rather than
        # letting it silently advance proposal_rounds again (see
        # ToolContext.calendar_find_slots_used_this_turn's docstring).
        return {
            "error": (
                "calendar_find_slots was already called once this turn. Wait for the "
                "visitor's next reply before proposing or re-proposing again."
            )
        }
    context.calendar_find_slots_used_this_turn = True

    state = await load_booking_state(context.db, context.session_id)

    if state.step is Step.IDLE:
        state = transition(state, Event(kind=EventKind.INTENT_DETECTED))

    if state.step not in (Step.INTENT_DETECTED, Step.SLOTS_PROPOSED):
        # Defense in depth (4.2): tool_defs_for_step already gates this per
        # LLM call, but a turn with several tool calls could see state move
        # past these steps between calls (e.g. an earlier call in the same
        # turn already hit the negotiation cap's email fallback).
        await save_booking_state(context.db, state)
        return {"error": "booking is not currently accepting a new time proposal"}

    if state.timezone_name is None:
        if context.caller_timezone:
            state = transition(
                state, Event(kind=EventKind.TIMEZONE_CAPTURED, timezone=context.caller_timezone)
            )
        else:
            await save_booking_state(context.db, state)
            return {
                "error": "timezone_required",
                "message": (
                    "Ask the visitor for their IANA timezone (e.g. 'America/Toronto') before "
                    "proposing times — none has been provided yet."
                ),
            }
    assert state.timezone_name is not None
    # Captured into a local: BookingState isn't frozen, so mypy discards the
    # assert's narrowing of state.timezone_name across the function calls
    # between here and generate_slots below.
    timezone_name: str = state.timezone_name

    try:
        policy = get_availability_policy()
    except AvailabilityPolicyError:
        logger.error("calendar_find_slots called but availability policy is not configured")
        await save_booking_state(context.db, state)
        return {
            "error": (
                "Availability isn't configured yet — offer to have the owner follow up by "
                "email instead."
            )
        }

    # Booking-specific rate limits (Issue #23, PRD: "booking abuse is
    # costlier than Q&A abuse" — stricter and separate from #24's general
    # chat-message limits). Checked here — after the timezone/policy
    # guards above, right before the call actually does its real work
    # (compute a proposal, re-propose, widen, or land on email fallback) —
    # not earlier: a call that only asked for a missing timezone, or that
    # bounced off a step mismatch or an unconfigured availability policy,
    # never produced a real proposal and shouldn't burn one of the
    # visitor's limited attempts (review finding: checking any earlier
    # meant a visitor who simply hadn't sent their timezone yet could be
    # rate-limited and flagged before ever getting a real proposal, and
    # the negotiation flow's own legitimate proposal/re-propose/widen/
    # fallback call count — 4, the default this cap is tuned to — no
    # longer matched what was actually being counted).
    session_allowed = await check_and_increment_rate_limit(
        context.db,
        f"book:sess:{context.session_id}",
        limit=settings.booking_attempts_per_session,
        window=None,
    )
    ip_allowed = True
    if context.caller_ip is not None:
        ip_allowed = await check_and_increment_rate_limit(
            context.db,
            f"book:ip:{context.caller_ip}",
            limit=settings.booking_attempts_per_ip_per_day,
            window=timedelta(days=1),
        )
    if not session_allowed or not ip_allowed:
        # "state -> abandoned" (issue text) only applies once there's an
        # actual in-progress flow to abandon (can_abandon, the same guard
        # transition()'s own ABANDON branch uses — shared rather than
        # duplicated, so the two can't silently drift apart). From idle --
        # possible only when this session's very first attempt is blocked
        # because it shares an already-exhausted IP with other sessions --
        # there's nothing to abandon, and the rate limiter itself (not the
        # FSM step) is what keeps blocking this session's future attempts
        # regardless of step, so leaving state at idle is still correctly
        # locked out.
        if can_abandon(state.step):
            state = transition(state, Event(kind=EventKind.ABANDON))
        await save_booking_state(context.db, state)
        if not session_allowed:
            # Only flag when *this* session's own attempts tripped its own
            # cap -- not when it's merely an innocent bystander blocked by
            # another session having already exhausted their shared IP's
            # daily quota (review finding: flagging every IP-capped
            # session as suspicious would false-positive on shared
            # office/NAT IPs once flagged sessions start driving owner
            # notifications, Issues #16/#27).
            await flag_session(context.db, context.session_id)
        return {
            "rate_limited": True,
            "message": (
                "The visitor has exceeded the limit on booking attempts for this "
                "conversation. Do not call calendar_find_slots again -- tell them to email "
                f"{settings.owner_contact_email} directly to schedule a call."
            ),
        }

    event_kind = next_proposal_event(state)

    if event_kind is EventKind.EMAIL_FALLBACK:
        state = transition(state, Event(kind=EventKind.EMAIL_FALLBACK))
        await save_booking_state(context.db, state)
        return {
            "fallback": True,
            "message": (
                "No slot has worked even after widening the search window — tell the visitor "
                f"to email {settings.owner_contact_email} directly to find a time."
            ),
        }

    resolved = resolve_window(args, policy)
    if event_kind is EventKind.WIDEN_WINDOW:
        resolved = widen_window(resolved)

    # Other sessions' unexpired soft holds (Issue #21) are additional busy
    # time on every call, first round included — two visitors must never
    # both be offered the same slot, not just re-proposals.
    exclude: list[tuple[datetime, datetime]] = list(
        await active_holds(context.db, exclude_session_id=context.session_id)
    )

    # Re-proposing or widening also excludes everything offered and
    # rejected across the *whole* negotiation so far, not just the
    # immediately preceding round — state.excluded_slots_json (the
    # accumulator) plus the current round's own proposed_slots_json, which
    # is about to be superseded and folded into that same accumulator by
    # transition() below. Without the accumulator half, a round-3 widened
    # window could re-offer a round-1 slot the visitor already rejected
    # twice, since RE_PROPOSE/WIDEN_WINDOW both *replace* proposed_slots_json
    # each round rather than growing it.
    if event_kind in (EventKind.RE_PROPOSE, EventKind.WIDEN_WINDOW):
        already_offered = list(state.excluded_slots_json or []) + list(
            state.proposed_slots_json or []
        )
        exclude.extend(
            (
                datetime.fromisoformat(str(s["start_iso"])),
                datetime.fromisoformat(str(s["end_iso"])),
            )
            for s in already_offered
        )

    try:
        slots = await generate_slots(
            context.calendar_client,
            resolved,
            policy,
            timezone_name,
            now=datetime.now(UTC),
            exclude=exclude,
        )
    except Exception:
        # The rate-limit increment above has already committed by this
        # point (review finding) — a failure here still costs the visitor
        # one of their limited attempts, a known, accepted trade-off
        # (refunding it would need a decrement primitive and reintroduce
        # exactly the check-then-write race the atomic upsert exists to
        # avoid for the per-IP counter, shared across sessions that aren't
        # otherwise serialized against each other). What *is* fixed here:
        # without this except block, an exception here would propagate
        # past every save_booking_state call in this function, silently
        # discarding whatever state.py transitions already happened
        # earlier in this same call (idle -> intent_detected, timezone_
        # captured) — persisting them (this session's honest progress
        # toward its own cap, not the failure) before degrading
        # gracefully, same pattern as the AvailabilityPolicyError case
        # above. Deliberately broad (not just CalendarError, the review
        # finding's original trigger): generate_slots also constructs a
        # ZoneInfo from the caller-supplied timezone_name, which is never
        # format-validated before reaching here (ChatRequest.timezone has
        # no validator, TIMEZONE_CAPTURED only checks truthiness) and
        # raises ZoneInfoNotFoundError, not CalendarError, on a malformed
        # value — narrowing this to CalendarError would have left that
        # trigger (fully caller-controlled, no real Calendar outage
        # needed) still silently discarding progress.
        logger.error(
            "calendar_find_slots: failed to generate slots for session %s", context.session_id
        )
        await save_booking_state(context.db, state)
        return {
            "error": (
                "Something went wrong checking availability — offer to have the owner follow "
                "up by email instead."
            )
        }
    slot_dicts = [slot.model_dump() for slot in slots]
    state = transition(state, Event(kind=event_kind, slots=slot_dicts))
    await save_booking_state(context.db, state)
    return {"slots": slot_dicts, "round": state.proposal_rounds}


class ProvideContactInfoArgs(BaseModel):
    name: str = Field(description="The visitor's full name, exactly as they gave it.")
    email: str = Field(description="The visitor's email address, exactly as they gave it.")


_EMAIL_MAX_LENGTH = 254  # RFC 5321's overall address length limit
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def _validate_email(raw: str) -> str | None:
    """Regex + length + no-injection-chars validation (Issue #21) — code-
    validated, never trusted from the LLM's own judgment of what "looks
    like an email." Returns the cleaned address, or None if it fails any
    check. `str.isprintable()` rejects newlines/control characters (the
    same header-injection-style concern email fields are classically
    vulnerable to) before the shape check even runs.
    """
    cleaned = raw.strip()
    if not cleaned or len(cleaned) > _EMAIL_MAX_LENGTH or not cleaned.isprintable():
        return None
    if not _EMAIL_RE.match(cleaned):
        return None
    return cleaned


def _build_confirmation_summary(state: BookingState) -> dict[str, object]:
    """{"slot", "timezone", "name", "email"} — Engineering Guide 4.8's
    documented `booking_confirmation_request` data shape exactly.
    """
    return {
        "slot": state.selected_slot_json,
        "timezone": state.timezone_name,
        "name": state.contact_name,
        "email": state.contact_email,
    }


async def _provide_contact_info(args: BaseModel, context: ToolContext) -> dict[str, object]:
    """Collects and code-validates name + email (Issue #21, 4.5's
    contact_info_collected step) — the model never decides an email is
    valid, only _validate_email does. On success, contact_info_collected
    and confirmed both fire within this one call (4.5: "Valid contact info
    -> contact_info_collected -> immediately assemble the confirmation
    summary -> confirmed state") — the second hop is purely mechanical
    once contact info is valid, so it doesn't need its own LLM round trip.
    """
    assert isinstance(args, ProvideContactInfoArgs)
    state = await load_booking_state(context.db, context.session_id)

    if state.step is not Step.SLOT_SELECTED:
        # Defense in depth (4.2), same rationale as calendar_find_slots'
        # own re-check: tool_defs_for_step already gates this per call, but
        # a multi-tool-call turn could see state move on between calls.
        return {"error": "not currently collecting contact info"}

    cleaned_name = _sanitize_short_text(args.name, _NAME_MAX_LENGTH)
    if not cleaned_name:
        return {
            "error": "name_invalid",
            "message": "Ask the visitor for their name again — it came through empty.",
        }

    valid_email = _validate_email(args.email)
    if valid_email is None:
        return {
            "error": "email_invalid",
            "message": (
                "That doesn't look like a valid email address. Ask the visitor to "
                "double-check it and share it again."
            ),
        }

    state = transition(
        state,
        Event(
            kind=EventKind.CONTACT_COLLECTED, contact_name=cleaned_name, contact_email=valid_email
        ),
    )
    state = transition(state, Event(kind=EventKind.CONFIRMED))
    await save_booking_state(context.db, state)

    # Feeds the pinned profile through its existing validated path (4.3:
    # "Booking contact fields ... feed the profile through their existing
    # validated paths"). Email has no dedicated `sessions` column (unlike
    # name/LinkedIn) per 4.7's actual schema, so it's rendered from
    # booking_states directly instead (app/agent/memory.py's
    # render_pinned_profile), mirroring how Issue #20 surfaced timezone.
    await set_visitor_info(context.db, context.session_id, name=cleaned_name)

    return {"confirmation_summary": _build_confirmation_summary(state)}


class CalendarCreateBookingArgs(BaseModel):
    """No fields — the issue text is explicit: "no free arguments —
    everything comes from persisted state" (the selected slot, timezone,
    and contact info already collected via the earlier steps).
    """


async def _calendar_create_booking(args: BaseModel, context: ToolContext) -> dict[str, object]:
    """The finale (Issue #22, 4.5's confirmed step): re-check free/busy for
    the exact slot right now (the race-condition guard — a lot can happen
    between proposing a slot and the visitor confirming it), then either
    create a tentative event and persist the booking, or bounce back to
    slots_proposed with the slot excluded if it was taken in the meantime.
    """
    assert isinstance(args, CalendarCreateBookingArgs)

    if not context.confirmation_is_affirmative:
        # Defense in depth (4.2): calendar_create_booking is gated to the
        # confirmed *step*, but that doesn't know what the visitor actually
        # said this turn — app/api/chat.py's deterministic classification
        # of the reply (app/booking/confirmation.py) is the real gate.
        return {
            "error": "not_confirmed",
            "message": (
                "The visitor hasn't clearly confirmed yet — ask again, or wait for a clear "
                "yes before calling this."
            ),
        }

    state = await load_booking_state(context.db, context.session_id)
    if state.step is not Step.CONFIRMED:
        return {"error": "booking is not currently awaiting confirmation"}

    selected = state.selected_slot_json
    if (
        selected is None
        or state.contact_name is None
        or state.contact_email is None
        or state.timezone_name is None
    ):
        # Shouldn't happen — reaching confirmed already requires all four
        # (slot_selected/contact_info_collected's own transition guards) —
        # but degrade gracefully rather than raising if it somehow does.
        logger.error(
            "calendar_create_booking: incomplete state for session %s: %r",
            context.session_id,
            state,
        )
        return {
            "error": "internal_error",
            "message": (
                "Something went wrong — ask the visitor to try again, or offer to have the "
                "owner follow up by email."
            ),
        }

    try:
        start = datetime.fromisoformat(str(selected["start_iso"]))
        end = datetime.fromisoformat(str(selected["end_iso"]))
    except (KeyError, ValueError):
        logger.error(
            "calendar_create_booking: malformed selected_slot_json for session %s: %r",
            context.session_id,
            selected,
        )
        return {
            "error": "internal_error",
            "message": "Something went wrong — ask the visitor to try again.",
        }

    busy = await context.calendar_client.get_free_busy(start, end)
    if any(b.start < end and b.end > start for b in busy):
        new_state = transition(state, Event(kind=EventKind.SLOT_TAKEN_AT_RECHECK))
        await save_booking_state(context.db, new_state)
        return {
            "taken": True,
            "message": (
                "That time was just taken by someone else — apologize briefly, then present "
                "any remaining times, or search again if none are left."
            ),
        }

    attendee = Attendee(name=state.contact_name, email=state.contact_email)
    description = (
        f"Call with {state.contact_name} ({state.contact_email}), booked via the site's "
        "AI assistant."
    )
    try:
        event_id = await context.calendar_client.create_event(start, end, attendee, description)
    except CalendarError:
        logger.error(
            "calendar_create_booking: event creation failed for session %s", context.session_id
        )
        return {
            "error": "creation_failed",
            "message": (
                "Something went wrong creating the event — ask the visitor to try again "
                "shortly, or offer to have the owner follow up by email."
            ),
        }

    # create_booking() and save_booking_state() each commit independently
    # (two separate transactions, not one) — `booking`/`booking_id` are
    # tracked outside the try so the except block below knows whether the
    # *first* commit already succeeded when only the *second* one fails,
    # and can clean up accordingly (review finding: treating both as a
    # single atomic unit missed exactly this case). `booking_id` is
    # captured as a plain int the moment it's available, separately from
    # the ORM object: a rollback anywhere in the except block expires
    # `booking`'s attributes, and logging `booking.id` after that would
    # silently trigger a fresh (and, in an outage, likely also-failing)
    # DB round trip just to format a log message.
    booking = None
    booking_id: int | None = None
    try:
        booking = await create_booking(
            context.db,
            session_id=context.session_id,
            slot_start_iso=str(selected["start_iso"]),
            slot_end_iso=str(selected["end_iso"]),
            timezone_name=state.timezone_name,
            contact_name=state.contact_name,
            contact_email=state.contact_email,
            gcal_event_id=event_id,
        )
        booking_id = booking.id
        new_state = transition(state, Event(kind=EventKind.CREATED))
        await save_booking_state(context.db, new_state)
    except Exception:
        # The calendar write above already succeeded — without this, a
        # failure here (a DB outage, a transient error) would leave a real
        # event on the calendar with no owner notification, permanently
        # invisible to the app: state stays at confirmed, so a retry
        # re-runs the free/busy re-check, sees the orphaned event as busy,
        # and wrongly bounces the visitor into "that time was just taken by
        # someone else" for a slot they already got. Best-effort
        # compensating cleanup closes that window; if any of it also fails,
        # it's logged for manual reconciliation rather than silently lost.
        logger.error(
            "calendar_create_booking: DB write failed after event creation for session %s "
            "(event_id=%s) — attempting cleanup",
            context.session_id,
            event_id,
        )
        # A failed commit can leave the session's transaction unusable for
        # further writes (including the compensating booking-row delete
        # below, and anything app/api/chat.py does with this same session
        # for the rest of the turn) until it's rolled back.
        await context.db.rollback()
        try:
            await context.calendar_client.delete_event(event_id)
        except Exception:
            logger.error(
                "calendar_create_booking: failed to delete orphaned event %s for session %s "
                "— needs manual cleanup",
                event_id,
                context.session_id,
            )
        if booking is not None:
            # create_booking()'s commit succeeded, so this exception must
            # have come from transition()/save_booking_state() — the
            # bookings row is now orphaned (event just deleted above, state
            # never advanced past confirmed) and, left in place, would let
            # a retry create a second event and a second bookings row for
            # the same session.
            try:
                await context.db.delete(booking)
                await context.db.commit()
            except Exception:
                # Same reasoning as the rollback above: leaving this
                # session's transaction unresolved would break every
                # subsequent write this request makes with it (e.g.
                # app/api/chat.py persisting this turn's messages).
                await context.db.rollback()
                logger.error(
                    "calendar_create_booking: failed to delete orphaned bookings row %s for "
                    "session %s — needs manual cleanup",
                    booking_id,
                    context.session_id,
                )
        return {
            "error": "creation_failed",
            "message": (
                "Something went wrong — ask the visitor to try again shortly, or offer to "
                "have the owner follow up by email."
            ),
        }

    # Scheduled as a background task by app/api/chat.py after run_agent
    # returns (see ToolContext.pending_owner_notifications) — a
    # notification failure must never affect this turn's reply, and the
    # booking above is already durably persisted regardless.
    context.pending_owner_notifications.append(
        (
            "New booking",
            f"{state.contact_name} ({state.contact_email}) booked a call: "
            f"{selected.get('label', '')}.",
        )
    )

    return {
        "booking_id": booking.id,
        "slot": selected,
        "timezone": state.timezone_name,
        "next_steps": f"To reschedule or cancel, email {settings.owner_contact_email} directly.",
    }


@dataclass
class _RegisteredTool:
    definition: ToolDef
    args_model: type[BaseModel]
    handler: Callable[[BaseModel, ToolContext], Awaitable[dict[str, object]]]
    # True for a tool whose availability is gated by the booking state
    # machine (app/booking/state.py's allowed_tools_for) rather than always
    # offered — used by tool_defs_for_step to decide which of the two lists
    # a tool belongs in (Issue #20).
    booking_gated: bool = False
    # False for tools whose outcome is either already surfaced elsewhere in
    # context (save_visitor_info -> the pinned profile) or is a pure
    # meta/side-effect with nothing visitor-facing worth a token in later
    # turns (flag_summary_conflict) — see app/agent/memory.py's ToolEvent.
    persist_receipt: bool = True


GET_CURRENT_DATE = _RegisteredTool(
    definition=ToolDef(
        name="get_current_date",
        description="Returns the current date and time in UTC.",
        parameters=GetCurrentDateArgs.model_json_schema(),
    ),
    args_model=GetCurrentDateArgs,
    handler=_get_current_date,
)

SAVE_VISITOR_INFO = _RegisteredTool(
    definition=ToolDef(
        name="save_visitor_info",
        description=(
            "Call this when the visitor volunteers their name, LinkedIn URL, or a short fact "
            "about themselves (e.g. their role or what they're looking for) — even in passing. "
            "All arguments are optional; pass only what was volunteered this turn. Do not ask "
            "the visitor for this information proactively, just capture it if they offer it."
        ),
        parameters=SaveVisitorInfoArgs.model_json_schema(),
    ),
    args_model=SaveVisitorInfoArgs,
    handler=_save_visitor_info,
    persist_receipt=False,
)

FLAG_SUMMARY_CONFLICT = _RegisteredTool(
    definition=ToolDef(
        name="flag_summary_conflict",
        description=(
            "Call this if the conversation summary or visitor profile you were given "
            "contradicts what the visitor is telling you now (e.g. they explicitly correct or "
            "walk back something stated earlier). Briefly explain the contradiction. This "
            "triggers a background correction — it does not change your current answer, so "
            "still answer the visitor's message normally based on what they just told you."
        ),
        parameters=FlagSummaryConflictArgs.model_json_schema(),
    ),
    args_model=FlagSummaryConflictArgs,
    handler=_flag_summary_conflict,
    persist_receipt=False,
)

CALENDAR_FIND_SLOTS = _RegisteredTool(
    definition=ToolDef(
        name="calendar_find_slots",
        description=(
            "Search the owner's calendar for available meeting times in a date range. Only "
            "call this once the visitor has clearly asked to schedule or book a call with the "
            "owner — never speculatively, and never in response to an off-topic or unrelated "
            "request, even one that mentions calendars or scheduling in passing. Calling this "
            "is itself how the system records that booking intent was detected, so restraint "
            "here matters. Needs the visitor's timezone first: if you don't have one, ask for "
            "it before calling this (a call made without a known timezone will just tell you "
            "to go get one instead of returning times). Returns up to 5 structured slots to "
            "present to the visitor."
        ),
        parameters=CoarseWindow.model_json_schema(),
    ),
    args_model=CoarseWindow,
    handler=_calendar_find_slots,
    booking_gated=True,
)

PROVIDE_CONTACT_INFO = _RegisteredTool(
    definition=ToolDef(
        name="provide_contact_info",
        description=(
            "Submit the visitor's name and email address once they've picked a time, so the "
            "booking can be locked in. Only call this after the visitor has actually given "
            "you both a name and an email — don't call it with partial or guessed "
            "information. If the email is rejected, ask the visitor to double-check it and "
            "call this again with the corrected value."
        ),
        parameters=ProvideContactInfoArgs.model_json_schema(),
    ),
    args_model=ProvideContactInfoArgs,
    handler=_provide_contact_info,
    booking_gated=True,
)

CALENDAR_CREATE_BOOKING = _RegisteredTool(
    definition=ToolDef(
        name="calendar_create_booking",
        description=(
            "Create the tentative calendar event for the confirmed booking. Takes no "
            "arguments — everything comes from the slot, timezone, and contact info already "
            "collected. Only call this after the visitor has clearly confirmed with a plain "
            "'yes', 'confirm', 'book it', or similar — never on a hesitant reply or a "
            "question, and never if they said no or asked for a different time."
        ),
        parameters=CalendarCreateBookingArgs.model_json_schema(),
    ),
    args_model=CalendarCreateBookingArgs,
    handler=_calendar_create_booking,
    booking_gated=True,
)

_REGISTRY: dict[str, _RegisteredTool] = {
    tool.definition.name: tool
    for tool in (
        GET_CURRENT_DATE,
        SAVE_VISITOR_INFO,
        FLAG_SUMMARY_CONFLICT,
        CALENDAR_FIND_SLOTS,
        PROVIDE_CONTACT_INFO,
        CALENDAR_CREATE_BOOKING,
    )
}

TOOL_DEFS: list[ToolDef] = [tool.definition for tool in _REGISTRY.values()]


def tool_defs_for_step(step: Step) -> list[ToolDef]:
    """The tools actually offered to the LLM this call (Issue #20, 4.2's
    "tool gating by state" safety property): every non-booking-gated tool,
    plus whichever booking-gated tools allowed_tools_for(step) currently
    permits. app/agent/loop.py recomputes this before every provider call,
    not just once per turn, so a state transition made by an earlier tool
    call in the same turn is reflected immediately for the next one.
    """
    allowed_booking_names = set(allowed_tools_for(step))
    return [
        tool.definition
        for tool in _REGISTRY.values()
        if not tool.booking_gated or tool.definition.name in allowed_booking_names
    ]


async def execute_tool(call: ToolCall, context: ToolContext) -> dict[str, object]:
    """Dispatch a tool call, validating arguments against its Pydantic
    schema. Invalid args (unknown tool, schema mismatch) *and* a failure
    inside the handler itself (a future tool making a real network call —
    RAG embeddings, calendar API — can fail for reasons unrelated to
    argument validation) both return a structured error dict *to the model
    as the tool result* — so it can self-correct — rather than raising;
    every failure is also logged server-side. Letting a handler exception
    escape here would turn one bad tool call into a 500 for the whole turn,
    rather than a self-correctable tool result (Engineering Guide 4.2: tool
    results are structured, in code, never inferred by letting the caller's
    exception handling improvise).
    """
    tool = _REGISTRY.get(call.name)
    if tool is None:
        logger.warning("unknown tool call: %s", call.name)
        return {"error": f"Unknown tool: {call.name}"}

    try:
        args = tool.args_model.model_validate(call.arguments)
    except ValidationError as exc:
        logger.warning("invalid arguments for tool %s: %s", call.name, exc)
        return {"error": f"Invalid arguments for {call.name}: {exc.errors()}"}
    except Exception as exc:
        # Pydantic v2 only wraps ValueError/TypeError/AssertionError raised
        # inside a @field_validator into ValidationError — any other
        # exception type a future tool's validator raises (e.g. a network
        # call inside a validator) would otherwise propagate raw and crash
        # the whole turn, exactly the failure class this function exists to
        # prevent. No current tool has a custom validator, but the registry
        # is generic, so this is guarded the same way handler failures are.
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.error(
            "tool %s argument validation raised unexpectedly\n%s", call.name, redact(tb_text)
        )
        return {"error": f"Invalid arguments for {call.name}."}

    try:
        return await tool.handler(args, context)
    except Exception as exc:
        # Sanitized the same way as app/api/errors.py's generic-exception
        # handler: a future tool making a real network call (calendar API,
        # embeddings) can raise an exception carrying credentials in a
        # header or connection string, so the traceback is redact()ed
        # before logging rather than passed through raw via
        # logger.exception(exc_info=exc).
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.error("tool %s raised during execution\n%s", call.name, redact(tb_text))
        return {"error": f"{call.name} failed to execute."}


def persist_receipt_for(tool_name: str) -> bool:
    """Whether a completed call to `tool_name` should be written to the
    messages log as a one-line receipt (Issue #11's window persistence) —
    used by app/agent/loop.py when building ToolEvents for persist_turn.
    Defaults True for an unknown name (fails toward keeping information,
    not silently dropping it).
    """
    tool = _REGISTRY.get(tool_name)
    return tool.persist_receipt if tool is not None else True
