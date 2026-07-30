"""Booking state machine: steps, transitions, and tool gating (Engineering
Guide 4.5) — pure, DB-free, LLM-free Python, deliberately built this way
first so its correctness is provable by exhaustive unit tests alone.

Persistence (converting to/from the `booking_states` DB row) lives in
app/db/session.py, per this codebase's rule that only that module talks to
the DB directly — this module never imports anything DB- or async-related.

State transitions happen only inside execute_tool/handler code (4.5) —
never inferred from the LLM's prose. This module is the single place that
logic lives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum

# "≤ 2 proposal rounds, then widen the window once, then offer email
# fallback" (4.5's slot_selected row) — proposal_rounds counts every
# proposal-shaped event (the initial proposal, each re-propose, and the
# one widen) linearly: 1 (initial) -> 2 (one re-propose) -> 3 (one widen).
_PROPOSAL_ROUND_CAP = 2


class Step(StrEnum):
    """Every box in 4.5's diagram, including the `abandoned` escape hatch
    reachable from any non-terminal step.
    """

    IDLE = "idle"
    INTENT_DETECTED = "intent_detected"
    SLOTS_PROPOSED = "slots_proposed"
    SLOT_SELECTED = "slot_selected"
    CONTACT_INFO_COLLECTED = "contact_info_collected"
    CONFIRMED = "confirmed"
    BOOKING_CREATED = "booking_created"  # terminal
    ABANDONED = "abandoned"  # terminal


class EventKind(StrEnum):
    """The transition() task list, in order: intent detected; tz captured;
    slots proposed; slot selected; re-propose; widen-window; email-
    fallback; contact collected; confirmed; created; slot-taken-at-
    recheck; abandonment.
    """

    INTENT_DETECTED = "intent_detected"
    TIMEZONE_CAPTURED = "timezone_captured"
    SLOTS_PROPOSED = "slots_proposed"
    SLOT_SELECTED = "slot_selected"
    RE_PROPOSE = "re_propose"
    WIDEN_WINDOW = "widen_window"
    EMAIL_FALLBACK = "email_fallback"
    CONTACT_COLLECTED = "contact_collected"
    CONFIRMED = "confirmed"
    CREATED = "created"
    SLOT_TAKEN_AT_RECHECK = "slot_taken_at_recheck"
    ABANDON = "abandon"


@dataclass
class Event:
    """A single FSM input. Only the fields relevant to `kind` need to be
    set — transition() validates that a required payload field is present
    for the events that need one.
    """

    kind: EventKind
    timezone: str | None = None
    slots: list[dict[str, object]] | None = None
    slot: dict[str, object] | None = None
    contact_name: str | None = None
    contact_email: str | None = None


@dataclass
class BookingState:
    """Mirrors the `booking_states` DB row (app/db/models.py) field for
    field — app/db/session.py's load_booking_state/save_booking_state
    convert between the two.
    """

    session_id: str
    step: Step = Step.IDLE
    timezone_name: str | None = None
    proposed_slots_json: list[dict[str, object]] | None = None
    selected_slot_json: dict[str, object] | None = None
    hold_expires_at: datetime | None = None
    proposal_rounds: int = 0
    contact_name: str | None = None
    contact_email: str | None = None


class InvalidTransition(Exception):
    """Raised when `event` cannot legally fire while in `step` — e.g.
    cannot reach `confirmed` from `slots_proposed` directly.
    """

    def __init__(self, step: Step, event: EventKind) -> None:
        self.step = step
        self.event = event
        super().__init__(f"cannot handle event {event.value!r} while in step {step.value!r}")


def transition(state: BookingState, event: Event) -> BookingState:
    """Apply `event` to `state`, returning a new BookingState — never
    mutates the input. Raises InvalidTransition for any (step, event)
    combination not explicitly legal below, and ValueError if an event
    fires without a payload field it requires.
    """
    step = state.step
    kind = event.kind

    if kind is EventKind.ABANDON:
        # 4.5: "(any step can -> abandoned)" — except the two terminals,
        # which have nothing left to abandon.
        if step in (Step.BOOKING_CREATED, Step.ABANDONED):
            raise InvalidTransition(step, kind)
        return replace(state, step=Step.ABANDONED)

    if kind is EventKind.INTENT_DETECTED:
        if step is not Step.IDLE:
            raise InvalidTransition(step, kind)
        return replace(state, step=Step.INTENT_DETECTED)

    if kind is EventKind.TIMEZONE_CAPTURED:
        # Self-transition: 4.5's intent_detected row "ensures timezone is
        # known" before slots can be proposed — this only records the
        # timezone, it doesn't itself advance the step.
        if step is not Step.INTENT_DETECTED:
            raise InvalidTransition(step, kind)
        if not event.timezone:
            raise ValueError("timezone_captured event requires a timezone")
        return replace(state, timezone_name=event.timezone)

    if kind is EventKind.SLOTS_PROPOSED:
        if step is not Step.INTENT_DETECTED:
            raise InvalidTransition(step, kind)
        if state.timezone_name is None:
            raise InvalidTransition(step, kind)  # timezone must be known first
        return replace(
            state,
            step=Step.SLOTS_PROPOSED,
            proposed_slots_json=list(event.slots or []),
            proposal_rounds=1,
        )

    if kind is EventKind.RE_PROPOSE:
        if step is not Step.SLOTS_PROPOSED:
            raise InvalidTransition(step, kind)
        if state.proposal_rounds >= _PROPOSAL_ROUND_CAP:
            raise InvalidTransition(step, kind)  # cap reached; must widen instead
        return replace(
            state,
            proposed_slots_json=list(event.slots or []),
            proposal_rounds=state.proposal_rounds + 1,
        )

    if kind is EventKind.WIDEN_WINDOW:
        if step is not Step.SLOTS_PROPOSED:
            raise InvalidTransition(step, kind)
        if state.proposal_rounds != _PROPOSAL_ROUND_CAP:
            raise InvalidTransition(step, kind)  # only legal right at the cap, and only once
        return replace(
            state,
            proposed_slots_json=list(event.slots or []),
            proposal_rounds=state.proposal_rounds + 1,
        )

    if kind is EventKind.EMAIL_FALLBACK:
        if step is not Step.SLOTS_PROPOSED:
            raise InvalidTransition(step, kind)
        if state.proposal_rounds <= _PROPOSAL_ROUND_CAP:
            raise InvalidTransition(step, kind)  # widen must happen first
        return replace(state, step=Step.ABANDONED)

    if kind is EventKind.SLOT_SELECTED:
        if step is not Step.SLOTS_PROPOSED:
            raise InvalidTransition(step, kind)
        if event.slot is None:
            raise ValueError("slot_selected event requires a slot")
        return replace(state, step=Step.SLOT_SELECTED, selected_slot_json=event.slot)

    if kind is EventKind.CONTACT_COLLECTED:
        if step is not Step.SLOT_SELECTED:
            raise InvalidTransition(step, kind)
        if not event.contact_name or not event.contact_email:
            raise ValueError("contact_collected event requires contact_name and contact_email")
        return replace(
            state,
            step=Step.CONTACT_INFO_COLLECTED,
            contact_name=event.contact_name,
            contact_email=event.contact_email,
        )

    if kind is EventKind.CONFIRMED:
        if step is not Step.CONTACT_INFO_COLLECTED:
            raise InvalidTransition(step, kind)
        return replace(state, step=Step.CONFIRMED)

    if kind is EventKind.CREATED:
        if step is not Step.CONFIRMED:
            raise InvalidTransition(step, kind)
        return replace(state, step=Step.BOOKING_CREATED)

    if kind is EventKind.SLOT_TAKEN_AT_RECHECK:
        if step is not Step.CONFIRMED:
            raise InvalidTransition(step, kind)
        burned_id = (state.selected_slot_json or {}).get("slot_id")
        remaining = [s for s in (state.proposed_slots_json or []) if s.get("slot_id") != burned_id]
        # Fresh negotiation cycle: the race-condition failure isn't the
        # visitor's fault, so it doesn't consume any of their 2-round cap.
        return replace(
            state,
            step=Step.SLOTS_PROPOSED,
            proposed_slots_json=remaining,
            selected_slot_json=None,
            proposal_rounds=1,
        )

    # Exhaustiveness guard, unreachable while every EventKind is handled above.
    raise AssertionError(f"unhandled event kind: {kind!r}")  # pragma: no cover


# calendar_find_slots only from intent/proposal steps; calendar_create_
# booking only from confirmed (4.5's tool-gating rule, enforced here so a
# jailbreak convincing the model to "just book it" fails structurally —
# the tool isn't even in the list it was given, 4.2).
_ALLOWED_TOOLS: dict[Step, tuple[str, ...]] = {
    Step.INTENT_DETECTED: ("calendar_find_slots",),
    Step.SLOTS_PROPOSED: ("calendar_find_slots",),
    Step.CONFIRMED: ("calendar_create_booking",),
}


def allowed_tools_for(step: Step) -> list[str]:
    return list(_ALLOWED_TOOLS.get(step, ()))
