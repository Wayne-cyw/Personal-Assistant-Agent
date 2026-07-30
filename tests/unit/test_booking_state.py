"""Exhaustive tests for app/booking/state.py (Issue #18 acceptance
criteria: 100% branch coverage on transition() and allowed_tools_for();
the 4.5 diagram and the code agree exactly).
"""

from datetime import UTC, datetime

import pytest

from app.booking.state import (
    BookingState,
    Event,
    EventKind,
    InvalidTransition,
    Step,
    allowed_tools_for,
    next_proposal_event,
    transition,
)

SID = "sess-1"


def _state(**overrides: object) -> BookingState:
    return BookingState(session_id=SID, **overrides)  # type: ignore[arg-type]


# --- every legal transition, one at a time ----------------------------------


def test_idle_to_intent_detected() -> None:
    result = transition(_state(), Event(kind=EventKind.INTENT_DETECTED))
    assert result.step is Step.INTENT_DETECTED


def test_timezone_captured_is_a_self_transition_that_records_timezone() -> None:
    state = _state(step=Step.INTENT_DETECTED)
    result = transition(state, Event(kind=EventKind.TIMEZONE_CAPTURED, timezone="America/New_York"))
    assert result.step is Step.INTENT_DETECTED  # unchanged
    assert result.timezone_name == "America/New_York"


def test_slots_proposed_from_intent_detected_sets_first_round() -> None:
    state = _state(step=Step.INTENT_DETECTED, timezone_name="America/New_York")
    slots = [{"slot_id": "s1"}, {"slot_id": "s2"}]
    result = transition(state, Event(kind=EventKind.SLOTS_PROPOSED, slots=slots))
    assert result.step is Step.SLOTS_PROPOSED
    assert result.proposed_slots_json == slots
    assert result.proposal_rounds == 1


def test_re_propose_increments_round() -> None:
    state = _state(step=Step.SLOTS_PROPOSED, proposal_rounds=1)
    result = transition(state, Event(kind=EventKind.RE_PROPOSE, slots=[{"slot_id": "s3"}]))
    assert result.step is Step.SLOTS_PROPOSED
    assert result.proposal_rounds == 2
    assert result.proposed_slots_json == [{"slot_id": "s3"}]


def test_re_propose_folds_the_outgoing_round_into_excluded_slots() -> None:
    state = _state(
        step=Step.SLOTS_PROPOSED, proposal_rounds=1, proposed_slots_json=[{"slot_id": "s1"}]
    )
    result = transition(state, Event(kind=EventKind.RE_PROPOSE, slots=[{"slot_id": "s3"}]))
    assert result.excluded_slots_json == [{"slot_id": "s1"}]


def test_re_propose_accumulates_onto_existing_excluded_slots() -> None:
    state = _state(
        step=Step.SLOTS_PROPOSED,
        proposal_rounds=1,
        proposed_slots_json=[{"slot_id": "s2"}],
        excluded_slots_json=[{"slot_id": "s1"}],
    )
    result = transition(state, Event(kind=EventKind.RE_PROPOSE, slots=[{"slot_id": "s3"}]))
    assert result.excluded_slots_json == [{"slot_id": "s1"}, {"slot_id": "s2"}]


def test_widen_window_at_cap_increments_round_again() -> None:
    state = _state(step=Step.SLOTS_PROPOSED, proposal_rounds=2)
    result = transition(state, Event(kind=EventKind.WIDEN_WINDOW, slots=[{"slot_id": "s4"}]))
    assert result.step is Step.SLOTS_PROPOSED
    assert result.proposal_rounds == 3


def test_widen_window_also_folds_the_outgoing_round_into_excluded_slots() -> None:
    state = _state(
        step=Step.SLOTS_PROPOSED,
        proposal_rounds=2,
        proposed_slots_json=[{"slot_id": "s3"}],
        excluded_slots_json=[{"slot_id": "s1"}],
    )
    result = transition(state, Event(kind=EventKind.WIDEN_WINDOW, slots=[{"slot_id": "s4"}]))
    assert result.excluded_slots_json == [{"slot_id": "s1"}, {"slot_id": "s3"}]


def test_email_fallback_after_widen_abandons() -> None:
    state = _state(step=Step.SLOTS_PROPOSED, proposal_rounds=3)
    result = transition(state, Event(kind=EventKind.EMAIL_FALLBACK))
    assert result.step is Step.ABANDONED


def test_slot_selected_from_slots_proposed() -> None:
    state = _state(step=Step.SLOTS_PROPOSED, proposal_rounds=1)
    slot = {"slot_id": "s1", "start_iso": "2026-08-03T09:00:00+00:00"}
    hold_expires_at = datetime(2026, 8, 3, 9, 10, tzinfo=UTC)
    result = transition(
        state, Event(kind=EventKind.SLOT_SELECTED, slot=slot, hold_expires_at=hold_expires_at)
    )
    assert result.step is Step.SLOT_SELECTED
    assert result.selected_slot_json == slot
    assert result.hold_expires_at == hold_expires_at


def test_contact_collected_from_slot_selected() -> None:
    state = _state(step=Step.SLOT_SELECTED)
    result = transition(
        state,
        Event(
            kind=EventKind.CONTACT_COLLECTED,
            contact_name="Priya",
            contact_email="priya@example.com",
        ),
    )
    assert result.step is Step.CONTACT_INFO_COLLECTED
    assert result.contact_name == "Priya"
    assert result.contact_email == "priya@example.com"


def test_confirmed_from_contact_info_collected() -> None:
    state = _state(step=Step.CONTACT_INFO_COLLECTED)
    result = transition(state, Event(kind=EventKind.CONFIRMED))
    assert result.step is Step.CONFIRMED


def test_created_from_confirmed_is_terminal() -> None:
    state = _state(step=Step.CONFIRMED)
    result = transition(state, Event(kind=EventKind.CREATED))
    assert result.step is Step.BOOKING_CREATED


def test_slot_taken_at_recheck_returns_to_slots_proposed_excluding_burned_slot() -> None:
    state = _state(
        step=Step.CONFIRMED,
        proposed_slots_json=[{"slot_id": "s1"}, {"slot_id": "s2"}, {"slot_id": "s3"}],
        selected_slot_json={"slot_id": "s2"},
        proposal_rounds=2,
    )
    result = transition(state, Event(kind=EventKind.SLOT_TAKEN_AT_RECHECK))
    assert result.step is Step.SLOTS_PROPOSED
    assert result.proposed_slots_json == [{"slot_id": "s1"}, {"slot_id": "s3"}]
    assert result.selected_slot_json is None
    assert result.proposal_rounds == 1  # fresh negotiation cycle, not the visitor's fault
    assert result.excluded_slots_json == [{"slot_id": "s2"}]


def test_confirmation_declined_returns_to_slots_proposed_excluding_declined_slot() -> None:
    state = _state(
        step=Step.CONFIRMED,
        proposed_slots_json=[{"slot_id": "s1"}, {"slot_id": "s2"}, {"slot_id": "s3"}],
        selected_slot_json={"slot_id": "s2"},
        proposal_rounds=2,
    )
    result = transition(state, Event(kind=EventKind.CONFIRMATION_DECLINED))
    assert result.step is Step.SLOTS_PROPOSED
    assert result.proposed_slots_json == [{"slot_id": "s1"}, {"slot_id": "s3"}]
    assert result.selected_slot_json is None
    assert result.proposal_rounds == 3  # unlike SLOT_TAKEN_AT_RECHECK, this counts as a round
    assert result.excluded_slots_json == [{"slot_id": "s2"}]


def test_confirmation_declined_accumulates_onto_existing_excluded_slots() -> None:
    """Regression test: a review pass found the declined slot dropped out
    of *both* proposed_slots_json and excluded_slots_json at once, so a
    later re-propose/widen call (which builds its exclude list from
    exactly those two fields) could legitimately re-offer a slot the
    visitor already explicitly declined at confirmation.
    """
    state = _state(
        step=Step.CONFIRMED,
        proposed_slots_json=[{"slot_id": "s2"}],
        selected_slot_json={"slot_id": "s2"},
        excluded_slots_json=[{"slot_id": "s1"}],
        proposal_rounds=1,
    )
    result = transition(state, Event(kind=EventKind.CONFIRMATION_DECLINED))
    assert result.excluded_slots_json == [{"slot_id": "s1"}, {"slot_id": "s2"}]
    assert result.proposed_slots_json == []


def test_confirmation_declined_outside_confirmed_is_illegal() -> None:
    with pytest.raises(InvalidTransition):
        transition(_state(step=Step.SLOT_SELECTED), Event(kind=EventKind.CONFIRMATION_DECLINED))


@pytest.mark.parametrize(
    "step",
    [
        Step.INTENT_DETECTED,
        Step.SLOTS_PROPOSED,
        Step.SLOT_SELECTED,
        Step.CONTACT_INFO_COLLECTED,
        Step.CONFIRMED,
    ],
)
def test_abandon_from_any_in_progress_step(step: Step) -> None:
    result = transition(_state(step=step), Event(kind=EventKind.ABANDON))
    assert result.step is Step.ABANDONED


# --- illegal transitions -----------------------------------------------------


def test_cannot_reach_confirmed_from_slots_proposed_directly() -> None:
    """The exact example named in the issue's acceptance criteria."""
    with pytest.raises(InvalidTransition):
        transition(_state(step=Step.SLOTS_PROPOSED), Event(kind=EventKind.CONFIRMED))


def test_cannot_detect_intent_twice() -> None:
    with pytest.raises(InvalidTransition):
        transition(_state(step=Step.INTENT_DETECTED), Event(kind=EventKind.INTENT_DETECTED))


def test_cannot_capture_timezone_outside_intent_detected() -> None:
    with pytest.raises(InvalidTransition):
        transition(_state(step=Step.IDLE), Event(kind=EventKind.TIMEZONE_CAPTURED, timezone="UTC"))


def test_cannot_propose_slots_without_timezone_known() -> None:
    state = _state(step=Step.INTENT_DETECTED, timezone_name=None)
    with pytest.raises(InvalidTransition):
        transition(state, Event(kind=EventKind.SLOTS_PROPOSED, slots=[]))


def test_cannot_propose_slots_from_idle() -> None:
    with pytest.raises(InvalidTransition):
        transition(_state(), Event(kind=EventKind.SLOTS_PROPOSED, slots=[]))


def test_re_propose_outside_slots_proposed_is_illegal() -> None:
    with pytest.raises(InvalidTransition):
        transition(_state(step=Step.INTENT_DETECTED), Event(kind=EventKind.RE_PROPOSE, slots=[]))


def test_widen_window_outside_slots_proposed_is_illegal() -> None:
    with pytest.raises(InvalidTransition):
        transition(_state(step=Step.SLOT_SELECTED), Event(kind=EventKind.WIDEN_WINDOW, slots=[]))


def test_re_propose_beyond_cap_is_illegal() -> None:
    state = _state(step=Step.SLOTS_PROPOSED, proposal_rounds=2)
    with pytest.raises(InvalidTransition):
        transition(state, Event(kind=EventKind.RE_PROPOSE, slots=[]))


def test_widen_window_before_cap_reached_is_illegal() -> None:
    state = _state(step=Step.SLOTS_PROPOSED, proposal_rounds=1)
    with pytest.raises(InvalidTransition):
        transition(state, Event(kind=EventKind.WIDEN_WINDOW, slots=[]))


def test_widen_window_twice_is_illegal() -> None:
    state = _state(step=Step.SLOTS_PROPOSED, proposal_rounds=3)
    with pytest.raises(InvalidTransition):
        transition(state, Event(kind=EventKind.WIDEN_WINDOW, slots=[]))


def test_email_fallback_before_widen_is_illegal() -> None:
    state = _state(step=Step.SLOTS_PROPOSED, proposal_rounds=2)
    with pytest.raises(InvalidTransition):
        transition(state, Event(kind=EventKind.EMAIL_FALLBACK))


def test_email_fallback_outside_slots_proposed_is_illegal() -> None:
    with pytest.raises(InvalidTransition):
        transition(_state(step=Step.SLOT_SELECTED), Event(kind=EventKind.EMAIL_FALLBACK))


def test_slot_selected_outside_slots_proposed_is_illegal() -> None:
    with pytest.raises(InvalidTransition):
        transition(_state(step=Step.INTENT_DETECTED), Event(kind=EventKind.SLOT_SELECTED, slot={}))


def test_contact_collected_outside_slot_selected_is_illegal() -> None:
    with pytest.raises(InvalidTransition):
        transition(
            _state(step=Step.SLOTS_PROPOSED),
            Event(kind=EventKind.CONTACT_COLLECTED, contact_name="Priya", contact_email="p@x.com"),
        )


def test_confirmed_outside_contact_info_collected_is_illegal() -> None:
    with pytest.raises(InvalidTransition):
        transition(_state(step=Step.SLOT_SELECTED), Event(kind=EventKind.CONFIRMED))


def test_created_outside_confirmed_is_illegal() -> None:
    with pytest.raises(InvalidTransition):
        transition(_state(step=Step.CONTACT_INFO_COLLECTED), Event(kind=EventKind.CREATED))


def test_slot_taken_at_recheck_outside_confirmed_is_illegal() -> None:
    with pytest.raises(InvalidTransition):
        transition(_state(step=Step.SLOT_SELECTED), Event(kind=EventKind.SLOT_TAKEN_AT_RECHECK))


@pytest.mark.parametrize("step", [Step.IDLE, Step.BOOKING_CREATED, Step.ABANDONED])
def test_cannot_abandon_idle_or_a_terminal_step(step: Step) -> None:
    """idle has no in-progress booking to abandon (nothing has started
    yet); booking_created/abandoned are terminal and have nothing left to
    abandon either.
    """
    with pytest.raises(InvalidTransition):
        transition(_state(step=step), Event(kind=EventKind.ABANDON))


# --- required-payload validation --------------------------------------------


def test_timezone_captured_without_timezone_raises_value_error() -> None:
    with pytest.raises(ValueError, match="timezone"):
        transition(_state(step=Step.INTENT_DETECTED), Event(kind=EventKind.TIMEZONE_CAPTURED))


def test_slot_selected_without_slot_raises_value_error() -> None:
    hold_expires_at = datetime(2026, 8, 3, 9, 10, tzinfo=UTC)
    with pytest.raises(ValueError, match="slot"):
        transition(
            _state(step=Step.SLOTS_PROPOSED),
            Event(kind=EventKind.SLOT_SELECTED, hold_expires_at=hold_expires_at),
        )


def test_slot_selected_without_hold_expires_at_raises_value_error() -> None:
    slot = {"slot_id": "s1"}
    with pytest.raises(ValueError, match="hold_expires_at"):
        transition(
            _state(step=Step.SLOTS_PROPOSED), Event(kind=EventKind.SLOT_SELECTED, slot=slot)
        )


def test_contact_collected_without_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="contact"):
        transition(
            _state(step=Step.SLOT_SELECTED),
            Event(kind=EventKind.CONTACT_COLLECTED, contact_email="p@x.com"),
        )


def test_contact_collected_without_email_raises_value_error() -> None:
    with pytest.raises(ValueError, match="contact"):
        transition(
            _state(step=Step.SLOT_SELECTED),
            Event(kind=EventKind.CONTACT_COLLECTED, contact_name="Priya"),
        )


# --- negotiation-cap arithmetic, end to end ---------------------------------


def test_negotiation_cap_arithmetic_two_rounds_then_widen_then_fallback() -> None:
    """The exact sequence named in the issue's testing task: 2 rounds ->
    widen once -> fallback.
    """
    state = _state(step=Step.INTENT_DETECTED, timezone_name="UTC")

    state = transition(state, Event(kind=EventKind.SLOTS_PROPOSED, slots=[{"slot_id": "a"}]))
    assert state.proposal_rounds == 1  # round 1: initial proposal

    state = transition(state, Event(kind=EventKind.RE_PROPOSE, slots=[{"slot_id": "b"}]))
    assert state.proposal_rounds == 2  # round 2: one re-propose

    with pytest.raises(InvalidTransition):
        transition(state, Event(kind=EventKind.RE_PROPOSE, slots=[]))  # cap reached

    state = transition(state, Event(kind=EventKind.WIDEN_WINDOW, slots=[{"slot_id": "c"}]))
    assert state.proposal_rounds == 3  # the one widen

    with pytest.raises(InvalidTransition):
        transition(state, Event(kind=EventKind.WIDEN_WINDOW, slots=[]))  # only widens once

    state = transition(state, Event(kind=EventKind.EMAIL_FALLBACK))
    assert state.step is Step.ABANDONED


# --- tool-gating table -------------------------------------------------------


def test_allowed_tools_for_idle() -> None:
    # Issue #20: no classifier exists yet, and state transitions only ever
    # happen inside execute_tool/handler code — so calendar_find_slots
    # itself has to be reachable from idle to be the mechanism that both
    # signals and records "the visitor wants to book."
    assert allowed_tools_for(Step.IDLE) == ["calendar_find_slots"]


def test_allowed_tools_for_intent_detected() -> None:
    assert allowed_tools_for(Step.INTENT_DETECTED) == ["calendar_find_slots"]


def test_allowed_tools_for_slots_proposed() -> None:
    assert allowed_tools_for(Step.SLOTS_PROPOSED) == ["calendar_find_slots"]


def test_allowed_tools_for_slot_selected() -> None:
    assert allowed_tools_for(Step.SLOT_SELECTED) == ["provide_contact_info"]


def test_allowed_tools_for_confirmed() -> None:
    assert allowed_tools_for(Step.CONFIRMED) == ["calendar_create_booking"]


@pytest.mark.parametrize(
    "step",
    [
        Step.CONTACT_INFO_COLLECTED,
        Step.BOOKING_CREATED,
        Step.ABANDONED,
    ],
)
def test_allowed_tools_empty_everywhere_else(step: Step) -> None:
    assert allowed_tools_for(step) == []


# --- next_proposal_event (Issue #20) -----------------------------------------


def test_next_proposal_event_from_idle_is_slots_proposed() -> None:
    assert next_proposal_event(_state()) is EventKind.SLOTS_PROPOSED


def test_next_proposal_event_from_intent_detected_is_slots_proposed() -> None:
    state = _state(step=Step.INTENT_DETECTED)
    assert next_proposal_event(state) is EventKind.SLOTS_PROPOSED


def test_next_proposal_event_first_round_is_re_propose() -> None:
    state = _state(step=Step.SLOTS_PROPOSED, proposal_rounds=1)
    assert next_proposal_event(state) is EventKind.RE_PROPOSE


def test_next_proposal_event_at_cap_is_widen_window() -> None:
    state = _state(step=Step.SLOTS_PROPOSED, proposal_rounds=2)
    assert next_proposal_event(state) is EventKind.WIDEN_WINDOW


def test_next_proposal_event_past_cap_is_email_fallback() -> None:
    state = _state(step=Step.SLOTS_PROPOSED, proposal_rounds=3)
    assert next_proposal_event(state) is EventKind.EMAIL_FALLBACK


@pytest.mark.parametrize(
    "step",
    [
        Step.SLOT_SELECTED,
        Step.CONTACT_INFO_COLLECTED,
        Step.CONFIRMED,
        Step.BOOKING_CREATED,
        Step.ABANDONED,
    ],
)
def test_next_proposal_event_raises_outside_the_reachable_steps(step: Step) -> None:
    with pytest.raises(AssertionError):
        next_proposal_event(_state(step=step))


# --- other -------------------------------------------------------------------


def test_invalid_transition_message_names_step_and_event() -> None:
    try:
        transition(_state(step=Step.SLOTS_PROPOSED), Event(kind=EventKind.CONFIRMED))
    except InvalidTransition as exc:
        assert exc.step is Step.SLOTS_PROPOSED
        assert exc.event is EventKind.CONFIRMED
        assert "slots_proposed" in str(exc)
        assert "confirmed" in str(exc)
    else:
        pytest.fail("expected InvalidTransition")


def test_transition_does_not_mutate_the_input_state() -> None:
    original = _state(step=Step.INTENT_DETECTED)
    transition(original, Event(kind=EventKind.TIMEZONE_CAPTURED, timezone="UTC"))
    assert original.step is Step.INTENT_DETECTED
    assert original.timezone_name is None
