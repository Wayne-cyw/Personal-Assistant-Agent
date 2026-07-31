"""calendar_find_slots tool handler (Issue #20) — the state-machine wiring
half of #20's scope (deterministic-selection matching is tested separately
in tests/unit/test_selection.py; the pure FSM/tool-gating logic it drives is
tested in tests/unit/test_booking_state.py).
"""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import app.tools.registry as registry_module
from app.agent.providers.base import ToolCall
from app.agent.providers.fake import FakeProvider
from app.booking.slots import AvailabilityPolicy, AvailabilityPolicyError
from app.booking.state import BookingState, Step, allowed_tools_for
from app.config import settings
from app.db.models import Base, SessionRow
from app.db.session import get_or_create_session, load_booking_state, save_booking_state
from app.tools.calendar import CalendarError
from app.tools.context import ToolContext
from app.tools.fake_calendar import FakeCalendar
from app.tools.registry import execute_tool, tool_defs_for_step

TZ = ZoneInfo("America/Toronto")
SID = "sess-1"


def _policy(**overrides: object) -> AvailabilityPolicy:
    defaults: dict[str, object] = dict(
        timezone=TZ,
        timezone_name="America/Toronto",
        work_start_day=0,
        work_end_day=4,
        work_start_time=datetime(2000, 1, 1, 9, 0).time(),
        work_end_time=datetime(2000, 1, 1, 17, 0).time(),
        meeting_length=timedelta(minutes=30),
        buffer=timedelta(0),
        min_notice=timedelta(hours=1),
    )
    defaults.update(overrides)
    return AvailabilityPolicy(**defaults)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _stub_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """knowledge/availability_policy.md is still a template (Issue #13) in
    this repo, so the real get_availability_policy() always raises here —
    tests that need a working policy stub it explicitly via
    _configure_policy; this autouse default keeps every other test's
    behavior independent of that unrelated, still-blocked issue.
    """
    monkeypatch.setattr(
        registry_module,
        "get_availability_policy",
        lambda: (_ for _ in ()).throw(AvailabilityPolicyError("not configured")),
    )


def _configure_policy(monkeypatch: pytest.MonkeyPatch, policy: AvailabilityPolicy) -> None:
    monkeypatch.setattr(registry_module, "get_availability_policy", lambda: policy)


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine]:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db(engine: AsyncEngine) -> AsyncGenerator[AsyncSession]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        await get_or_create_session(session, SID)
        yield session


def _new_context(
    db: AsyncSession,
    *,
    caller_timezone: str | None = "America/Toronto",
    caller_ip: str | None = None,
) -> ToolContext:
    """One ToolContext per simulated turn — app/api/chat.py constructs a
    fresh ToolContext per HTTP request, and calendar_find_slots_used_this_
    turn (Issue #20 review fix) is scoped to a single ToolContext instance,
    so a test simulating several separate visitor turns must build a new
    one for each, exactly like the real handler does. Reusing one context
    across simulated turns would trip the once-per-turn guard as if the
    model had called calendar_find_slots twice in a single turn.
    """
    return ToolContext(
        db=db,
        session_id=SID,
        summarizer_provider=FakeProvider(responses=[]),
        calendar_client=FakeCalendar(),
        caller_timezone=caller_timezone,
        caller_ip=caller_ip,
    )


@pytest.fixture
def context(db: AsyncSession) -> ToolContext:
    return _new_context(db)


def _call(**args: object) -> ToolCall:
    return ToolCall(id="call_1", name="calendar_find_slots", arguments=args)


# --- tool gating --------------------------------------------------------------


def test_calendar_find_slots_is_offered_from_idle() -> None:
    assert "calendar_find_slots" in [t.name for t in tool_defs_for_step(Step.IDLE)]


def test_calendar_find_slots_offered_from_intent_detected_and_slots_proposed() -> None:
    for step in (Step.INTENT_DETECTED, Step.SLOTS_PROPOSED):
        assert "calendar_find_slots" in [t.name for t in tool_defs_for_step(step)]


def test_calendar_find_slots_not_offered_outside_reachable_steps() -> None:
    for step in (
        Step.SLOT_SELECTED,
        Step.CONTACT_INFO_COLLECTED,
        Step.CONFIRMED,
        Step.BOOKING_CREATED,
        Step.ABANDONED,
    ):
        assert "calendar_find_slots" not in [t.name for t in tool_defs_for_step(step)]


def test_general_tools_always_offered_regardless_of_booking_step() -> None:
    for step in Step:
        names = [t.name for t in tool_defs_for_step(step)]
        assert "get_current_date" in names
        assert "save_visitor_info" in names


def test_tool_defs_for_step_matches_allowed_tools_for_booking_gated_names() -> None:
    for step in Step:
        gated_names = {t.name for t in tool_defs_for_step(step) if t.name == "calendar_find_slots"}
        assert gated_names == set(allowed_tools_for(step)) & {"calendar_find_slots"}


# --- happy path: idle -> slots_proposed --------------------------------------


async def test_first_call_from_idle_detects_intent_and_proposes_slots(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_policy(monkeypatch, _policy())

    result = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), context
    )

    assert "error" not in result
    assert isinstance(result["slots"], list)
    assert len(result["slots"]) > 0
    assert result["round"] == 1

    state = await load_booking_state(context.db, SID)
    assert state.step is Step.SLOTS_PROPOSED
    assert state.timezone_name == "America/Toronto"
    assert state.proposal_rounds == 1
    assert state.proposed_slots_json == result["slots"]


async def test_call_from_idle_without_timezone_asks_for_one_but_still_records_intent(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_policy(monkeypatch, _policy())
    context.caller_timezone = None

    result = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), context
    )

    assert result["error"] == "timezone_required"

    state = await load_booking_state(context.db, SID)
    assert state.step is Step.INTENT_DETECTED
    assert state.timezone_name is None


async def test_second_call_with_timezone_now_known_proposes_slots(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The visitor supplies a timezone on a later turn (a real client would
    resend ChatRequest.timezone) — intent, already recorded, isn't re-asked.
    """
    _configure_policy(monkeypatch, _policy())
    await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"),
        _new_context(db, caller_timezone=None),
    )

    result = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )

    assert "slots" in result
    state = await load_booking_state(db, SID)
    assert state.step is Step.SLOTS_PROPOSED


async def test_availability_not_configured_returns_structured_error_and_saves_progress(
    context: ToolContext,
) -> None:
    """_stub_policy (autouse) makes this the default — no monkeypatch
    override needed.
    """
    result = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), context
    )

    assert "error" in result
    state = await load_booking_state(context.db, SID)
    # Intent detection (idle -> intent_detected) still happened and was
    # saved, even though the policy lookup failed afterward.
    assert state.step is Step.INTENT_DETECTED


# --- re-propose / widen / email fallback (negotiation cap) -------------------


async def test_re_propose_from_slots_proposed_increments_round(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_policy(monkeypatch, _policy())
    first = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )
    assert first["round"] == 1

    second = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )
    assert second["round"] == 2

    state = await load_booking_state(db, SID)
    assert state.step is Step.SLOTS_PROPOSED
    assert state.proposal_rounds == 2


async def test_re_propose_excludes_previously_offered_slots(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A short window with exactly one bookable slot, so the second call
    # (which excludes it) has nothing left to offer.
    _configure_policy(
        monkeypatch,
        _policy(
            work_start_time=datetime(2000, 1, 1, 9, 0).time(),
            work_end_time=datetime(2000, 1, 1, 9, 30).time(),
        ),
    )
    first = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )
    assert len(first["slots"]) == 1  # type: ignore[arg-type]

    second = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )
    assert second["slots"] == []


async def test_widen_still_excludes_round_one_slots_not_just_round_two(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test (Issue #20 review finding): RE_PROPOSE/WIDEN_WINDOW
    both *replace* proposed_slots_json each round, so an exclude list built
    only from the immediately-preceding round would forget round 1's
    already-rejected slot by the time round 3 (widen) runs. A single
    30-minute window has exactly one bookable slot; if round 3 doesn't
    still exclude it (from round 1) alongside round 2's, it would come back.
    """
    _configure_policy(
        monkeypatch,
        _policy(
            work_start_time=datetime(2000, 1, 1, 9, 0).time(),
            work_end_time=datetime(2000, 1, 1, 9, 30).time(),
        ),
    )
    first = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )
    first_slots = cast("list[object]", first["slots"])
    assert len(first_slots) == 1
    first_slot = first_slots[0]

    second = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )
    assert second["slots"] == []  # round 1's only slot is excluded

    third = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )
    # Widening extends the end date, but the policy still only opens a
    # 30-min window each workday — round 1's slot must still be excluded,
    # not just round 2's (which had nothing to offer in the first place).
    third_slots = cast("list[object]", third["slots"])
    assert first_slot not in third_slots

    state = await load_booking_state(db, SID)
    assert state.excluded_slots_json is not None
    assert first_slot in state.excluded_slots_json


async def test_third_call_widens_the_window_and_increments_round_again(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_policy(monkeypatch, _policy())
    await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )  # round 1
    await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )  # round 2

    third = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )

    assert third["round"] == 3
    state = await load_booking_state(db, SID)
    assert state.step is Step.SLOTS_PROPOSED
    assert state.proposal_rounds == 3


async def test_fourth_call_falls_back_to_email_and_abandons(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_policy(monkeypatch, _policy())
    for _ in range(3):
        await execute_tool(
            _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
        )

    fourth = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )

    assert fourth["fallback"] is True
    assert "email" in str(fourth["message"]).lower()
    state = await load_booking_state(db, SID)
    assert state.step is Step.ABANDONED


# --- cross-session soft holds (Issue #21) ------------------------------------


async def test_another_sessions_active_hold_blocks_the_first_proposal(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held slot must be excluded on a session's very first
    calendar_find_slots call, not just on re-propose/widen — two visitors
    must never both be offered the same slot.
    """
    _configure_policy(
        monkeypatch,
        _policy(
            work_start_time=datetime(2000, 1, 1, 9, 0).time(),
            work_end_time=datetime(2000, 1, 1, 9, 30).time(),
        ),
    )
    await save_booking_state(
        db,
        BookingState(
            session_id="sess-other",
            step=Step.SLOT_SELECTED,
            selected_slot_json={
                "slot_id": "held",
                "start_iso": "2026-08-03T09:00:00-04:00",
                "end_iso": "2026-08-03T09:30:00-04:00",
            },
            hold_expires_at=datetime(2099, 1, 1, tzinfo=UTC),
        ),
    )

    result = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )

    assert result["slots"] == []


async def test_second_call_within_the_same_turn_is_refused_and_does_not_advance_round(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test (Issue #20 review finding): a single LLM response
    (or several run_agent iterations within one turn) could otherwise emit
    calendar_find_slots more than once, each independently advancing
    proposal_rounds — burning through the entire negotiation cap without
    the visitor ever having rejected a real proposal. Reusing `context`
    (one ToolContext, i.e. one simulated turn) across both calls is the
    point of this test.
    """
    _configure_policy(monkeypatch, _policy())
    first = await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)
    assert first["round"] == 1

    second = await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)

    assert "error" in second
    assert "already called" in str(second["error"])
    state = await load_booking_state(context.db, SID)
    assert state.proposal_rounds == 1  # unchanged — the second call never ran


# --- defense in depth: state gating checked inside the handler too -----------


async def test_handler_refuses_when_state_has_moved_past_reachable_steps(
    context: ToolContext,
) -> None:
    await save_booking_state(context.db, BookingState(session_id=SID, step=Step.CONFIRMED))

    result = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), context
    )

    assert "error" in result
    state = await load_booking_state(context.db, SID)
    assert state.step is Step.CONFIRMED  # unchanged


# --- booking-specific rate limits (Issue #23) ---------------------------------


async def test_a_call_that_only_asks_for_the_timezone_does_not_consume_an_attempt(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: a review pass found the rate-limit check originally
    ran before the timezone-required early return, so a visitor who simply
    hadn't sent their timezone yet -- an ordinary, spec-anticipated case,
    not abuse -- burned one of their limited attempts just asking for it.
    The check now runs after the timezone/policy guards, right before the
    call's real work, so it's the negotiation flow's own legitimate call
    count (initial proposal, re-propose, widen, one more call that lands
    on email fallback -- 4, exactly BOOKING_ATTEMPTS_PER_SESSION's
    default) that's actually being counted.
    """
    _configure_policy(monkeypatch, _policy())
    monkeypatch.setattr(settings, "booking_attempts_per_session", 4)

    ask = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"),
        _new_context(db, caller_timezone=None),
    )
    assert ask["error"] == "timezone_required"

    # All 4 of a full legitimate negotiation's real calls still succeed --
    # none of the cap was spent on the timezone-ask call above.
    for _ in range(3):
        result = await execute_tool(
            _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
        )
        assert "rate_limited" not in result
    fourth = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )
    assert fourth["fallback"] is True  # negotiation cap's own conclusion, not a rate limit


async def test_session_cap_allows_up_to_the_limit_then_refuses(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_policy(monkeypatch, _policy())
    monkeypatch.setattr(settings, "booking_attempts_per_session", 2)

    first = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )
    assert "rate_limited" not in first
    second = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )
    assert "rate_limited" not in second

    third = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"), _new_context(db)
    )

    assert third["rate_limited"] is True
    assert settings.owner_contact_email in str(third["message"])
    state = await load_booking_state(db, SID)
    assert state.step is Step.ABANDONED  # there was an in-progress flow to abandon
    row = await db.get(SessionRow, SID)
    assert row is not None
    assert row.flagged is True


async def test_session_blocked_only_by_a_shared_ip_is_not_flagged(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session whose *own* attempts never exceeded its own cap, but
    which gets refused because another session already exhausted their
    shared IP's daily quota, is an innocent bystander -- not evidence of
    this session being suspicious (review finding: flagging every
    IP-capped session would false-positive on shared office/NAT IPs once
    flagged sessions start driving owner notifications, Issues #16/#27).
    Only a session whose own per-session cap trips gets flagged (see
    test_session_cap_allows_up_to_the_limit_then_refuses above).

    The rate-limit check runs after intent_detected/timezone_captured
    already fired for this call (it only guards the call's real work, not
    the bookkeeping that happens on the way there -- see the handler's own
    comment), so by the time it's evaluated there's already a real,
    abandon-able in-progress flow, even on this session's very first call.
    """
    _configure_policy(monkeypatch, _policy())
    monkeypatch.setattr(settings, "booking_attempts_per_session", 1)
    monkeypatch.setattr(settings, "booking_attempts_per_ip_per_day", 1)

    # Burn the shared IP's daily cap from a different session first.
    await get_or_create_session(db, "sess-other")
    other_context = ToolContext(
        db=db,
        session_id="sess-other",
        summarizer_provider=FakeProvider(responses=[]),
        calendar_client=FakeCalendar(),
        caller_timezone="America/Toronto",
        caller_ip="1.2.3.4",
    )
    burned = await execute_tool(
        ToolCall(
            id="call_1",
            name="calendar_find_slots",
            arguments={"date_from": "2026-08-03", "date_to": "2026-08-03"},
        ),
        other_context,
    )
    assert "rate_limited" not in burned

    result = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"),
        _new_context(db, caller_ip="1.2.3.4"),
    )

    assert result["rate_limited"] is True
    state = await load_booking_state(db, SID)
    assert state.step is Step.ABANDONED  # a real flow had already started this call
    row = await db.get(SessionRow, SID)
    assert row is not None
    assert row.flagged is False  # this session's own cap was never exceeded


async def test_ip_cap_is_shared_across_sessions_from_the_same_ip(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_policy(monkeypatch, _policy())
    monkeypatch.setattr(settings, "booking_attempts_per_ip_per_day", 1)

    first = await execute_tool(
        _call(date_from="2026-08-03", date_to="2026-08-03"),
        _new_context(db, caller_ip="9.9.9.9"),
    )
    assert "rate_limited" not in first

    await get_or_create_session(db, "sess-2")
    second_context = ToolContext(
        db=db,
        session_id="sess-2",
        summarizer_provider=FakeProvider(responses=[]),
        calendar_client=FakeCalendar(),
        caller_timezone="America/Toronto",
        caller_ip="9.9.9.9",
    )
    second = await execute_tool(
        ToolCall(
            id="call_1",
            name="calendar_find_slots",
            arguments={"date_from": "2026-08-03", "date_to": "2026-08-03"},
        ),
        second_context,
    )

    assert second["rate_limited"] is True


async def test_no_caller_ip_skips_the_ip_cap(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """caller_ip is None whenever the request has no client info (Issue
    #23) -- the per-IP cap simply doesn't apply then, rather than blocking
    or erroring; only the per-session cap is checked.
    """
    _configure_policy(monkeypatch, _policy())
    monkeypatch.setattr(settings, "booking_attempts_per_ip_per_day", 1)
    monkeypatch.setattr(settings, "booking_attempts_per_session", 2)

    for _ in range(2):
        result = await execute_tool(
            _call(date_from="2026-08-03", date_to="2026-08-03"),
            _new_context(db, caller_ip=None),
        )
        assert "rate_limited" not in result


async def test_calendar_failure_after_the_rate_limit_check_still_persists_state(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: a review pass found the rate-limit check's
    increment (already committed by this point) is followed by a real
    calendar_client.get_free_busy() call inside generate_slots that can
    raise CalendarError -- without a try/except around it, that exception
    would propagate past every save_booking_state call in this function,
    silently discarding this call's own state.py progress (idle ->
    intent_detected, timezone_captured) even though the rate-limit
    increment itself was never undone. This doesn't refund the attempt
    (accepted trade-off, see the handler's own comment) but does verify
    the in-memory state progress this call made isn't lost on top of that.
    """
    _configure_policy(monkeypatch, _policy())
    failing_calendar = FakeCalendar(fail_with=CalendarError("boom"))
    context = ToolContext(
        db=db,
        session_id=SID,
        summarizer_provider=FakeProvider(responses=[]),
        calendar_client=failing_calendar,
        caller_timezone="America/Toronto",
    )

    result = await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)

    assert "error" in result
    assert "rate_limited" not in result
    state = await load_booking_state(db, SID)
    # The failure happened after timezone_captured already fired in-memory
    # this same call -- that progress must survive the calendar failure.
    assert state.step is Step.INTENT_DETECTED
    assert state.timezone_name == "America/Toronto"


async def test_a_malformed_timezone_also_still_persists_state(
    db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: a review pass on the fix above found it only
    caught CalendarError, but generate_slots also constructs a ZoneInfo
    from the caller-supplied timezone_name -- which is never
    format-validated before reaching here (ChatRequest.timezone has no
    validator, TIMEZONE_CAPTURED only checks truthiness) -- and raises
    ZoneInfoNotFoundError, not CalendarError, on a malformed value. That
    trigger is fully caller-controlled (no real Calendar outage needed)
    and would have kept silently discarding this call's own progress if
    the except clause weren't broadened.
    """
    _configure_policy(monkeypatch, _policy())
    context = ToolContext(
        db=db,
        session_id=SID,
        summarizer_provider=FakeProvider(responses=[]),
        calendar_client=FakeCalendar(),
        caller_timezone="Not/ARealZone",
    )

    result = await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)

    assert "error" in result
    assert "rate_limited" not in result
    state = await load_booking_state(db, SID)
    assert state.step is Step.INTENT_DETECTED
    assert state.timezone_name == "Not/ARealZone"
