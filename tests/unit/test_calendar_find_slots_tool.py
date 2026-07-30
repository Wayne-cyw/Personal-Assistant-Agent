"""calendar_find_slots tool handler (Issue #20) — the state-machine wiring
half of #20's scope (deterministic-selection matching is tested separately
in tests/unit/test_selection.py; the pure FSM/tool-gating logic it drives is
tested in tests/unit/test_booking_state.py).
"""

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from pathlib import Path
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
from app.db.models import Base
from app.db.session import get_or_create_session, load_booking_state, save_booking_state
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


@pytest.fixture
def context(db: AsyncSession) -> ToolContext:
    return ToolContext(
        db=db,
        session_id=SID,
        summarizer_provider=FakeProvider(responses=[]),
        calendar_client=FakeCalendar(),
        caller_timezone="America/Toronto",
    )


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
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The visitor supplies a timezone on a later turn (a real client would
    resend ChatRequest.timezone) — intent, already recorded, isn't re-asked.
    """
    _configure_policy(monkeypatch, _policy())
    context.caller_timezone = None
    await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)

    context.caller_timezone = "America/Toronto"
    result = await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)

    assert "slots" in result
    state = await load_booking_state(context.db, SID)
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
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_policy(monkeypatch, _policy())
    first = await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)
    assert first["round"] == 1

    second = await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)
    assert second["round"] == 2

    state = await load_booking_state(context.db, SID)
    assert state.step is Step.SLOTS_PROPOSED
    assert state.proposal_rounds == 2


async def test_re_propose_excludes_previously_offered_slots(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
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
    first = await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)
    assert len(first["slots"]) == 1  # type: ignore[arg-type]

    second = await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)
    assert second["slots"] == []


async def test_third_call_widens_the_window_and_increments_round_again(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_policy(monkeypatch, _policy())
    await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)  # round 1
    await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)  # round 2

    third = await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)

    assert third["round"] == 3
    state = await load_booking_state(context.db, SID)
    assert state.step is Step.SLOTS_PROPOSED
    assert state.proposal_rounds == 3


async def test_fourth_call_falls_back_to_email_and_abandons(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_policy(monkeypatch, _policy())
    for _ in range(3):
        await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)

    fourth = await execute_tool(_call(date_from="2026-08-03", date_to="2026-08-03"), context)

    assert fourth["fallback"] is True
    assert "email" in str(fourth["message"]).lower()
    state = await load_booking_state(context.db, SID)
    assert state.step is Step.ABANDONED


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
