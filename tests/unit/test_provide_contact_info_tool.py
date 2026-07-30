"""provide_contact_info tool handler (Issue #21): name/email collection and
code-side validation, and the immediate contact_info_collected -> confirmed
double-hop once contact info is valid.
"""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.agent.providers.base import ToolCall
from app.agent.providers.fake import FakeProvider
from app.booking.state import BookingState, Step, allowed_tools_for
from app.db.models import Base, SessionRow
from app.db.session import get_or_create_session, load_booking_state, save_booking_state
from app.tools.context import ToolContext
from app.tools.fake_calendar import FakeCalendar
from app.tools.registry import execute_tool, tool_defs_for_step

SID = "sess-1"


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
    )


async def _put_state_at_slot_selected(db: AsyncSession) -> None:
    await save_booking_state(
        db,
        BookingState(
            session_id=SID,
            step=Step.SLOT_SELECTED,
            timezone_name="America/Toronto",
            selected_slot_json={
                "slot_id": "s1",
                "start_iso": "2026-08-03T09:00:00-04:00",
                "end_iso": "2026-08-03T09:30:00-04:00",
                "label": "Mon Aug 3, 9:00-9:30 AM EDT",
            },
        ),
    )


def _call(**args: object) -> ToolCall:
    return ToolCall(id="call_1", name="provide_contact_info", arguments=args)


# --- tool gating --------------------------------------------------------------


def test_provide_contact_info_offered_from_slot_selected() -> None:
    names = [t.name for t in tool_defs_for_step(Step.SLOT_SELECTED)]
    assert "provide_contact_info" in names


def test_provide_contact_info_not_offered_outside_slot_selected() -> None:
    for step in (
        Step.IDLE,
        Step.INTENT_DETECTED,
        Step.SLOTS_PROPOSED,
        Step.CONTACT_INFO_COLLECTED,
        Step.CONFIRMED,
        Step.BOOKING_CREATED,
        Step.ABANDONED,
    ):
        names = [t.name for t in tool_defs_for_step(step)]
        assert "provide_contact_info" not in names


def test_provide_contact_info_matches_allowed_tools_for() -> None:
    assert allowed_tools_for(Step.SLOT_SELECTED) == ["provide_contact_info"]


# --- happy path ----------------------------------------------------------------


async def test_valid_contact_info_reaches_confirmed_with_summary(context: ToolContext) -> None:
    await _put_state_at_slot_selected(context.db)

    result = await execute_tool(
        _call(name="Priya Patel", email="priya@example.com"), context
    )

    assert "error" not in result
    summary = result["confirmation_summary"]
    assert isinstance(summary, dict)
    assert summary["timezone"] == "America/Toronto"
    assert summary["name"] == "Priya Patel"
    assert summary["email"] == "priya@example.com"
    assert summary["slot"] == {
        "slot_id": "s1",
        "start_iso": "2026-08-03T09:00:00-04:00",
        "end_iso": "2026-08-03T09:30:00-04:00",
        "label": "Mon Aug 3, 9:00-9:30 AM EDT",
    }

    state = await load_booking_state(context.db, SID)
    assert state.step is Step.CONFIRMED
    assert state.contact_name == "Priya Patel"
    assert state.contact_email == "priya@example.com"


async def test_valid_name_feeds_the_pinned_profile(context: ToolContext) -> None:
    await _put_state_at_slot_selected(context.db)

    await execute_tool(_call(name="Priya Patel", email="priya@example.com"), context)

    session = await context.db.get(SessionRow, SID)
    assert session is not None
    assert session.visitor_name == "Priya Patel"


async def test_name_is_sanitized_the_same_way_as_save_visitor_info(context: ToolContext) -> None:
    await _put_state_at_slot_selected(context.db)

    result = await execute_tool(
        _call(name="  Priya   Patel  \n", email="priya@example.com"), context
    )

    summary = result["confirmation_summary"]
    assert isinstance(summary, dict)
    assert summary["name"] == "Priya Patel"


# --- email validation ----------------------------------------------------------


@pytest.mark.parametrize(
    "bad_email",
    [
        "not-an-email",
        "missing-at-sign.com",
        "@no-local-part.com",
        "no-domain@",
        "spaced out@example.com",
        "trailing-dot@example.",
        "x" * 255 + "@example.com",  # over the RFC 5321 length limit
        "injected\n@example.com",  # newline — header-injection-style
        "injected@example.com\r\nBcc: evil@example.com",
    ],
)
async def test_invalid_emails_are_rejected_without_advancing_state(
    context: ToolContext, bad_email: str
) -> None:
    await _put_state_at_slot_selected(context.db)

    result = await execute_tool(_call(name="Priya Patel", email=bad_email), context)

    assert result["error"] == "email_invalid"
    state = await load_booking_state(context.db, SID)
    assert state.step is Step.SLOT_SELECTED
    assert state.contact_email is None


@pytest.mark.parametrize(
    "good_email",
    [
        "priya@example.com",
        "priya.patel+booking@sub.example.co.uk",
        "p_riya-1@example-domain.com",
    ],
)
async def test_valid_emails_are_accepted(context: ToolContext, good_email: str) -> None:
    await _put_state_at_slot_selected(context.db)

    result = await execute_tool(_call(name="Priya Patel", email=good_email), context)

    assert "error" not in result


async def test_invalid_email_can_be_corrected_on_a_later_call(context: ToolContext) -> None:
    await _put_state_at_slot_selected(context.db)
    await execute_tool(_call(name="Priya Patel", email="not-an-email"), context)

    result = await execute_tool(_call(name="Priya Patel", email="priya@example.com"), context)

    assert "error" not in result
    state = await load_booking_state(context.db, SID)
    assert state.step is Step.CONFIRMED


# --- name validation -------------------------------------------------------------


async def test_empty_name_after_sanitization_is_rejected(context: ToolContext) -> None:
    await _put_state_at_slot_selected(context.db)

    result = await execute_tool(_call(name="   ", email="priya@example.com"), context)

    assert result["error"] == "name_invalid"
    state = await load_booking_state(context.db, SID)
    assert state.step is Step.SLOT_SELECTED


# --- defense in depth: state gating checked inside the handler too -----------


@pytest.mark.parametrize(
    "step",
    [
        Step.IDLE,
        Step.INTENT_DETECTED,
        Step.SLOTS_PROPOSED,
        Step.CONTACT_INFO_COLLECTED,
        Step.CONFIRMED,
    ],
)
async def test_handler_refuses_outside_slot_selected(context: ToolContext, step: Step) -> None:
    await save_booking_state(context.db, BookingState(session_id=SID, step=step))

    result = await execute_tool(_call(name="Priya Patel", email="priya@example.com"), context)

    assert "error" in result
    state = await load_booking_state(context.db, SID)
    assert state.step is step  # unchanged
