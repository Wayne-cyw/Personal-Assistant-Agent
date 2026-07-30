"""End-to-end booking-flow integration tests (Issue #20 acceptance
criteria): happy path through slot_selected, and the negotiation cap's
re-propose -> widen -> email-fallback progression. Uses FakeProvider (the
LLM's *decisions* — when to call calendar_find_slots, when to answer in
prose — are scripted) and FakeCalendar (no real Google API calls); the
availability policy is stubbed the same way tests/unit/test_calendar_find_
slots_tool.py does, since knowledge/availability_policy.md is still a
template pending Issue #13.
"""

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import app.tools.registry as registry_module
from app.agent.providers.base import LLMResponse, ToolCall, Usage
from app.agent.providers.fake import FakeProvider
from app.api.chat import get_chat_calendar_client, get_main_provider
from app.booking.slots import AvailabilityPolicy, AvailabilityPolicyError
from app.booking.state import BookingState, Step
from app.db.models import Base
from app.db.session import get_db, load_booking_state
from app.main import app
from app.tools.fake_calendar import FakeCalendar

TZ = ZoneInfo("America/Toronto")


def _policy() -> AvailabilityPolicy:
    return AvailabilityPolicy(
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


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine]:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def client(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[httpx.AsyncClient]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession]:
        async with session_factory() as session:
            yield session

    monkeypatch.setattr(registry_module, "get_availability_policy", _policy)
    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_chat_calendar_client] = lambda: FakeCalendar()
    try:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_main_provider, None)
        app.dependency_overrides.pop(get_chat_calendar_client, None)


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text, usage=Usage(input_tokens=1, output_tokens=1), finish_reason="stop"
    )


def _find_slots_call(
    call_id: str, date_from: str = "2026-08-03", date_to: str = "2026-08-03"
) -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="calendar_find_slots",
                arguments={"date_from": date_from, "date_to": date_to},
            )
        ],
        usage=Usage(input_tokens=1, output_tokens=1),
        finish_reason="tool_calls",
    )


def _use_fake_provider(responses: list[LLMResponse]) -> FakeProvider:
    fake = FakeProvider(responses=responses)
    app.dependency_overrides[get_main_provider] = lambda: fake
    return fake


async def _get_booking_state(engine: AsyncEngine, session_id: str) -> BookingState:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as db:
        return await load_booking_state(db, session_id)


async def _prime_past_turn_zero(client: httpx.AsyncClient, session_id: str) -> None:
    response = await client.post(
        "/v1/chat", json={"session_id": session_id, "message": "(priming turn zero)"}
    )
    assert response.status_code == 200


async def test_happy_path_intent_to_slot_selected(
    client: httpx.AsyncClient, engine: AsyncEngine
) -> None:
    await _prime_past_turn_zero(client, "sess-1")
    _use_fake_provider(
        [
            _find_slots_call("call_1"),
            _response("Here are some times that work — let me know which suits you."),
            _response("Great, you're set for that time."),
        ]
    )

    proposal = await client.post(
        "/v1/chat",
        json={
            "session_id": "sess-1",
            "message": "can we book a call next week?",
            "timezone": "America/Toronto",
        },
    )
    assert proposal.status_code == 200
    body = proposal.json()
    assert body["type"] == "booking_proposal"
    assert body["data"]["round"] == 1
    slots = body["data"]["slots"]
    assert len(slots) >= 2  # test picks slot "2" below

    state_after_proposal = await _get_booking_state(engine, "sess-1")
    assert state_after_proposal.step is Step.SLOTS_PROPOSED
    assert state_after_proposal.timezone_name == "America/Toronto"

    selection = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "2"}
    )
    assert selection.status_code == 200
    assert selection.json()["reply"] == "Great, you're set for that time."

    state_after_selection = await _get_booking_state(engine, "sess-1")
    assert state_after_selection.step is Step.SLOT_SELECTED
    assert state_after_selection.selected_slot_json == slots[1]


async def test_repeated_rejection_widens_then_falls_back_to_email(
    client: httpx.AsyncClient, engine: AsyncEngine
) -> None:
    await _prime_past_turn_zero(client, "sess-1")
    _use_fake_provider(
        [
            _find_slots_call("call_1"),
            _response("Here are some times."),  # round 1
            _find_slots_call("call_2"),
            _response("Let me check a few other times."),  # round 2 (re-propose)
            _find_slots_call("call_3"),
            _response("I widened the search window."),  # round 3 (widen)
            _find_slots_call("call_4"),
            _response("No luck — please email the owner directly to find a time."),  # fallback
        ]
    )

    first = await client.post(
        "/v1/chat",
        json={
            "session_id": "sess-1",
            "message": "can we book a call?",
            "timezone": "America/Toronto",
        },
    )
    assert first.json()["type"] == "booking_proposal"
    assert first.json()["data"]["round"] == 1

    second = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "none of those work for me"}
    )
    assert second.json()["data"]["round"] == 2

    third = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "none of those work either"}
    )
    assert third.json()["data"]["round"] == 3

    fourth = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "still none of those work"}
    )
    assert fourth.status_code == 200
    fourth_body = fourth.json()
    # No dedicated response type for the email-fallback outcome (Engineering
    # Guide 4.8) — it's an ordinary message reply.
    assert fourth_body["type"] == "message"
    assert fourth_body["data"] is None
    assert fourth_body["reply"] == "No luck — please email the owner directly to find a time."

    final_state = await _get_booking_state(engine, "sess-1")
    assert final_state.step is Step.ABANDONED


async def test_availability_policy_not_configured_degrades_gracefully(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A visitor trying to book while the (still-template, Issue #13)
    availability policy genuinely isn't configured gets a normal message
    reply, not a 500 — execute_tool's structured-error path, exercised
    end-to-end.
    """

    def _raise() -> AvailabilityPolicy:
        raise AvailabilityPolicyError("still a template")

    monkeypatch.setattr(registry_module, "get_availability_policy", _raise)
    await _prime_past_turn_zero(client, "sess-1")
    _use_fake_provider(
        [
            _find_slots_call("call_1"),
            _response("Availability isn't set up yet — I'll have the owner follow up by email."),
        ]
    )

    response = await client.post(
        "/v1/chat",
        json={"session_id": "sess-1", "message": "can we book a call?", "timezone": "UTC"},
    )

    assert response.status_code == 200
    assert response.json()["type"] == "message"
