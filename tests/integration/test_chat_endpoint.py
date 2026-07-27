from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.agent.intro import INTRO_MESSAGE
from app.agent.providers.base import (
    LLMResponse,
    Message,
    StreamEvent,
    ToolDef,
    UpstreamError,
    Usage,
)
from app.agent.providers.fake import FakeProvider
from app.api.chat import get_main_provider
from app.db.models import Base, SessionRow
from app.db.session import get_db
from app.main import app


class _RaisingProvider:
    """A provider that always raises, to exercise the 503/500 error paths."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def complete(
        self, messages: list[Message], tools: list[ToolDef], max_tokens: int
    ) -> LLMResponse:
        raise self._exc

    async def complete_stream(
        self, messages: list[Message], tools: list[ToolDef], max_tokens: int
    ) -> AsyncIterator[StreamEvent]:
        raise self._exc
        yield  # pragma: no cover — unreachable; makes this an async generator function


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine]:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def client(engine: AsyncEngine) -> AsyncGenerator[httpx.AsyncClient]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    try:
        # raise_app_exceptions=False: let the registered Exception handler
        # convert unhandled errors into the real HTTP response a deployed
        # server would send, instead of re-raising for pytest debugging.
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_main_provider, None)


def _response(text: str) -> LLMResponse:
    usage = Usage(input_tokens=1, output_tokens=1)
    return LLMResponse(text=text, usage=usage, finish_reason="stop")


def _use_fake_provider(responses: list[LLMResponse]) -> FakeProvider:
    fake = FakeProvider(responses=responses)
    app.dependency_overrides[get_main_provider] = lambda: fake
    return fake


async def _get_session(engine: AsyncEngine, session_id: str) -> SessionRow | None:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as db:
        return await db.get(SessionRow, session_id)


async def _prime_past_turn_zero(client: httpx.AsyncClient, session_id: str) -> None:
    """Every session's first message gets the deterministic intro reply
    (Issue #9), not an LLM call — send and discard one throwaway turn-zero
    message so the rest of a test can exercise the real LLM-driven path.
    """
    response = await client.post(
        "/v1/chat", json={"session_id": session_id, "message": "(priming turn zero)"}
    )
    assert response.status_code == 200


async def test_two_turn_conversation_remembers_first_turn(client: httpx.AsyncClient) -> None:
    await _prime_past_turn_zero(client, "sess-1")
    fake = _use_fake_provider([_response("Nice to meet you, Sam."), _response("Your name is Sam.")])

    # "call me Sam" deliberately avoids the Issue #9 name-extraction patterns
    # ("my name is"/"I'm") so this test stays focused on history/memory, not
    # visitor-info capture (covered separately below).
    first = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "call me Sam"}
    )
    assert first.status_code == 200
    assert first.json() == {"reply": "Nice to meet you, Sam.", "type": "message", "data": None}

    second = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "what's my name?"}
    )
    assert second.status_code == 200
    assert second.json()["reply"] == "Your name is Sam."

    second_call_messages = fake.calls[1]
    contents = [m.content for m in second_call_messages]
    assert "call me Sam" in contents
    assert "Nice to meet you, Sam." in contents
    assert "what's my name?" in contents


async def test_fresh_session_first_reply_is_byte_identical_to_intro(
    client: httpx.AsyncClient,
) -> None:
    _use_fake_provider([_response("unused — turn zero must not call the LLM")])

    response = await client.post(
        "/v1/chat", json={"session_id": "sess-new", "message": "hello?"}
    )

    assert response.status_code == 200
    assert response.json() == {"reply": INTRO_MESSAGE, "type": "message", "data": None}


async def test_turn_zero_spends_zero_llm_tokens(client: httpx.AsyncClient) -> None:
    fake = _use_fake_provider([_response("unused")])

    await client.post("/v1/chat", json={"session_id": "sess-new", "message": "hello?"})

    assert fake.calls == []


async def test_turn_zero_question_is_answered_on_next_send(client: httpx.AsyncClient) -> None:
    """Issue #9: if the caller's first payload already contains a question,
    the prefix is still returned for that call; the question is answered on
    the next send, and the first message remains in history for that reply.
    """
    fake = _use_fake_provider([_response("Yes, they know Rust.")])

    first = await client.post(
        "/v1/chat", json={"session_id": "sess-new", "message": "does the owner know Rust?"}
    )
    assert first.status_code == 200
    assert first.json()["reply"] == INTRO_MESSAGE

    second = await client.post(
        "/v1/chat", json={"session_id": "sess-new", "message": "well?"}
    )
    assert second.status_code == 200
    assert second.json()["reply"] == "Yes, they know Rust."

    contents = [m.content for m in fake.calls[0]]
    assert "does the owner know Rust?" in contents
    assert "well?" in contents


async def test_name_volunteered_on_turn_zero_is_captured_silently(
    client: httpx.AsyncClient, engine: AsyncEngine
) -> None:
    """Turn zero's reply must stay byte-identical to intro.md even if a name
    was volunteered in that same first message (Issue #9) — capture happens,
    acknowledgment does not.
    """
    _use_fake_provider([_response("unused — turn zero must not call the LLM")])

    response = await client.post(
        "/v1/chat", json={"session_id": "sess-new", "message": "hi, my name is Priya"}
    )

    assert response.json()["reply"] == INTRO_MESSAGE  # no appended acknowledgment
    session = await _get_session(engine, "sess-new")
    assert session is not None
    assert session.visitor_name == "Priya"


async def test_name_volunteered_after_turn_zero_is_captured_and_acknowledged_once(
    client: httpx.AsyncClient, engine: AsyncEngine
) -> None:
    await _prime_past_turn_zero(client, "sess-1")
    fake = _use_fake_provider([_response("Nice to meet you."), _response("How can I help?")])

    first = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "I'm Priya"}
    )
    assert first.json()["reply"] == "Nice to meet you.\n\n(Thanks for sharing your name, Priya!)"

    session = await _get_session(engine, "sess-1")
    assert session is not None
    assert session.visitor_name == "Priya"

    # Repeating the name on a later turn must not re-acknowledge or overwrite.
    second = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "I'm Priya, just checking in"}
    )
    assert second.json()["reply"] == "How can I help?"  # no acknowledgment appended twice

    assert fake.calls  # sanity: the LLM path was actually exercised


async def test_linkedin_url_captured_from_message(
    client: httpx.AsyncClient, engine: AsyncEngine
) -> None:
    await _prime_past_turn_zero(client, "sess-1")
    _use_fake_provider([_response("Got it.")])

    response = await client.post(
        "/v1/chat",
        json={
            "session_id": "sess-1",
            "message": "here's my linkedin: https://www.linkedin.com/in/priya-example",
        },
    )

    assert response.json()["reply"] == "Got it.\n\n(Thanks for sharing your LinkedIn!)"
    session = await _get_session(engine, "sess-1")
    assert session is not None
    assert session.visitor_linkedin == "https://www.linkedin.com/in/priya-example"


async def test_uses_real_system_prompt_not_a_placeholder(client: httpx.AsyncClient) -> None:
    from app.agent.prompts import SYSTEM_PROMPT

    await _prime_past_turn_zero(client, "sess-1")
    fake = _use_fake_provider([_response("hi")])

    await client.post("/v1/chat", json={"session_id": "sess-1", "message": "hello"})

    system_message = fake.calls[0][0]
    assert system_message.role == "system"
    assert system_message.content == SYSTEM_PROMPT


async def test_history_window_covers_ten_full_turns(client: httpx.AsyncClient) -> None:
    """Regression test: n passed to recent_messages must be row count (2 per
    turn), not turn count — otherwise "last 10 turns" only covers 5.
    """
    await _prime_past_turn_zero(client, "sess-1")
    fake = _use_fake_provider([_response(f"reply {i}") for i in range(11)])

    for i in range(11):
        await client.post("/v1/chat", json={"session_id": "sess-1", "message": f"turn {i}"})

    last_call_messages = fake.calls[-1]
    contents = [m.content for m in last_call_messages]
    assert "turn 0" in contents  # still in the 10-turn window as of turn 10 (0-indexed)
    assert "reply 0" in contents


async def test_empty_session_id_returns_invalid_request_envelope(client: httpx.AsyncClient) -> None:
    _use_fake_provider([_response("unused")])

    response = await client.post("/v1/chat", json={"session_id": "", "message": "hello"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


async def test_user_message_persisted_even_when_provider_fails(client: httpx.AsyncClient) -> None:
    """4.3: the DB log is complete — a message that triggers an upstream
    failure must still be recorded, even though no reply exists for it.
    """
    await _prime_past_turn_zero(client, "sess-1")
    app.dependency_overrides[get_main_provider] = lambda: _RaisingProvider(
        UpstreamError(status_code=503, error_type="APIStatusError", message="boom")
    )

    failed = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "this triggers a failure"}
    )
    assert failed.status_code == 503

    fake = _use_fake_provider([_response("ok now")])
    recovered = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "second attempt"}
    )
    assert recovered.status_code == 200

    contents = [m.content for m in fake.calls[0]]
    assert "this triggers a failure" in contents
    assert "second attempt" in contents


async def test_empty_message_returns_invalid_request_envelope(client: httpx.AsyncClient) -> None:
    _use_fake_provider([_response("unused")])

    response = await client.post("/v1/chat", json={"session_id": "sess-1", "message": ""})

    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_request", "message": "The request was invalid."}
    }


async def test_missing_field_returns_invalid_request_envelope(client: httpx.AsyncClient) -> None:
    _use_fake_provider([_response("unused")])

    response = await client.post("/v1/chat", json={"session_id": "sess-1"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


async def test_upstream_error_returns_503_envelope(client: httpx.AsyncClient) -> None:
    await _prime_past_turn_zero(client, "sess-1")
    app.dependency_overrides[get_main_provider] = lambda: _RaisingProvider(
        UpstreamError(status_code=503, error_type="APIStatusError", message="boom")
    )

    response = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "hello"}
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "upstream_unavailable",
            "message": "The service is temporarily unavailable. Please try again.",
        }
    }


async def test_unhandled_exception_returns_500_envelope_without_leaking_details(
    client: httpx.AsyncClient,
) -> None:
    await _prime_past_turn_zero(client, "sess-1")
    app.dependency_overrides[get_main_provider] = lambda: _RaisingProvider(
        RuntimeError("sensitive internal detail that must never reach the client")
    )

    response = await client.post(
        "/v1/chat", json={"session_id": "sess-1", "message": "hello"}
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "An internal error occurred."}
    }
    assert "sensitive internal detail" not in response.text


async def test_health_endpoint(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok", "llm": "unchecked"}


async def test_openapi_schema_includes_chat_contract(client: httpx.AsyncClient) -> None:
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/v1/chat" in paths
    assert "post" in paths["/v1/chat"]
