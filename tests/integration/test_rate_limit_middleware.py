"""RateLimitMiddleware (Issue #24) — per-session and per-IP message rate
limits for all /v1/chat traffic, checked before routing/the LLM ever runs.

Every session's *first* message is turn zero (Issue #9: a deterministic,
owner-authored intro, zero LLM tokens spent, per app/agent/intro.py) —
these tests prime each session past turn zero first, so later assertions
about the LLM-backed reply/call count aren't confused by that separate
zero-LLM-call path.
"""

from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.agent.providers.base import LLMResponse, Usage
from app.agent.providers.fake import FakeProvider
from app.api.chat import get_main_provider
from app.api.health import get_health_calendar_client
from app.config import settings
from app.db.models import Base
from app.db.session import get_db
from app.main import app
from app.tools.fake_calendar import FakeCalendar


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine]:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


def _client(
    engine: AsyncEngine, *, peer: tuple[str, int] = ("127.0.0.1", 123)
) -> httpx.AsyncClient:
    """Unlike the other integration test files' `client` fixture, this
    takes the simulated TCP peer address as a parameter (httpx.ASGITransport's
    own `client` kwarg) -- per-IP rate-limit tests need to simulate
    requests arriving from *different* IPs, which a single fixture-wide
    default can't do.
    """
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    app.state.db_session_factory = session_factory
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False, client=peer)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _teardown() -> None:
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_main_provider, None)
    if hasattr(app.state, "db_session_factory"):
        del app.state.db_session_factory


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text, usage=Usage(input_tokens=1, output_tokens=1), finish_reason="stop"
    )


def _use_fake_provider(responses: list[LLMResponse]) -> FakeProvider:
    fake = FakeProvider(responses=responses)
    app.dependency_overrides[get_main_provider] = lambda: fake
    return fake


async def _send(
    client: httpx.AsyncClient, session_id: str, message: str = "hello"
) -> httpx.Response:
    return await client.post("/v1/chat", json={"session_id": session_id, "message": message})


async def _prime(client: httpx.AsyncClient, session_id: str) -> httpx.Response:
    """Turn zero -- counts against the rate limit like any other request,
    but never calls the LLM, so tests that need a *real* (LLM-backed)
    exchange send this first to get each session past it.
    """
    return await _send(client, session_id, "(priming turn zero)")


async def test_session_cap_returns_429_with_the_documented_envelope(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "messages_per_session_per_min", 3)
    # Loose enough that the per-IP cap doesn't trip first and confound
    # which cap actually produced the 429.
    monkeypatch.setattr(settings, "messages_per_ip_per_min", 100)
    _use_fake_provider([_response(f"reply {i}") for i in range(10)])

    async with _client(engine) as client:
        try:
            await _prime(client, "sess-1")  # 1
            first = await _send(client, "sess-1")  # 2
            assert first.status_code == 200
            second = await _send(client, "sess-1")  # 3
            assert second.status_code == 200

            third = await _send(client, "sess-1")  # 4 -- over the cap of 3

            assert third.status_code == 429
            assert third.json() == {
                "error": {
                    "code": "rate_limited",
                    "message": "Too many requests. Try again in a minute.",
                }
            }
        finally:
            _teardown()


async def test_session_cap_is_independent_per_session(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "messages_per_session_per_min", 1)
    monkeypatch.setattr(settings, "messages_per_ip_per_min", 100)

    async with _client(engine) as client:
        try:
            first = await _prime(client, "sess-a")
            assert first.status_code == 200
            # A different session, same (default) IP -- its own cap is untouched.
            second = await _prime(client, "sess-b")
            assert second.status_code == 200
        finally:
            _teardown()


async def test_ip_cap_is_shared_across_sessions_from_the_same_ip(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "messages_per_session_per_min", 100)
    monkeypatch.setattr(settings, "messages_per_ip_per_min", 1)

    async with _client(engine, peer=("9.9.9.9", 1)) as client:
        try:
            first = await _prime(client, "sess-a")
            assert first.status_code == 200

            second = await _prime(client, "sess-b")

            assert second.status_code == 429
        finally:
            _teardown()


async def test_zero_llm_spend_for_a_blocked_request(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance criterion: a blocked request never reaches the handler
    or the LLM. FakeProvider.calls records every messages list it was
    actually invoked with (app/agent/providers/fake.py) -- asserting its
    length directly, rather than inferring non-invocation indirectly,
    proves the blocked call never reached the provider at all.
    """
    monkeypatch.setattr(settings, "messages_per_session_per_min", 2)
    monkeypatch.setattr(settings, "messages_per_ip_per_min", 100)
    fake = _use_fake_provider([_response("only response")])

    async with _client(engine) as client:
        try:
            await _prime(client, "sess-1")  # 1, no LLM call (turn zero)
            first = await _send(client, "sess-1")  # 2, one LLM call
            assert first.status_code == 200
            assert first.json()["reply"] == "only response"
            assert len(fake.calls) == 1

            blocked = await _send(client, "sess-1")  # 3, over the cap of 2

            assert blocked.status_code == 429
            assert len(fake.calls) == 1  # unchanged -- never reached the provider
        finally:
            _teardown()


async def test_forwarded_for_ignored_when_not_trusted(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "trust_x_forwarded_for", False)
    monkeypatch.setattr(settings, "messages_per_session_per_min", 100)
    monkeypatch.setattr(settings, "messages_per_ip_per_min", 1)

    async with _client(engine, peer=("5.5.5.5", 1)) as client:
        try:
            first = await client.post(
                "/v1/chat",
                json={"session_id": "sess-a", "message": "hi"},
                headers={"X-Forwarded-For": "1.1.1.1"},
            )
            assert first.status_code == 200

            # Different claimed X-Forwarded-For, but the real (raw TCP)
            # peer is the same "5.5.5.5" -- untrusted, so the header is
            # ignored and this still counts against the same IP bucket.
            second = await client.post(
                "/v1/chat",
                json={"session_id": "sess-b", "message": "hi"},
                headers={"X-Forwarded-For": "2.2.2.2"},
            )

            assert second.status_code == 429
        finally:
            _teardown()


async def test_forwarded_for_honored_when_trusted(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "trust_x_forwarded_for", True)
    monkeypatch.setattr(settings, "messages_per_session_per_min", 100)
    monkeypatch.setattr(settings, "messages_per_ip_per_min", 1)

    # Both requests share the same raw TCP peer (as they would behind one
    # reverse proxy) but claim different original clients via the header.
    async with _client(engine, peer=("10.0.0.1", 1)) as client:
        try:
            first = await client.post(
                "/v1/chat",
                json={"session_id": "sess-a", "message": "hi"},
                headers={"X-Forwarded-For": "1.1.1.1"},
            )
            assert first.status_code == 200

            second = await client.post(
                "/v1/chat",
                json={"session_id": "sess-b", "message": "hi"},
                headers={"X-Forwarded-For": "2.2.2.2, 10.0.0.1"},
            )

            # Trusted and a distinct claimed original client (the header's
            # *first* entry) -- its own, untouched IP bucket.
            assert second.status_code == 200
        finally:
            _teardown()


async def test_health_endpoint_is_not_rate_limited(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "messages_per_ip_per_min", 1)
    app.dependency_overrides[get_health_calendar_client] = lambda: FakeCalendar()
    async with _client(engine) as client:
        try:
            # Two health checks from the same peer would trip a
            # per-IP-per-minute cap of 1 if this path were covered by the
            # middleware -- it isn't (scoped to /v1/chat only).
            first = await client.get("/health")
            second = await client.get("/health")
            assert first.status_code == 200
            assert second.status_code == 200
        finally:
            app.dependency_overrides.pop(get_health_calendar_client, None)
            _teardown()
