"""Unit tests for RateLimitMiddleware's pure helpers (Issue #24) — the
end-to-end blocking/allowing behavior (both caps, forwarded-IP handling,
zero-LLM-spend) is covered by tests/integration/test_rate_limit_middleware.py;
these focus on the smaller pieces in isolation, plus proving the
middleware wires the correct window/limit into check_and_increment_rate_limit.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from starlette.requests import Request

from app.config import settings
from app.middleware import rate_limit as rate_limit_module
from app.middleware.rate_limit import (
    RateLimitMiddleware,
    _client_ip,
    _extract_session_id,
    _replay_receive,
)


def _request(
    *, headers: dict[str, str] | None = None, client: tuple[str, int] | None = ("1.2.3.4", 123)
) -> Request:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope: dict[str, object] = {"type": "http", "headers": raw_headers, "client": client}
    return Request(scope)


# --- _extract_session_id -------------------------------------------------------


def test_extract_session_id_from_valid_body() -> None:
    body = b'{"session_id": "sess-1", "message": "hi"}'
    assert _extract_session_id(body) == "sess-1"


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not json",
        b"[]",
        b"null",
        b'"just a string"',
        b'{"message": "hi"}',
        b'{"session_id": 123}',
        b'{"session_id": ""}',
        b'{"session_id": null}',
        b"\xff\xfe not valid utf-8 \x00",
    ],
)
def test_extract_session_id_returns_none_for_anything_unusable(body: bytes) -> None:
    assert _extract_session_id(body) is None


# --- _client_ip ------------------------------------------------------------


def test_client_ip_uses_raw_peer_when_untrusted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "trust_x_forwarded_for", False)
    request = _request(headers={"X-Forwarded-For": "9.9.9.9"}, client=("1.2.3.4", 123))
    assert _client_ip(request) == "1.2.3.4"


def test_client_ip_uses_forwarded_header_first_entry_when_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "trust_x_forwarded_for", True)
    request = _request(
        headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.1, 10.0.0.2"}, client=("1.2.3.4", 123)
    )
    assert _client_ip(request) == "9.9.9.9"


def test_client_ip_falls_back_to_peer_when_trusted_but_header_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "trust_x_forwarded_for", True)
    request = _request(headers={}, client=("1.2.3.4", 123))
    assert _client_ip(request) == "1.2.3.4"


def test_client_ip_none_when_no_peer_and_not_trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "trust_x_forwarded_for", False)
    request = _request(client=None)
    assert _client_ip(request) is None


# --- _replay_receive ---------------------------------------------------------


async def test_replay_receive_returns_the_body_once_then_disconnect() -> None:
    receive = _replay_receive(b'{"session_id": "sess-1"}')

    first = await receive()
    assert first == {
        "type": "http.request",
        "body": b'{"session_id": "sess-1"}',
        "more_body": False,
    }

    second = await receive()
    assert second == {"type": "http.disconnect"}


# --- middleware wiring: window/limit passed through correctly ----------------


async def test_middleware_checks_both_caps_with_a_one_minute_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Doesn't re-test check_and_increment_rate_limit's own window/reset
    correctness (already covered directly in tests/unit/test_db_session.py)
    -- proves the middleware calls it with the right keys, limits, and
    window for both the session and IP counters.
    """
    monkeypatch.setattr(settings, "messages_per_session_per_min", 7)
    monkeypatch.setattr(settings, "messages_per_ip_per_min", 13)
    monkeypatch.setattr(settings, "trust_x_forwarded_for", False)

    calls: list[dict[str, object]] = []

    async def _fake_check(db: object, key: str, *, limit: int, window: timedelta | None) -> bool:
        calls.append({"key": key, "limit": limit, "window": window})
        return True

    monkeypatch.setattr(rate_limit_module, "check_and_increment_rate_limit", _fake_check)

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(rate_limit_module, "get_session_factory", lambda: _FakeSession)

    downstream_called = False

    async def _downstream_app(scope: dict, receive: object, send: object) -> None:
        nonlocal downstream_called
        downstream_called = True

    middleware = RateLimitMiddleware(_downstream_app)  # type: ignore[arg-type]

    body = b'{"session_id": "sess-1", "message": "hi"}'
    sent_body = False

    async def receive() -> dict[str, object]:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(_message: dict[str, object]) -> None:
        pass

    scope: dict[str, object] = {
        "type": "http",
        "path": "/v1/chat",
        "headers": [],
        "client": ("1.2.3.4", 123),
        "app": _FakeAppWithoutState(),
    }
    await middleware(scope, receive, send)  # type: ignore[arg-type]

    assert downstream_called is True
    assert {"key": "sess:sess-1", "limit": 7, "window": timedelta(minutes=1)} in calls
    assert {"key": "ip:1.2.3.4", "limit": 13, "window": timedelta(minutes=1)} in calls


class _FakeAppWithoutState:
    """Mimics FastAPI's app.state -- an empty namespace object with no
    db_session_factory attribute set, so the middleware falls back to
    get_session_factory() (monkeypatched above).
    """

    class _State:
        pass

    state = _State()
