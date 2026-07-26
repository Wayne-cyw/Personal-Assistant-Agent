import json
import logging
from collections.abc import Generator

import httpx
import pytest
from fastapi import FastAPI, Request

from app.api.errors import register_exception_handlers
from app.middleware.logging import LoggingMiddleware, request_logger


def _build_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.add_middleware(LoggingMiddleware)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"pong": "ok"}

    @app.get("/with-session")
    async def with_session(request: Request) -> dict[str, str]:
        request.state.session_id = "sess-test-1"
        request.state.llm_tokens_in = 42
        request.state.llm_tokens_out = 7
        return {"ok": "yes"}

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("Authorization: Bearer sk-dummy-secret-value-99999")

    return app


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def request_log(caplog: pytest.LogCaptureFixture) -> Generator[_ListHandler]:
    """request_logger has propagate=False (Issue #6: avoid duplicate output
    under uvicorn's own logging config), so pytest's caplog — which listens
    via the root logger — cannot see its records. Attach a handler directly
    to request_logger instead. Also enables caplog for every other logger
    (e.g. app.api.errors), which does propagate normally.
    """
    caplog.set_level(logging.DEBUG)
    handler = _ListHandler()
    request_logger.addHandler(handler)
    try:
        yield handler
    finally:
        request_logger.removeHandler(handler)


async def test_one_json_line_per_request_parseable_by_json(request_log: _ListHandler) -> None:
    app = _build_app()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ping")

    assert response.status_code == 200
    assert len(request_log.records) == 1
    entry = json.loads(request_log.records[0].getMessage())
    assert entry["path"] == "/ping"
    assert entry["status"] == 200
    assert isinstance(entry["latency_ms"], int | float)


async def test_session_id_and_token_fields_populated_when_set(request_log: _ListHandler) -> None:
    app = _build_app()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/with-session")

    entry = json.loads(request_log.records[-1].getMessage())
    assert entry["session_id"] == "sess-test-1"
    assert entry["llm_tokens_in"] == 42
    assert entry["llm_tokens_out"] == 7


async def test_fields_are_null_when_not_set(request_log: _ListHandler) -> None:
    app = _build_app()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/ping")

    entry = json.loads(request_log.records[-1].getMessage())
    assert entry["session_id"] is None
    assert entry["llm_tokens_in"] is None
    assert entry["llm_tokens_out"] is None
    assert entry["tool_name"] is None
    assert entry["turn_tag"] is None


async def test_status_reflects_final_response_after_exception_handling(
    request_log: _ListHandler,
) -> None:
    """The log line's status must be the response actually sent to the
    client (500, converted by the exception handler) — not raise past the
    middleware.
    """
    app = _build_app()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/boom")

    assert response.status_code == 500
    entry = json.loads(request_log.records[-1].getMessage())
    assert entry["status"] == 500


async def test_unhandled_exception_never_leaks_secret_in_any_log_output(
    request_log: _ListHandler, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression guard (Issue #6 acceptance criteria): a dummy Authorization
    header value carried by an exception must not appear anywhere in the
    captured log output — headers, message, or stringified repr — across
    every logger this request path touches, not just app.request.
    """
    app = _build_app()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/boom")

    assert response.status_code == 500
    assert "sk-dummy-secret-value-99999" not in response.text

    request_log_output = "\n".join(r.getMessage() for r in request_log.records)
    assert "sk-dummy-secret-value-99999" not in request_log_output

    errors_log_output = "\n".join(f"{r.name}:{r.getMessage()}" for r in caplog.records)
    assert "sk-dummy-secret-value-99999" not in errors_log_output


def test_request_logger_emits_pure_json_lines() -> None:
    """The configured handler formats with "%(message)s" only, so each
    logged line is exactly the JSON payload — required for `jq` to parse it
    directly (Issue #6 acceptance criteria).
    """
    assert request_logger.handlers, "request_logger must have a handler configured"
    formatter = request_logger.handlers[0].formatter
    assert formatter is not None
    assert formatter._fmt == "%(message)s"
