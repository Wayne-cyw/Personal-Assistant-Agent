"""Error envelope and exception handlers (Engineering Guide 4.8).

`message` is always a fixed, hand-written, generic string per code — never
an interpolated exception. This is what keeps secrets (API keys, DB
credentials) that a raw exception object might carry out of every client
response, no matter where the exception originated.
"""

from __future__ import annotations

import logging
import traceback
from enum import StrEnum

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.agent.providers.base import UpstreamError
from app.agent.session_lock import SessionBusyError
from app.safety.pii import redact

logger = logging.getLogger(__name__)


class ErrorCode(StrEnum):
    RATE_LIMITED = "rate_limited"
    INVALID_REQUEST = "invalid_request"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    INTERNAL_ERROR = "internal_error"


_FIXED_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.RATE_LIMITED: "Too many requests. Try again in a minute.",
    ErrorCode.INVALID_REQUEST: "The request was invalid.",
    ErrorCode.UPSTREAM_UNAVAILABLE: "The service is temporarily unavailable. Please try again.",
    ErrorCode.INTERNAL_ERROR: "An internal error occurred.",
}

_STATUS_CODES: dict[ErrorCode, int] = {
    ErrorCode.RATE_LIMITED: status.HTTP_429_TOO_MANY_REQUESTS,
    ErrorCode.INVALID_REQUEST: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.UPSTREAM_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.INTERNAL_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def error_response(code: ErrorCode) -> JSONResponse:
    """Public (not just used by the exception handlers below): Issue #24's
    rate-limit middleware constructs its own 429 response outside FastAPI's
    exception-handling machinery (it short-circuits before the request
    ever reaches routing), and needs the exact same envelope shape rather
    than a hand-duplicated copy that could drift from this one.
    """
    return JSONResponse(
        status_code=_STATUS_CODES[code],
        content={"error": {"code": code.value, "message": _FIXED_MESSAGES[code]}},
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return error_response(ErrorCode.INVALID_REQUEST)

    @app.exception_handler(SessionBusyError)
    async def _handle_session_busy(_request: Request, exc: SessionBusyError) -> JSONResponse:
        # str(exc) is just the session_id (Issue #11) — no secret-bearing
        # payload risk here, unlike the generic-exception handler below.
        logger.info("session turn lock busy", extra={"session_id": str(exc)})
        return error_response(ErrorCode.RATE_LIMITED)

    @app.exception_handler(UpstreamError)
    async def _handle_upstream_error(_request: Request, exc: UpstreamError) -> JSONResponse:
        # exc carries only the already-sanitized {status_code, error_type,
        # message} fields (app/agent/providers/base.py) — safe to log, but
        # never echoed into the response body regardless.
        logger.warning(
            "upstream provider error",
            extra={"status_code": exc.status_code, "error_type": exc.error_type},
        )
        return error_response(ErrorCode.UPSTREAM_UNAVAILABLE)

    @app.exception_handler(Exception)
    async def _handle_unhandled_exception(_request: Request, exc: Exception) -> JSONResponse:
        # Full traceback logged server-side only, never in the response body
        # (Engineering Guide 4.8). The traceback is built and redacted
        # explicitly rather than via logger.exception(exc_info=exc) — that
        # would let Python's logging machinery format str(exc) and the raw
        # traceback straight into the log unredacted, bypassing the safety
        # net an arbitrary (not pre-sanitized like UpstreamError) exception
        # needs (Issue #6).
        #
        # Unlike the UpstreamError handler above, this can't do construction-
        # time (layer-1) sanitization: an arbitrary Exception has no typed,
        # pre-sanitized shape to extract fields from — that's the whole
        # reason Issue #5 wants the raw traceback here (server-side
        # debuggability for a truly unexpected error). So this path leans
        # entirely on redact() (layer 2), which is exactly the "safety net
        # against a future mistake" role Issue #6 describes for it.
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.error("unhandled exception\n%s", redact(tb_text))
        return error_response(ErrorCode.INTERNAL_ERROR)
