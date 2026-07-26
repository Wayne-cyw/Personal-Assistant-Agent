"""Structured JSON-line request logging middleware (Engineering Guide 4.3,
PRD Section 2's observability backbone).

By construction, this middleware only ever receives already-sanitized
fields: message *length* is logged nowhere here at all (no message content
passes through this module), and token counts come from `LLMResponse.usage`
(Issue #4), not raw prompts. `redact()` (app/safety/pii.py) is the
defense-in-depth safety net behind that.

Implemented as raw ASGI middleware, not `BaseHTTPMiddleware`: Starlette's
`BaseHTTPMiddleware.call_next()` re-raises the original exception past any
wrapping middleware even after an inner exception handler already produced
and sent the response (a documented Starlette quirk), which would silently
drop the log line for exactly the requests this middleware most needs to
observe.

Capturing the status via a wrapped `send` alone is not sufficient either:
Starlette's `build_middleware_stack()` pulls the generic-`Exception` handler
out specially and binds it to `ServerErrorMiddleware`, which is always the
absolute outermost layer — outside every user middleware, including this
one. A truly unhandled exception therefore propagates *past* this
middleware's `send` wrapper entirely and is turned into a response further
out, using a `send` this middleware never sees. The `except Exception` below
records status 500 itself (the only outcome `ServerErrorMiddleware` ever
produces here, per the registered handler in app/api/errors.py) before
re-raising, so the log line is still written and still accurate.
"""

from __future__ import annotations

import json
import logging
import time

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import settings
from app.safety.pii import redact

request_logger = logging.getLogger("app.request")
request_logger.setLevel(settings.log_level)
if not request_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    request_logger.addHandler(_handler)
    request_logger.propagate = False


class LoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        start = time.monotonic()
        status_code: int | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            status_code = 500
            raise
        finally:
            latency_ms = round((time.monotonic() - start) * 1000, 2)
            entry = {
                "ts": time.time(),
                "session_id": getattr(request.state, "session_id", None),
                "ip": request.client.host if request.client else None,
                "path": request.url.path,
                "status": status_code,
                "latency_ms": latency_ms,
                "llm_tokens_in": getattr(request.state, "llm_tokens_in", None),
                "llm_tokens_out": getattr(request.state, "llm_tokens_out", None),
                "tool_name": getattr(request.state, "tool_name", None),
                "turn_tag": getattr(request.state, "turn_tag", None),
            }
            # redact() wraps this log sink (Issue #6): a safety net in case
            # a future field ever carries something key-shaped, even though
            # every field above is already sanitized by construction (no
            # message content or raw prompts pass through this module).
            request_logger.info(redact(json.dumps(entry)))
