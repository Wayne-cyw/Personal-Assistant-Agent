"""Per-session and per-IP message rate limiting middleware (Issue #24,
Engineering Guide 4.1: "[Middleware] Rate limit check (per session_id AND
per IP) -> over limit -> 429"). Sits in front of every /v1/chat request,
before the handler, before the LLM is ever called — Issue #23's stricter
booking-attempt caps sit *underneath* this, checked later inside the
calendar_find_slots tool call itself, once a booking flow is already in
progress.

Implemented as raw ASGI middleware (matching app/middleware/logging.py's
own reasoning for avoiding BaseHTTPMiddleware) rather than a FastAPI
dependency: a dependency only runs *after* routing has already resolved
the endpoint and validated the request body against its Pydantic model,
which is too late for the acceptance criteria's "zero LLM spend for
blocked requests" — this needs to intercept before FastAPI's own routing/
validation machinery does anything, not just before the handler body runs.

session_id lives in the POST body (ChatRequest, Issue #5), not a header or
URL param, and an ASGI request body is a single-consume stream — this
middleware reads it once to extract session_id, then replays the exact
same bytes to the downstream app via a wrapped `receive`, so /v1/chat's
own handler can still parse ChatRequest normally afterward.
"""

from __future__ import annotations

import json
from datetime import timedelta

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import ErrorCode, error_response
from app.config import settings
from app.db.session import check_and_increment_rate_limit, get_session_factory

_RATE_LIMITED_PATH = "/v1/chat"
_WINDOW = timedelta(minutes=1)


def _client_ip(request: Request) -> str | None:
    """The visitor's IP for the per-IP counter — X-Forwarded-For only
    when settings.trust_x_forwarded_for is set (Issue #24: "honor
    X-Forwarded-For only from the trusted host proxy"; see that setting's
    own docstring in app/config.py for why this is a deploy-time on/off
    switch rather than a per-hop IP allowlist).

    Deliberately the *last* entry, not the first (review finding —
    caught before merge): the header is client-suppliable, and a proxy
    that appends rather than replaces it (the common case — e.g. nginx's
    `$proxy_add_x_forwarded_for`) puts each hop's own observed peer
    address at the *end* of the chain, appending to whatever the client
    already sent. With exactly one trusted hop between the client and
    this app (this deploy model's single edge proxy, per the setting's
    own docstring), the last entry is the one value in the header that
    hop itself actually observed and added — everything to its left,
    including the "first" entry, is attacker-controlled: a client can
    freely prepend `X-Forwarded-For: 1.2.3.4` (or a fresh spoofed value
    per request) before the request ever reaches the trusted proxy.
    Trusting the first entry would let that spoofing defeat the per-IP
    cap entirely, in exactly the deployment configuration where this
    setting is turned on.
    """
    if settings.trust_x_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            last = forwarded.split(",")[-1].strip()
            if last:
                return last
    return request.client.host if request.client else None


async def _read_body(receive: Receive) -> bytes:
    # No hard size cap here (review finding, left as a documented gap
    # rather than a fix bundled into this change): nothing else in this
    # codebase enforces a request-body size limit either, and truncating
    # here would corrupt what gets replayed to the real handler below —
    # a genuine cap would need to reject the request outright instead of
    # silently reading a prefix, which is a bigger design decision than
    # this middleware's own scope (Issue #24 is rate limiting, not body
    # size enforcement).
    chunks: list[bytes] = []
    more_body = True
    while more_body:
        message = await receive()
        if message["type"] == "http.disconnect":
            # The client went away mid-body -- there is nothing more to
            # read, and no downstream app will run for this connection
            # either way, so stop rather than looping on more receive()
            # calls that would never produce a "more_body": False message.
            break
        chunks.append(message.get("body", b""))
        more_body = message.get("more_body", False)
    return b"".join(chunks)


def _extract_session_id(body: bytes) -> str | None:
    # Deliberately tolerant of malformed/missing JSON here (mirrors Issue
    # #23's "if info is missing, skip that specific check rather than
    # blocking" graceful-degradation pattern) — a malformed body still
    # gets a proper 422 from ChatRequest's own validation downstream, and
    # the per-IP counter (which needs no body parsing) still applies
    # regardless, so nothing is bypassed by a request this can't parse.
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    session_id = payload.get("session_id")
    return session_id if isinstance(session_id, str) and session_id else None


def _replay_receive(body: bytes) -> Receive:
    """A fresh `receive` callable that replays `body` exactly once, for
    the downstream app to consume — this middleware already fully drained
    the real `receive` while reading the body itself.
    """
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        # A well-behaved ASGI client doesn't call receive() again after
        # more_body=False; this is just a safe terminal fallback if one
        # does, mirroring Starlette's own disconnect-sentinel convention.
        return {"type": "http.disconnect"}

    return receive


class RateLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # method check matters (review finding): without it, a GET/OPTIONS/
        # etc. to this path would still have its body buffered and consume
        # a rate-limit slot before falling through to routing's own 405,
        # instead of just 405ing immediately like every other wrong-method
        # request on this path already does.
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != _RATE_LIMITED_PATH
        ):
            await self.app(scope, receive, send)
            return

        body = await _read_body(receive)
        session_id = _extract_session_id(body)
        request = Request(scope)
        ip = _client_ip(request)

        # Reads app.state rather than calling get_session_factory()
        # directly: this is raw ASGI middleware, outside FastAPI's
        # dependency-injection tree, so app.dependency_overrides (what
        # every test's client fixture uses to point get_db at an isolated
        # test engine) has no effect here. app.main's lifespan sets
        # app.state.db_session_factory for real deployments; a test
        # fixture sets the same attribute to its own test session_factory
        # the same way it already overrides get_db. Falling back to the
        # process-wide get_session_factory() when unset keeps this
        # correct even if lifespan never ran (e.g. a host that doesn't
        # invoke ASGI lifespan events).
        session_factory = getattr(request.app.state, "db_session_factory", None)
        if session_factory is None:
            session_factory = get_session_factory()

        allowed = True
        async with session_factory() as db:
            if session_id is not None:
                session_allowed = await check_and_increment_rate_limit(
                    db,
                    f"sess:{session_id}",
                    limit=settings.messages_per_session_per_min,
                    window=_WINDOW,
                )
                allowed = allowed and session_allowed
            if ip is not None:
                ip_allowed = await check_and_increment_rate_limit(
                    db,
                    f"ip:{ip}",
                    limit=settings.messages_per_ip_per_min,
                    window=_WINDOW,
                )
                allowed = allowed and ip_allowed

        if not allowed:
            response = error_response(ErrorCode.RATE_LIMITED)
            await response(scope, receive, send)
            return

        await self.app(scope, _replay_receive(body), send)
