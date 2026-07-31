"""Per-turn context threaded into tool handlers that need it (Issue #11).

`get_current_date` needs nothing beyond its own arguments, but
`save_visitor_info` and `flag_summary_conflict` need DB access scoped to the
current session, and calendar_find_slots (Issue #20) needs a calendar client
and the caller's timezone too — this is a single, reusable seam for that
rather than a bespoke parameter per tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.providers.base import LLMProvider
from app.tools.calendar import CalendarClient


@dataclass
class ToolContext:
    db: AsyncSession
    session_id: str
    # Used by flag_summary_conflict's reconciliation path (Engineering Guide
    # 4.3) — a separate role/cache-namespace from the main loop's provider,
    # per 4.3's "fully isolated call sites" rule, so it's threaded
    # separately rather than reusing whatever provider the main loop got.
    summarizer_provider: LLMProvider
    # calendar_find_slots (Issue #20) — injected the same way as
    # summarizer_provider, so tests can swap in a FakeCalendar the same way
    # they swap in a FakeProvider.
    calendar_client: CalendarClient
    # The visitor's IANA timezone, from ChatRequest.timezone (PRD: "explicit
    # API parameter in this phase" — no LLM extraction of freeform text like
    # "I'm in EST" in v1). None until the visitor's client sends it; the
    # booking flow's own guidance asks for it while it's still missing.
    caller_timezone: str | None = None
    # The visitor's IP address, from the request (app/api/chat.py's
    # http_request.client.host — same source the logging middleware
    # already uses). calendar_find_slots (Issue #23) uses this for the
    # per-IP booking-attempt rate limit; None if unavailable (e.g. no
    # client info on the request), in which case only the per-session cap
    # applies. Unlike Issue #24's general message rate limits, this
    # doesn't honor X-Forwarded-For — that's explicitly deferred to #24's
    # "trusted host proxy" handling, not duplicated here.
    #
    # Known interim gap (review finding, user-confirmed acceptable for
    # now): behind a reverse proxy (Render/Railway/Fly.io, per the Tech
    # Stack), `client.host` is typically the proxy's own address, not the
    # visitor's real IP — every visitor could share one bucket, or the cap
    # could be a no-op, depending on the platform's proxy behavior, until
    # #24 adds real client-IP resolution. The per-session cap is
    # unaffected either way (it never depends on IP).
    caller_ip: str | None = None
    # flag_summary_conflict (app/tools/registry.py) appends the model's
    # explanation here instead of running reconciliation synchronously —
    # reload_and_reconcile makes a real summarizer LLM call, and 4.3's
    # memory-bookkeeping design principle is that none of that ever adds
    # user-facing latency to the turn. app/api/chat.py reads this after
    # run_agent returns and schedules app/agent/memory.py's
    # run_reconciliation as a post-response background task per entry.
    pending_reconciliations: list[str] = field(default_factory=list)
    # Set by calendar_find_slots' handler the first time it runs, and
    # checked on every subsequent call within the same ToolContext (Issue
    # #20 review fix): a negotiation round is meant to correspond to one
    # visitor turn, but nothing else stops the model from emitting several
    # calendar_find_slots tool calls within a single response (or across
    # several run_agent iterations of the same turn) — each of which would
    # otherwise independently advance proposal_rounds, letting one turn
    # burn through the whole negotiation cap (2 rounds + 1 widen) without
    # the visitor ever having rejected a real proposal. One ToolContext is
    # constructed per HTTP request in app/api/chat.py and reused for the
    # entire run_agent call, so this correctly scopes "once per turn."
    calendar_find_slots_used_this_turn: bool = False
    # Set by app/api/chat.py from the deterministic classification of the
    # visitor's raw reply (app/booking/confirmation.py) — never left to the
    # model's own judgment of whether a reply "sounded like" a yes (4.5:
    # "Only an affirmative reply advances"). calendar_create_booking
    # (Issue #22) is gated to the confirmed step's tool list, but that's
    # step-level, not per-message: this is the message-level check the
    # handler itself enforces as defense in depth, same rationale as
    # calendar_find_slots' own state re-check.
    confirmation_is_affirmative: bool = False
    # calendar_create_booking (Issue #22) appends (subject, body) pairs here
    # instead of calling app/notify.py directly — mirrors
    # pending_reconciliations: notifying the owner is a side effect with no
    # bearing on this turn's reply, so app/api/chat.py schedules it as a
    # background task after run_agent returns, the same "never add
    # user-facing latency for a side effect the visitor isn't waiting on"
    # discipline as eviction/reconciliation (Engineering Guide 4.3).
    pending_owner_notifications: list[tuple[str, str]] = field(default_factory=list)
