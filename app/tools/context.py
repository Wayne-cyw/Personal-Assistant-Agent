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
    # flag_summary_conflict (app/tools/registry.py) appends the model's
    # explanation here instead of running reconciliation synchronously —
    # reload_and_reconcile makes a real summarizer LLM call, and 4.3's
    # memory-bookkeeping design principle is that none of that ever adds
    # user-facing latency to the turn. app/api/chat.py reads this after
    # run_agent returns and schedules app/agent/memory.py's
    # run_reconciliation as a post-response background task per entry.
    pending_reconciliations: list[str] = field(default_factory=list)
