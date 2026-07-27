"""Per-turn context threaded into tool handlers that need it (Issue #11).

`get_current_date` needs nothing beyond its own arguments, but
`save_visitor_info` and `flag_summary_conflict` need DB access scoped to the
current session, and future tools (rag_search's `kb_chunks` lookup, #15;
calendar tools, #17) will too — this is a single, reusable seam for that
rather than a bespoke parameter per tool.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.providers.base import LLMProvider


@dataclass
class ToolContext:
    db: AsyncSession
    session_id: str
    # Used by flag_summary_conflict's reconciliation path (Engineering Guide
    # 4.3) — a separate role/cache-namespace from the main loop's provider,
    # per 4.3's "fully isolated call sites" rule, so it's threaded
    # separately rather than reusing whatever provider the main loop got.
    summarizer_provider: LLMProvider
