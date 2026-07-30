"""The conversation memory system (Engineering Guide 4.3, Issue #11):
pinned visitor profile, token-budgeted rolling window with chunked
(high-water/low-water) eviction, a structured JSON summary created only at
first eviction, and reconciliation when a summary is found to be stale.

Context assembly is volatility-sorted, and the order is load-bearing —
**do not reorder it**: system prompt, then pinned profile + summary (rare
change), then the rolling window (append-only between evictions), then the
current message (changes every turn). Provider prompt caching bills
everything before the first changed byte at a steep discount (4.3), so an
"harmless" reorder silently forfeits that discount on every turn.

Booking-state contact/timezone fields belong in the pinned profile per 4.3,
but the booking state machine doesn't exist until Issue #18 — the profile
is assembled from visitor_name/visitor_linkedin/pinned_facts_json only until
then; the seam (`render_pinned_profile`) is where #18 adds them.

Reconciliation trigger 1 (explicit correction via the input classifier's
label set) is deferred: the classifier is Issue #25, which isn't a
dependency of #11 and isn't in this batch. Triggers 2 (model self-flagged
`summary_conflict`, via the `flag_summary_conflict` tool in
app/tools/registry.py) and 3 (the scheduled drift audit, below) are fully
implemented and both drive the same `reload_and_reconcile` used here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.prompts import SUMMARIZER_SYSTEM_PROMPT
from app.agent.providers.base import LLMProvider, UpstreamError, Usage
from app.agent.providers.base import Message as LLMMessage
from app.agent.session_lock import SessionBusyError, session_turn_lock
from app.agent.tokens import count_tokens, effective_tokens_from_usage
from app.config import settings
from app.db.models import Message as MessageRow
from app.db.models import SessionRow
from app.db.session import (
    add_token_budget_used,
    advance_summary,
    append_message,
    get_session_factory,
    increment_eviction_count,
    messages_after,
    messages_up_to,
    overwrite_summary_content,
)

logger = logging.getLogger(__name__)

_RECEIPT_MAX_LENGTH = 200
# Headroom above SUMMARY_MAX_TOKENS for JSON structural overhead (braces,
# quotes, field names) that isn't part of the field content itself.
_SUMMARIZER_MAX_TOKENS_FLOOR = 300


class SummaryJSON(BaseModel):
    visitor_context: str = ""
    open_questions: list[str] = []
    commitments: list[str] = []
    notes: list[str] = []


@dataclass
class ToolEvent:
    """One tool call executed during a turn's orchestration loop (Issue
    #10), as reported by run_agent — enough for persist_turn to write a
    receipt row. `persist_receipt=False` excludes purely side-effecting
    "meta" tools (save_visitor_info, flag_summary_conflict) whose outcome
    either already appears elsewhere in context (the pinned profile) or
    isn't visitor-facing content worth a token in future turns.
    """

    name: str
    result: dict[str, object]
    persist_receipt: bool = True


@dataclass
class Memory:
    pinned_profile_text: str
    summary_text: str
    window_messages: list[MessageRow] = field(default_factory=list)


def render_pinned_profile(session: SessionRow) -> str:
    """Compact structured block (~50 tokens), never evicted (4.3)."""
    lines: list[str] = []
    if session.visitor_name:
        lines.append(f"Name: {session.visitor_name}")
    if session.visitor_linkedin:
        lines.append(f"LinkedIn: {session.visitor_linkedin}")
    for fact in session.pinned_facts_json or []:
        lines.append(f"- {fact}")
    if not lines:
        return ""
    return "Visitor profile:\n" + "\n".join(lines)


def _render_summary(summary_json: dict[str, object] | None) -> str:
    if not summary_json:
        return ""
    try:
        summary = SummaryJSON.model_validate(summary_json)
    except ValidationError:
        logger.error("sessions.summary_json failed schema validation; omitting from context")
        return ""
    lines: list[str] = []
    if summary.visitor_context:
        lines.append(f"Context: {summary.visitor_context}")
    if summary.open_questions:
        lines.append("Open questions: " + "; ".join(summary.open_questions))
    if summary.commitments:
        lines.append("Commitments: " + "; ".join(summary.commitments))
    if summary.notes:
        lines.append("Notes: " + "; ".join(summary.notes))
    if not lines:
        return ""
    return "Conversation summary:\n" + "\n".join(lines)


async def load_memory(db: AsyncSession, session: SessionRow) -> Memory:
    window_rows = await messages_after(db, session.id, session.summary_through_message_id)
    return Memory(
        pinned_profile_text=render_pinned_profile(session),
        summary_text=_render_summary(session.summary_json),
        window_messages=window_rows,
    )


def _render_window_message(message: MessageRow) -> LLMMessage:
    if message.role == "tool":
        # One-line receipts only — the live loop's actual tool-call/result
        # protocol (with matching tool_call_id pairs) only ever exists
        # inside a single turn's run_agent invocation; replaying it across
        # turns after an eviction has stripped the middle out would break
        # OpenAI's strict pairing requirement. A synthetic assistant-role
        # note carries the outcome forward without that constraint.
        return LLMMessage(role="assistant", content=f"[{message.tool_name}: {message.content}]")
    role: str = message.role if message.role in ("user", "assistant") else "assistant"
    return LLMMessage(role=role, content=message.content)  # type: ignore[arg-type]


def assemble_messages(memory: Memory, system_prompt: str, current_message: str) -> list[LLMMessage]:
    messages = [LLMMessage(role="system", content=system_prompt)]

    pinned_and_summary = "\n\n".join(
        block for block in (memory.pinned_profile_text, memory.summary_text) if block
    )
    if pinned_and_summary:
        # Cache breakpoint (4.3): place it right after the pinned+summary
        # block, since everything up to here changes rarely while the
        # window below changes every turn. cache_breakpoint is a no-op for
        # OpenAI today (base.py — prefix caching is automatic), but marks
        # the intended boundary for a future provider that needs one
        # explicitly, without changing this function.
        messages.append(
            LLMMessage(role="system", content=pinned_and_summary, cache_breakpoint=True)
        )

    messages.extend(_render_window_message(m) for m in memory.window_messages)
    messages.append(LLMMessage(role="user", content=current_message))
    return messages


def _one_line_receipt(result: dict[str, object]) -> str:
    text = json.dumps(result, separators=(",", ":"), default=str)
    if len(text) <= _RECEIPT_MAX_LENGTH:
        return text
    return text[: _RECEIPT_MAX_LENGTH - 1] + "…"


async def persist_turn(
    db: AsyncSession,
    session_id: str,
    user_message: str,
    assistant_reply: str,
    tool_events: list[ToolEvent],
    *,
    user_already_persisted: bool = False,
) -> bool:
    """Append the turn to the messages log and run the eviction check.
    Returns True if the window has crossed WINDOW_HIGH_TOKENS and eviction
    should be scheduled — the actual summarization call runs separately
    (`run_eviction`), post-response, so this never adds LLM latency to the
    turn the caller is about to respond to.

    `user_already_persisted=True` skips the user-message append: 4.3's "the
    DB log is complete" invariant requires the user's message to survive
    even a provider failure, so app/api/chat.py appends it eagerly before
    calling the provider at all, rather than only here after a successful
    reply.
    """
    if not user_already_persisted:
        await append_message(db, session_id, "user", user_message)
    for event in tool_events:
        if not event.persist_receipt:
            continue
        await append_message(
            db,
            session_id,
            "tool",
            _one_line_receipt(event.result),
            tool_name=event.name,
            tool_payload=event.result,
        )
    await append_message(db, session_id, "assistant", assistant_reply)

    session = await db.get(SessionRow, session_id)
    assert session is not None, f"persist_turn called for unknown session_id={session_id!r}"
    window_rows = await messages_after(db, session_id, session.summary_through_message_id)
    window_tokens = count_tokens("\n".join(m.content for m in window_rows))
    return window_tokens > settings.window_high_tokens


def _split_for_eviction(
    window_rows: list[MessageRow],
) -> tuple[list[MessageRow], list[MessageRow]]:
    """Evict oldest-first until the *kept* remainder is at/under the
    low-water mark — one batch, not per-turn FIFO (4.3), so the window
    prefix stays append-only between evictions.
    """
    kept = list(window_rows)
    evicted: list[MessageRow] = []
    while kept and count_tokens("\n".join(m.content for m in kept)) > settings.window_low_tokens:
        evicted.append(kept.pop(0))
    return evicted, kept


def _parse_summary_json(text: str) -> SummaryJSON | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    try:
        return SummaryJSON.model_validate(data)
    except ValidationError:
        return None


_ZERO_USAGE = Usage(input_tokens=0, output_tokens=0, cached_input_tokens=0)


def _add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cached_input_tokens=a.cached_input_tokens + b.cached_input_tokens,
    )


async def _summarize(
    provider: LLMProvider, existing_summary: SummaryJSON | None, turns: list[MessageRow]
) -> tuple[SummaryJSON | None, Usage]:
    """Merge `turns` into `existing_summary` (or start fresh if None — used
    both for a normal eviction fold and for reload_and_reconcile's
    from-scratch regeneration). Retries once on failure, then skips
    gracefully (4.3): the evicted turns remain in the DB log regardless, so
    nothing is lost, only the compressed context update is deferred to the
    next eviction.

    Returns real usage across every attempt made (even a failed/retried one
    still spent real tokens) so the caller can charge it to the session's
    token budget (Issue #12) regardless of whether the fold succeeded.
    """
    existing_text = existing_summary.model_dump_json() if existing_summary else "none"
    turns_text = "\n".join(f"{m.role}: {m.content}" for m in turns)
    user_content = f"Existing summary:\n{existing_text}\n\nNew turns to fold in:\n{turns_text}"
    messages = [
        LLMMessage(role="system", content=SUMMARIZER_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_content),
    ]
    # MAX_TOKENS_PER_TURN is enforced as the outer ceiling on every provider
    # call (4.3/Issue #12), including the summarizer's own, tighter budget.
    max_tokens = min(
        max(_SUMMARIZER_MAX_TOKENS_FLOOR, settings.summary_max_tokens * 2),
        settings.max_tokens_per_turn,
    )
    total_usage = _ZERO_USAGE

    for attempt in range(2):
        try:
            response = await provider.complete(messages=messages, tools=[], max_tokens=max_tokens)
        except UpstreamError:
            logger.warning("summarizer call failed (attempt %d/2)", attempt + 1)
            continue
        total_usage = _add_usage(total_usage, response.usage)
        parsed = _parse_summary_json(response.text)
        if parsed is not None:
            return parsed, total_usage
        logger.warning(
            "summarizer produced invalid JSON (attempt %d/2): %r", attempt + 1, response.text
        )
    logger.error("summarizer failed twice; skipping this fold (evicted turns remain in the DB log)")
    return None, total_usage


def _memory_lock_key(session_id: str) -> str:
    """A namespaced lock key distinct from app/api/chat.py's own
    `session_turn_lock(session_id)` hold during live-turn processing.

    Background memory bookkeeping (eviction, reconciliation) only needs to
    be serialized against *itself* — two such operations for the same
    session both write summary_json/summary_through_message_id and can
    race each other — it does not need to be serialized against live-turn
    processing, since a live turn never writes those fields. Sharing the
    live-turn lock key would work for the eviction-vs-eviction race too,
    but at the cost of blocking (and, past the wait window, 429-ing) a
    visitor's next message for as long as a background summarizer call is
    in flight — directly contradicting 4.3's "memory bookkeeping adds zero
    user-facing latency" guarantee. Namespacing the key avoids that
    entirely while still closing the original race.

    Note for Issue #18 (booking state machine): 4.3's concurrency-guard
    description reads as one shared lock domain covering "the booking
    state machine as well as memory bookkeeping." Whoever builds #18
    should follow this same split-by-namespace precedent for any
    background booking work rather than reusing the bare session_id key —
    otherwise it either reintroduces this latency problem (if it shares
    the live-turn key) or silently fails to serialize against memory
    bookkeeping (if it invents a third, uncoordinated key).
    """
    return f"mem:{session_id}"


async def run_eviction(session_id: str, summarizer_provider: LLMProvider) -> None:
    """The actual eviction + summarization work, scheduled by
    app/api/chat.py as a FastAPI background task so it runs post-response
    (4.3: "adding zero user-facing latency"). Opens its own DB session —
    the request-scoped one is closed by the time a background task runs.

    Runs under a namespaced per-session lock (see _memory_lock_key) so two
    eviction runs racing on the same session's summary_through_message_id
    boundary can't regress it (a later, larger eviction's advance getting
    overwritten by an earlier, smaller one that reads stale state and
    commits after) — silently re-exposing already-summarized turns in the
    live window and violating 4.3's "each turn still summarized at most
    once, ever" guarantee — without contending with live-turn processing.

    If the lock is still held (SessionBusyError) after the wait window —
    e.g. another eviction or a reconciliation is already in flight for this
    session — this skips gracefully rather than raising out of the
    background task: the window just stays over budget until the next
    turn's persist_turn re-triggers eviction, which is a cost/latency
    concern, not a correctness one (unlike run_reconciliation's skip case
    below, eviction's trigger condition is recomputed from durable state
    every turn, so a skipped run is never silently lost).
    """
    try:
        async with session_turn_lock(_memory_lock_key(session_id)):
            await _run_eviction_locked(session_id, summarizer_provider)
    except SessionBusyError:
        logger.warning(
            "eviction for session %s skipped: session busy, will retry next turn", session_id
        )


async def _run_eviction_locked(session_id: str, summarizer_provider: LLMProvider) -> None:
    async with get_session_factory()() as db:
        session = await db.get(SessionRow, session_id)
        if session is None:
            return
        window_rows = await messages_after(db, session_id, session.summary_through_message_id)
        evicted, _kept = _split_for_eviction(window_rows)
        if not evicted:
            return  # nothing to evict (e.g. a concurrent turn already ran this)

        existing_summary = (
            SummaryJSON.model_validate(session.summary_json) if session.summary_json else None
        )
        new_summary, usage = await _summarize(summarizer_provider, existing_summary, evicted)
        await add_token_budget_used(db, session_id, effective_tokens_from_usage(usage))
        if new_summary is None:
            return

        await advance_summary(
            db,
            session_id,
            summary_json=new_summary.model_dump(),
            summary_through_message_id=evicted[-1].id,
        )
        new_count = await increment_eviction_count(db, session_id)

        if new_count % settings.summary_audit_interval == 0:
            await _run_drift_audit(db, session_id, summarizer_provider)


async def _run_drift_audit(
    db: AsyncSession, session_id: str, summarizer_provider: LLMProvider
) -> None:
    """Trigger 3 (4.3): every SUMMARY_AUDIT_INTERVAL evictions, regenerate
    the summary from scratch off the full raw range and log a diff against
    the incrementally-merged version. A canary, not an auto-fix — never
    overwrites summary_json.
    """
    session = await db.get(SessionRow, session_id)
    if session is None or session.summary_through_message_id is None:
        return
    raw_messages = await messages_up_to(db, session_id, session.summary_through_message_id)
    from_scratch, usage = await _summarize(
        summarizer_provider, existing_summary=None, turns=raw_messages
    )
    await add_token_budget_used(db, session_id, effective_tokens_from_usage(usage))
    if from_scratch is None:
        return
    incremental = session.summary_json
    if from_scratch.model_dump() != incremental:
        logger.warning(
            "drift audit: incremental summary diverges from a from-scratch regeneration "
            "for session %s\nincremental=%s\nfrom_scratch=%s",
            session_id,
            incremental,
            from_scratch.model_dump(),
        )
    else:
        logger.info("drift audit: no divergence for session %s", session_id)


async def reload_and_reconcile(
    db: AsyncSession,
    session_id: str,
    summarizer_provider: LLMProvider,
    *,
    trigger: str,
    detail: str = "",
) -> SummaryJSON | None:
    """Triggers 1 (deferred, see module docstring) and 2 (model
    self-flagged, via flag_summary_conflict) both call this (4.3): fetch the
    raw messages for the range the summary claims to cover, regenerate it
    from source, and overwrite the summary's *content* in place —
    summary_through_message_id is unchanged, since no new coverage was
    added, the existing coverage was corrected. Every reconciliation is
    logged.
    """
    session = await db.get(SessionRow, session_id)
    if session is None or session.summary_through_message_id is None:
        return None  # nothing summarized yet, nothing to reconcile
    raw_messages = await messages_up_to(db, session_id, session.summary_through_message_id)
    new_summary, usage = await _summarize(
        summarizer_provider, existing_summary=None, turns=raw_messages
    )
    await add_token_budget_used(db, session_id, effective_tokens_from_usage(usage))
    if new_summary is None:
        return None
    before = session.summary_json
    await overwrite_summary_content(db, session_id, summary_json=new_summary.model_dump())
    logger.warning(
        "summary reconciled for session %s (trigger=%s, detail=%r)\nbefore=%s\nafter=%s",
        session_id,
        trigger,
        detail,
        before,
        new_summary.model_dump(),
    )
    return new_summary


async def run_reconciliation(
    session_id: str, summarizer_provider: LLMProvider, *, trigger: str, detail: str = ""
) -> None:
    """Background-task entry point for reload_and_reconcile (Issue #11),
    scheduled by app/api/chat.py for each of ToolContext.
    pending_reconciliations after run_agent returns — mirrors run_eviction:
    own DB session (the request-scoped one is gone by the time this runs),
    the same namespaced per-session lock (see _memory_lock_key —
    reconciliation also mutates summary_json, so it needs the same race
    protection eviction does, against eviction as well as against another
    reconciliation), same graceful skip if the lock is still held after the
    wait window.

    Unlike eviction, a skipped reconciliation here is *not* automatically
    retried: there is no durable "reconciliation pending" state anywhere,
    so if this is skipped, the visitor's flagged correction is simply not
    applied unless the model happens to call flag_summary_conflict again on
    a later turn. Accepted as a rare-contention corner case rather than
    building a durable retry queue for what is a self-correction layer on
    top of an already-correct incremental summary, not the source of truth
    (the verbatim messages log always is) — but worth flagging explicitly
    rather than leaving the old, inaccurate "will retry" log message.
    """
    try:
        async with session_turn_lock(_memory_lock_key(session_id)):
            async with get_session_factory()() as db:
                await reload_and_reconcile(
                    db, session_id, summarizer_provider, trigger=trigger, detail=detail
                )
    except SessionBusyError:
        logger.warning(
            "reconciliation for session %s skipped (session busy): the flagged correction "
            "was not applied and will not be retried automatically",
            session_id,
        )
