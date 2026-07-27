"""Tool definitions, argument validation, and execute_tool dispatch
(Engineering Guide 4.2, Issue #10; save_visitor_info/flag_summary_conflict
added in Issue #11).

`get_current_date` is a trivial proof tool — real tools (RAG search,
calendar) register here the same way in later issues (#15, #17).
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ValidationError

from app.agent.providers.base import ToolCall, ToolDef
from app.agent.visitor_info import extract_linkedin_url
from app.config import settings
from app.db.session import add_pinned_fact, set_visitor_info
from app.safety.pii import redact
from app.tools.context import ToolContext

logger = logging.getLogger(__name__)

_NAME_MAX_LENGTH = 100
_FACT_MAX_LENGTH = 200


class GetCurrentDateArgs(BaseModel):
    """No arguments — the tool takes none."""


async def _get_current_date(_args: BaseModel, _context: ToolContext) -> dict[str, object]:
    now = datetime.now(UTC)
    return {"date": now.date().isoformat(), "iso": now.isoformat()}


class SaveVisitorInfoArgs(BaseModel):
    name: str | None = None
    linkedin: str | None = None
    fact: str | None = None


def _sanitize_short_text(value: str, max_length: int) -> str:
    # Replace (not delete) non-printable characters before collapsing
    # whitespace and clamping — a pinned-profile field is rendered verbatim
    # into every future prompt (Engineering Guide 4.6: visitor-volunteered
    # strings are data, and a multi-line/oddly-formatted value shouldn't be
    # able to smuggle extra structure into that block). Replacing with a
    # space rather than deleting matters: deleting a newline glues the
    # words on either side of it together ("no\nthing" -> "nothing"),
    # silently changing the text's meaning instead of just its formatting.
    cleaned = "".join(ch if ch.isprintable() else " " for ch in value)
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_length]


async def _save_visitor_info(args: BaseModel, context: ToolContext) -> dict[str, object]:
    assert isinstance(args, SaveVisitorInfoArgs)
    saved: dict[str, str] = {}
    rejected: dict[str, str] = {}

    name: str | None = None
    if args.name is not None:
        cleaned_name = _sanitize_short_text(args.name, _NAME_MAX_LENGTH)
        if cleaned_name:
            name = cleaned_name
            saved["name"] = cleaned_name
        else:
            rejected["name"] = "empty after sanitization"

    linkedin: str | None = None
    if args.linkedin is not None:
        matched = extract_linkedin_url(args.linkedin)
        if matched is not None:
            linkedin = matched
            saved["linkedin"] = matched
        else:
            rejected["linkedin"] = "not a recognizable linkedin.com/in/... URL"

    if name is not None or linkedin is not None:
        await set_visitor_info(context.db, context.session_id, name=name, linkedin=linkedin)

    if args.fact is not None:
        cleaned_fact = _sanitize_short_text(args.fact, _FACT_MAX_LENGTH)
        if not cleaned_fact:
            rejected["fact"] = "empty after sanitization"
        else:
            added = await add_pinned_fact(
                context.db, context.session_id, cleaned_fact, max_facts=settings.pinned_facts_max
            )
            if added:
                saved["fact"] = cleaned_fact
            else:
                rejected["fact"] = "duplicate, or the pinned-facts limit is already reached"

    return {"saved": saved, "rejected": rejected}


class FlagSummaryConflictArgs(BaseModel):
    explanation: str


async def _flag_summary_conflict(args: BaseModel, context: ToolContext) -> dict[str, object]:
    """Records the conflict for app/api/chat.py to schedule as a background
    reconciliation (app/agent/memory.py's run_reconciliation) rather than
    running reload_and_reconcile here — that makes a real summarizer LLM
    call, and reconciliation must not add user-facing latency to the turn
    any more than eviction does (Engineering Guide 4.3).
    """
    assert isinstance(args, FlagSummaryConflictArgs)
    context.pending_reconciliations.append(args.explanation)
    return {"acknowledged": True}


@dataclass
class _RegisteredTool:
    definition: ToolDef
    args_model: type[BaseModel]
    handler: Callable[[BaseModel, ToolContext], Awaitable[dict[str, object]]]
    # False for tools whose outcome is either already surfaced elsewhere in
    # context (save_visitor_info -> the pinned profile) or is a pure
    # meta/side-effect with nothing visitor-facing worth a token in later
    # turns (flag_summary_conflict) — see app/agent/memory.py's ToolEvent.
    persist_receipt: bool = True


GET_CURRENT_DATE = _RegisteredTool(
    definition=ToolDef(
        name="get_current_date",
        description="Returns the current date and time in UTC.",
        parameters=GetCurrentDateArgs.model_json_schema(),
    ),
    args_model=GetCurrentDateArgs,
    handler=_get_current_date,
)

SAVE_VISITOR_INFO = _RegisteredTool(
    definition=ToolDef(
        name="save_visitor_info",
        description=(
            "Call this when the visitor volunteers their name, LinkedIn URL, or a short fact "
            "about themselves (e.g. their role or what they're looking for) — even in passing. "
            "All arguments are optional; pass only what was volunteered this turn. Do not ask "
            "the visitor for this information proactively, just capture it if they offer it."
        ),
        parameters=SaveVisitorInfoArgs.model_json_schema(),
    ),
    args_model=SaveVisitorInfoArgs,
    handler=_save_visitor_info,
    persist_receipt=False,
)

FLAG_SUMMARY_CONFLICT = _RegisteredTool(
    definition=ToolDef(
        name="flag_summary_conflict",
        description=(
            "Call this if the conversation summary or visitor profile you were given "
            "contradicts what the visitor is telling you now (e.g. they explicitly correct or "
            "walk back something stated earlier). Briefly explain the contradiction. This "
            "triggers a background correction — it does not change your current answer, so "
            "still answer the visitor's message normally based on what they just told you."
        ),
        parameters=FlagSummaryConflictArgs.model_json_schema(),
    ),
    args_model=FlagSummaryConflictArgs,
    handler=_flag_summary_conflict,
    persist_receipt=False,
)

_REGISTRY: dict[str, _RegisteredTool] = {
    tool.definition.name: tool
    for tool in (GET_CURRENT_DATE, SAVE_VISITOR_INFO, FLAG_SUMMARY_CONFLICT)
}

TOOL_DEFS: list[ToolDef] = [tool.definition for tool in _REGISTRY.values()]


async def execute_tool(call: ToolCall, context: ToolContext) -> dict[str, object]:
    """Dispatch a tool call, validating arguments against its Pydantic
    schema. Invalid args (unknown tool, schema mismatch) *and* a failure
    inside the handler itself (a future tool making a real network call —
    RAG embeddings, calendar API — can fail for reasons unrelated to
    argument validation) both return a structured error dict *to the model
    as the tool result* — so it can self-correct — rather than raising;
    every failure is also logged server-side. Letting a handler exception
    escape here would turn one bad tool call into a 500 for the whole turn,
    rather than a self-correctable tool result (Engineering Guide 4.2: tool
    results are structured, in code, never inferred by letting the caller's
    exception handling improvise).
    """
    tool = _REGISTRY.get(call.name)
    if tool is None:
        logger.warning("unknown tool call: %s", call.name)
        return {"error": f"Unknown tool: {call.name}"}

    try:
        args = tool.args_model.model_validate(call.arguments)
    except ValidationError as exc:
        logger.warning("invalid arguments for tool %s: %s", call.name, exc)
        return {"error": f"Invalid arguments for {call.name}: {exc.errors()}"}
    except Exception as exc:
        # Pydantic v2 only wraps ValueError/TypeError/AssertionError raised
        # inside a @field_validator into ValidationError — any other
        # exception type a future tool's validator raises (e.g. a network
        # call inside a validator) would otherwise propagate raw and crash
        # the whole turn, exactly the failure class this function exists to
        # prevent. No current tool has a custom validator, but the registry
        # is generic, so this is guarded the same way handler failures are.
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.error(
            "tool %s argument validation raised unexpectedly\n%s", call.name, redact(tb_text)
        )
        return {"error": f"Invalid arguments for {call.name}."}

    try:
        return await tool.handler(args, context)
    except Exception as exc:
        # Sanitized the same way as app/api/errors.py's generic-exception
        # handler: a future tool making a real network call (calendar API,
        # embeddings) can raise an exception carrying credentials in a
        # header or connection string, so the traceback is redact()ed
        # before logging rather than passed through raw via
        # logger.exception(exc_info=exc).
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.error("tool %s raised during execution\n%s", call.name, redact(tb_text))
        return {"error": f"{call.name} failed to execute."}


def persist_receipt_for(tool_name: str) -> bool:
    """Whether a completed call to `tool_name` should be written to the
    messages log as a one-line receipt (Issue #11's window persistence) —
    used by app/agent/loop.py when building ToolEvents for persist_turn.
    Defaults True for an unknown name (fails toward keeping information,
    not silently dropping it).
    """
    tool = _REGISTRY.get(tool_name)
    return tool.persist_receipt if tool is not None else True
