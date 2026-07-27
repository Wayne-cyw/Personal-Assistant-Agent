"""Tool definitions, argument validation, and execute_tool dispatch
(Engineering Guide 4.2, Issue #10).

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
from app.safety.pii import redact

logger = logging.getLogger(__name__)


class GetCurrentDateArgs(BaseModel):
    """No arguments — the tool takes none."""


async def _get_current_date(_args: BaseModel) -> dict[str, object]:
    now = datetime.now(UTC)
    return {"date": now.date().isoformat(), "iso": now.isoformat()}


@dataclass
class _RegisteredTool:
    definition: ToolDef
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], Awaitable[dict[str, object]]]


GET_CURRENT_DATE = _RegisteredTool(
    definition=ToolDef(
        name="get_current_date",
        description="Returns the current date and time in UTC.",
        parameters=GetCurrentDateArgs.model_json_schema(),
    ),
    args_model=GetCurrentDateArgs,
    handler=_get_current_date,
)

_REGISTRY: dict[str, _RegisteredTool] = {GET_CURRENT_DATE.definition.name: GET_CURRENT_DATE}

TOOL_DEFS: list[ToolDef] = [tool.definition for tool in _REGISTRY.values()]


async def execute_tool(call: ToolCall) -> dict[str, object]:
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
        return await tool.handler(args)
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
