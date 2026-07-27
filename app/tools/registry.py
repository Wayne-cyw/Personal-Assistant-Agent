"""Tool definitions, argument validation, and execute_tool dispatch
(Engineering Guide 4.2, Issue #10).

`get_current_date` is a trivial proof tool — real tools (RAG search,
calendar) register here the same way in later issues (#15, #17).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ValidationError

from app.agent.providers.base import ToolCall, ToolDef

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
    schema. Invalid args (unknown tool, schema mismatch) return a
    structured error dict *to the model as the tool result* — so it can
    self-correct — rather than raising; the failure is also logged
    server-side.
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

    return await tool.handler(args)
