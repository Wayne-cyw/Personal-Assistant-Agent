from collections.abc import Generator

import pytest
from pydantic import BaseModel

import app.tools.registry as registry_module
from app.agent.providers.base import ToolCall, ToolDef
from app.tools.registry import GET_CURRENT_DATE, TOOL_DEFS, _RegisteredTool, execute_tool


def test_get_current_date_is_registered() -> None:
    names = [t.name for t in TOOL_DEFS]
    assert "get_current_date" in names


def test_tool_def_has_description_and_schema() -> None:
    assert GET_CURRENT_DATE.definition.description
    assert isinstance(GET_CURRENT_DATE.definition.parameters, dict)


async def test_execute_tool_get_current_date_returns_iso_date() -> None:
    call = ToolCall(id="call_1", name="get_current_date", arguments={})
    result = await execute_tool(call)
    assert "date" in result
    assert "iso" in result
    # YYYY-MM-DD
    date_str = result["date"]
    assert isinstance(date_str, str)
    assert len(date_str) == 10
    assert date_str[4] == "-" and date_str[7] == "-"


async def test_execute_tool_unknown_tool_returns_structured_error() -> None:
    """Invalid tool name surfaces as a tool-result error, not an exception."""
    call = ToolCall(id="call_1", name="not_a_real_tool", arguments={})
    result = await execute_tool(call)
    assert "error" in result
    assert isinstance(result["error"], str)
    assert "not_a_real_tool" in result["error"]


async def test_execute_tool_extra_args_on_zero_arg_tool_are_ignored_not_fatal() -> None:
    call = ToolCall(id="call_1", name="get_current_date", arguments={"unexpected": "value"})
    result = await execute_tool(call)
    assert "date" in result  # succeeds; extra args ignored, not fatal


class _RequiredArgs(BaseModel):
    required_field: int


async def _handler(_args: BaseModel) -> dict[str, object]:
    return {"ok": True}


@pytest.fixture
def tool_with_required_args() -> Generator[None]:
    """Registers a temporary test-only tool with a required field, so
    "invalid args -> structured error, not an exception" can be exercised
    genuinely — get_current_date has no args, so it can't trigger this path.
    """
    registry_module._REGISTRY["test_tool_with_required_args"] = _RegisteredTool(
        definition=ToolDef(
            name="test_tool_with_required_args",
            description="test-only",
            parameters=_RequiredArgs.model_json_schema(),
        ),
        args_model=_RequiredArgs,
        handler=_handler,
    )
    yield
    del registry_module._REGISTRY["test_tool_with_required_args"]


async def test_execute_tool_missing_required_arg_returns_structured_error_not_exception(
    tool_with_required_args: None,
) -> None:
    call = ToolCall(id="call_1", name="test_tool_with_required_args", arguments={})
    result = await execute_tool(call)  # must not raise
    assert "error" in result
    assert isinstance(result["error"], str)
    assert "test_tool_with_required_args" in result["error"]


async def test_execute_tool_wrong_type_arg_returns_structured_error_not_exception(
    tool_with_required_args: None,
) -> None:
    call = ToolCall(
        id="call_1",
        name="test_tool_with_required_args",
        arguments={"required_field": "not-an-int"},
    )
    result = await execute_tool(call)  # must not raise
    assert "error" in result
