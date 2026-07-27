from collections.abc import Generator

import pytest
from pydantic import BaseModel, field_validator

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


class _ArgsWithMisbehavingValidator(BaseModel):
    """Pydantic v2 only wraps ValueError/TypeError/AssertionError raised by a
    @field_validator into ValidationError — any other exception type
    propagates raw from model_validate(). Regression fixture for that gap.
    """

    x: int

    @field_validator("x")
    @classmethod
    def _check(cls, _value: int) -> int:
        raise RuntimeError("validator raised a non-ValueError exception")


async def _unused_handler(_args: BaseModel) -> dict[str, object]:
    return {"ok": True}


@pytest.fixture
def tool_with_misbehaving_validator() -> Generator[None]:
    registry_module._REGISTRY["test_tool_with_misbehaving_validator"] = _RegisteredTool(
        definition=ToolDef(
            name="test_tool_with_misbehaving_validator",
            description="test-only",
            parameters=_ArgsWithMisbehavingValidator.model_json_schema(),
        ),
        args_model=_ArgsWithMisbehavingValidator,
        handler=_unused_handler,
    )
    yield
    del registry_module._REGISTRY["test_tool_with_misbehaving_validator"]


async def test_execute_tool_validator_raising_non_value_error_returns_structured_error(
    tool_with_misbehaving_validator: None,
) -> None:
    call = ToolCall(
        id="call_1", name="test_tool_with_misbehaving_validator", arguments={"x": 1}
    )
    result = await execute_tool(call)  # must not raise
    assert "error" in result
    assert isinstance(result["error"], str)
    assert "test_tool_with_misbehaving_validator" in result["error"]


class _EmptyArgs(BaseModel):
    pass


async def _raising_handler(_args: BaseModel) -> dict[str, object]:
    raise RuntimeError("Authorization: Bearer sk-dummy-secret-value-77777")


@pytest.fixture
def tool_that_raises() -> Generator[None]:
    """A tool whose handler itself fails — not an argument-validation
    failure. Regression test for a handler-level exception (e.g. a future
    RAG/calendar tool's network call failing) reaching execute_tool.
    """
    registry_module._REGISTRY["test_tool_that_raises"] = _RegisteredTool(
        definition=ToolDef(
            name="test_tool_that_raises", description="test-only", parameters={}
        ),
        args_model=_EmptyArgs,
        handler=_raising_handler,
    )
    yield
    del registry_module._REGISTRY["test_tool_that_raises"]


async def test_execute_tool_handler_exception_returns_structured_error_not_raised(
    tool_that_raises: None,
) -> None:
    call = ToolCall(id="call_1", name="test_tool_that_raises", arguments={})
    result = await execute_tool(call)  # must not raise
    assert "error" in result
    assert isinstance(result["error"], str)
    assert "test_tool_that_raises" in result["error"]


async def test_execute_tool_handler_exception_never_leaks_secret_via_logging(
    tool_that_raises: None, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    caplog.set_level(logging.DEBUG)
    call = ToolCall(id="call_1", name="test_tool_that_raises", arguments={})

    result = await execute_tool(call)

    assert "sk-dummy-secret-value-77777" not in str(result)
    log_output = "\n".join(r.getMessage() for r in caplog.records)
    assert "sk-dummy-secret-value-77777" not in log_output
