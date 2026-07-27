from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import app.tools.registry as registry_module
from app.agent.providers.base import LLMResponse, ToolCall, ToolDef, Usage
from app.agent.providers.fake import FakeProvider
from app.db.models import Base
from app.db.session import get_or_create_session
from app.tools.context import ToolContext
from app.tools.registry import (
    FLAG_SUMMARY_CONFLICT,
    GET_CURRENT_DATE,
    SAVE_VISITOR_INFO,
    TOOL_DEFS,
    _RegisteredTool,
    execute_tool,
    persist_receipt_for,
)


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine]:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db(engine: AsyncEngine) -> AsyncGenerator[AsyncSession]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        await get_or_create_session(session, "sess-1")
        yield session


@pytest.fixture
def summarizer() -> FakeProvider:
    return FakeProvider(responses=[])


@pytest.fixture
def context(db: AsyncSession, summarizer: FakeProvider) -> ToolContext:
    return ToolContext(db=db, session_id="sess-1", summarizer_provider=summarizer)


def test_get_current_date_is_registered() -> None:
    names = [t.name for t in TOOL_DEFS]
    assert "get_current_date" in names


def test_save_visitor_info_and_flag_summary_conflict_are_registered() -> None:
    names = [t.name for t in TOOL_DEFS]
    assert "save_visitor_info" in names
    assert "flag_summary_conflict" in names


def test_tool_def_has_description_and_schema() -> None:
    assert GET_CURRENT_DATE.definition.description
    assert isinstance(GET_CURRENT_DATE.definition.parameters, dict)


async def test_execute_tool_get_current_date_returns_iso_date(context: ToolContext) -> None:
    call = ToolCall(id="call_1", name="get_current_date", arguments={})
    result = await execute_tool(call, context)
    assert "date" in result
    assert "iso" in result
    # YYYY-MM-DD
    date_str = result["date"]
    assert isinstance(date_str, str)
    assert len(date_str) == 10
    assert date_str[4] == "-" and date_str[7] == "-"


async def test_execute_tool_unknown_tool_returns_structured_error(context: ToolContext) -> None:
    """Invalid tool name surfaces as a tool-result error, not an exception."""
    call = ToolCall(id="call_1", name="not_a_real_tool", arguments={})
    result = await execute_tool(call, context)
    assert "error" in result
    assert isinstance(result["error"], str)
    assert "not_a_real_tool" in result["error"]


async def test_execute_tool_extra_args_on_zero_arg_tool_are_ignored_not_fatal(
    context: ToolContext,
) -> None:
    call = ToolCall(id="call_1", name="get_current_date", arguments={"unexpected": "value"})
    result = await execute_tool(call, context)
    assert "date" in result  # succeeds; extra args ignored, not fatal


class _RequiredArgs(BaseModel):
    required_field: int


async def _handler(_args: BaseModel, _context: ToolContext) -> dict[str, object]:
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
    tool_with_required_args: None, context: ToolContext
) -> None:
    call = ToolCall(id="call_1", name="test_tool_with_required_args", arguments={})
    result = await execute_tool(call, context)  # must not raise
    assert "error" in result
    assert isinstance(result["error"], str)
    assert "test_tool_with_required_args" in result["error"]


async def test_execute_tool_wrong_type_arg_returns_structured_error_not_exception(
    tool_with_required_args: None, context: ToolContext
) -> None:
    call = ToolCall(
        id="call_1",
        name="test_tool_with_required_args",
        arguments={"required_field": "not-an-int"},
    )
    result = await execute_tool(call, context)  # must not raise
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


async def _unused_handler(_args: BaseModel, _context: ToolContext) -> dict[str, object]:
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
    tool_with_misbehaving_validator: None, context: ToolContext
) -> None:
    call = ToolCall(id="call_1", name="test_tool_with_misbehaving_validator", arguments={"x": 1})
    result = await execute_tool(call, context)  # must not raise
    assert "error" in result
    assert isinstance(result["error"], str)
    assert "test_tool_with_misbehaving_validator" in result["error"]


class _EmptyArgs(BaseModel):
    pass


async def _raising_handler(_args: BaseModel, _context: ToolContext) -> dict[str, object]:
    raise RuntimeError("Authorization: Bearer sk-dummy-secret-value-77777")


@pytest.fixture
def tool_that_raises() -> Generator[None]:
    """A tool whose handler itself fails — not an argument-validation
    failure. Regression test for a handler-level exception (e.g. a future
    RAG/calendar tool's network call failing) reaching execute_tool.
    """
    registry_module._REGISTRY["test_tool_that_raises"] = _RegisteredTool(
        definition=ToolDef(name="test_tool_that_raises", description="test-only", parameters={}),
        args_model=_EmptyArgs,
        handler=_raising_handler,
    )
    yield
    del registry_module._REGISTRY["test_tool_that_raises"]


async def test_execute_tool_handler_exception_returns_structured_error_not_raised(
    tool_that_raises: None, context: ToolContext
) -> None:
    call = ToolCall(id="call_1", name="test_tool_that_raises", arguments={})
    result = await execute_tool(call, context)  # must not raise
    assert "error" in result
    assert isinstance(result["error"], str)
    assert "test_tool_that_raises" in result["error"]


async def test_execute_tool_handler_exception_never_leaks_secret_via_logging(
    tool_that_raises: None, context: ToolContext, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    caplog.set_level(logging.DEBUG)
    call = ToolCall(id="call_1", name="test_tool_that_raises", arguments={})

    result = await execute_tool(call, context)

    assert "sk-dummy-secret-value-77777" not in str(result)
    log_output = "\n".join(r.getMessage() for r in caplog.records)
    assert "sk-dummy-secret-value-77777" not in log_output


# --- save_visitor_info (Issue #11) ------------------------------------------


async def test_save_visitor_info_persists_name(context: ToolContext) -> None:
    call = ToolCall(id="call_1", name="save_visitor_info", arguments={"name": "Priya Patel"})
    result = await execute_tool(call, context)
    assert result["saved"] == {"name": "Priya Patel"}

    session = await get_or_create_session(context.db, "sess-1")
    assert session.visitor_name == "Priya Patel"


async def test_save_visitor_info_persists_valid_linkedin(context: ToolContext) -> None:
    call = ToolCall(
        id="call_1",
        name="save_visitor_info",
        arguments={"linkedin": "https://www.linkedin.com/in/priya-example"},
    )
    result = await execute_tool(call, context)
    assert result["saved"] == {"linkedin": "https://www.linkedin.com/in/priya-example"}

    session = await get_or_create_session(context.db, "sess-1")
    assert session.visitor_linkedin == "https://www.linkedin.com/in/priya-example"


async def test_save_visitor_info_rejects_invalid_linkedin(context: ToolContext) -> None:
    call = ToolCall(
        id="call_1", name="save_visitor_info", arguments={"linkedin": "not-a-linkedin-url"}
    )
    result = await execute_tool(call, context)
    assert result["saved"] == {}
    assert "linkedin" in result["rejected"]  # type: ignore[operator]

    session = await get_or_create_session(context.db, "sess-1")
    assert session.visitor_linkedin is None


async def test_save_visitor_info_persists_fact_and_overwrites_name(context: ToolContext) -> None:
    """Regression test: unlike Issue #9's regex-based turn-zero capture-once
    semantics, a deliberate tool call is a visitor correction and must
    overwrite, not be silently dropped.
    """
    await execute_tool(
        ToolCall(id="call_1", name="save_visitor_info", arguments={"name": "Priya"}), context
    )
    result = await execute_tool(
        ToolCall(
            id="call_2",
            name="save_visitor_info",
            arguments={"name": "Priya Patel", "fact": "hiring for a backend role"},
        ),
        context,
    )
    assert result["saved"] == {"name": "Priya Patel", "fact": "hiring for a backend role"}

    session = await get_or_create_session(context.db, "sess-1")
    assert session.visitor_name == "Priya Patel"
    assert session.pinned_facts_json == ["hiring for a backend role"]


async def test_save_visitor_info_fact_cap_reports_rejection(
    context: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "pinned_facts_max", 1)
    await execute_tool(
        ToolCall(id="call_1", name="save_visitor_info", arguments={"fact": "likes Rust"}), context
    )
    result = await execute_tool(
        ToolCall(id="call_2", name="save_visitor_info", arguments={"fact": "likes Go"}), context
    )
    assert result["rejected"] == {"fact": "duplicate, or the pinned-facts limit is already reached"}


async def test_save_visitor_info_no_args_is_a_harmless_noop(context: ToolContext) -> None:
    call = ToolCall(id="call_1", name="save_visitor_info", arguments={})
    result = await execute_tool(call, context)
    assert result == {"saved": {}, "rejected": {}}


# --- flag_summary_conflict (Issue #11) --------------------------------------


async def test_flag_summary_conflict_no_existing_summary_reports_not_reconciled(
    context: ToolContext,
) -> None:
    call = ToolCall(
        id="call_1", name="flag_summary_conflict", arguments={"explanation": "no summary yet"}
    )
    result = await execute_tool(call, context)
    assert result == {"reconciled": False}


async def test_flag_summary_conflict_triggers_reconciliation(db: AsyncSession) -> None:
    from app.db.session import advance_summary

    await advance_summary(
        db,
        "sess-1",
        summary_json={
            "visitor_context": "stale",
            "open_questions": [],
            "commitments": [],
            "notes": [],
        },
        summary_through_message_id=0,
    )
    usage = Usage(input_tokens=1, output_tokens=1)
    corrected_json = (
        '{"visitor_context": "corrected", "open_questions": [], '
        '"commitments": [], "notes": []}'
    )
    fake = FakeProvider(
        responses=[LLMResponse(text=corrected_json, usage=usage, finish_reason="stop")]
    )
    context = ToolContext(db=db, session_id="sess-1", summarizer_provider=fake)

    call = ToolCall(
        id="call_1",
        name="flag_summary_conflict",
        arguments={"explanation": "visitor said they're no longer interested in that role"},
    )
    result = await execute_tool(call, context)
    assert result == {"reconciled": True}


# --- persist_receipt_for (Issue #11) -----------------------------------------


def test_persist_receipt_for_domain_tool_is_true() -> None:
    assert persist_receipt_for("get_current_date") is True


def test_persist_receipt_for_meta_tools_is_false() -> None:
    assert persist_receipt_for("save_visitor_info") is False
    assert persist_receipt_for("flag_summary_conflict") is False


def test_persist_receipt_for_unknown_tool_defaults_true() -> None:
    assert persist_receipt_for("not_a_real_tool") is True


def test_save_visitor_info_and_flag_summary_conflict_do_not_persist_receipts() -> None:
    assert SAVE_VISITOR_INFO.persist_receipt is False
    assert FLAG_SUMMARY_CONFLICT.persist_receipt is False
