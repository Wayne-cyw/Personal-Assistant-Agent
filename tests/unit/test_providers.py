import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from openai import APIConnectionError, APIStatusError

from app.agent.providers import get_provider
from app.agent.providers.base import LLMResponse, Message, ToolCall, ToolDef, Usage
from app.agent.providers.fake import FakeProvider
from app.agent.providers.openai import OpenAIProvider


def _messages() -> list[Message]:
    return [Message(role="user", content="say hello")]


async def test_fake_provider_returns_scripted_response() -> None:
    response = LLMResponse(
        text="hi there", usage=Usage(input_tokens=1, output_tokens=1), finish_reason="stop"
    )
    provider = FakeProvider(responses=[response])

    result = await provider.complete(_messages(), tools=[], max_tokens=100)

    assert result is response
    assert provider.calls == [_messages()]


async def test_fake_provider_serves_responses_in_order() -> None:
    usage = Usage(input_tokens=1, output_tokens=1)
    first = LLMResponse(text="one", usage=usage, finish_reason="stop")
    second = LLMResponse(text="two", usage=usage, finish_reason="stop")
    provider = FakeProvider(responses=[first, second])

    assert (await provider.complete(_messages(), [], 100)).text == "one"
    assert (await provider.complete(_messages(), [], 100)).text == "two"


def _fake_status_error(status_code: int) -> APIStatusError:
    """Build a fake SDK exception carrying a dummy Authorization header, the
    way a real httpx/OpenAI error would — used to verify UpstreamError never
    leaks it.
    """
    import httpx

    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": "Bearer sk-dummy-secret-value-12345"},
    )
    body = {"error": {"message": "boom"}}
    response = httpx.Response(status_code=status_code, request=request, json=body)
    return APIStatusError("boom", response=response, body=body)


async def test_upstream_error_never_leaks_the_authorization_header() -> None:
    provider = OpenAIProvider(api_key="test-key", model="gpt-5.6-luna")
    error = _fake_status_error(500)

    with patch.object(
        provider._client.chat.completions, "create", new=AsyncMock(side_effect=error)
    ):
        with pytest.raises(Exception) as exc_info:  # noqa: PT011 (UpstreamError, asserted below)
            await provider._call_with_retry(_messages(), [], 100)

    from app.agent.providers.base import UpstreamError

    assert isinstance(exc_info.value, UpstreamError)
    err = exc_info.value
    dumped = f"{err.message} {err.status_code} {err.error_type} {err!r}"
    assert "sk-dummy-secret-value-12345" not in dumped


async def test_retries_then_raises_on_repeated_5xx() -> None:
    provider = OpenAIProvider(api_key="test-key", model="gpt-5.6-luna")
    error = _fake_status_error(503)
    mock_create = AsyncMock(side_effect=error)

    with patch.object(provider._client.chat.completions, "create", new=mock_create):
        from app.agent.providers.base import UpstreamError

        with pytest.raises(UpstreamError) as exc_info:
            await provider._call_with_retry(_messages(), [], 100)

    assert mock_create.call_count == 3  # initial attempt + 2 retries
    assert exc_info.value.status_code == 503


async def test_non_retryable_status_raises_immediately() -> None:
    provider = OpenAIProvider(api_key="test-key", model="gpt-5.6-luna")
    error = _fake_status_error(400)
    mock_create = AsyncMock(side_effect=error)

    with patch.object(provider._client.chat.completions, "create", new=mock_create):
        from app.agent.providers.base import UpstreamError

        with pytest.raises(UpstreamError):
            await provider._call_with_retry(_messages(), [], 100)

    assert mock_create.call_count == 1


async def test_connection_error_is_retryable() -> None:
    import httpx

    provider = OpenAIProvider(api_key="test-key", model="gpt-5.6-luna")
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = APIConnectionError(message="connection failed", request=request)
    mock_create = AsyncMock(side_effect=error)

    with patch.object(provider._client.chat.completions, "create", new=mock_create):
        from app.agent.providers.base import UpstreamError

        with pytest.raises(UpstreamError):
            await provider._call_with_retry(_messages(), [], 100)

    assert mock_create.call_count == 3


def _fake_chat_completion(text: str, cached_tokens: int = 5) -> Any:
    _cached_tokens = cached_tokens

    class _Msg:
        content = text
        tool_calls = None

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _PromptDetails:
        cached_tokens = _cached_tokens

    class _Usage:
        prompt_tokens = 20
        completion_tokens = 4
        prompt_tokens_details = _PromptDetails()

    class _Completion:
        choices = [_Choice()]
        usage = _Usage()

    return _Completion()


class _FakeFunctionDelta:
    def __init__(self, name: str | None, arguments: str | None):
        self.name = name
        self.arguments = arguments


class _FakeToolCallDelta:
    def __init__(
        self,
        index: int,
        id: str | None = None,  # noqa: A002
        name: str | None = None,
        arguments: str | None = None,
        type: str | None = "function",  # noqa: A002
    ):
        self.index = index
        self.id = id
        self.type = type
        self.function = _FakeFunctionDelta(name, arguments) if (name or arguments) else None


class _FakeDelta:
    def __init__(
        self, content: str | None = None, tool_calls: list[_FakeToolCallDelta] | None = None
    ):
        self.content = content
        self.tool_calls = tool_calls


class _FakeStreamChoice:
    def __init__(self, delta: _FakeDelta, finish_reason: str | None = None):
        self.delta = delta
        self.finish_reason = finish_reason


class _FakeStreamUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int):
        class _Details:
            pass

        details = _Details()
        details.cached_tokens = cached_tokens  # type: ignore[attr-defined]
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = details


class _FakeChunk:
    def __init__(self, choices: list[_FakeStreamChoice], usage: _FakeStreamUsage | None = None):
        self.choices = choices
        self.usage = usage


class _FakeAsyncStream:
    def __init__(self, chunks: list[_FakeChunk], raise_after: int | None = None):
        self._chunks = chunks
        self._raise_after = raise_after

    def __aiter__(self) -> Any:
        return self._gen()

    async def _gen(self) -> Any:
        for i, chunk in enumerate(self._chunks):
            if self._raise_after is not None and i == self._raise_after:
                raise _fake_status_error(500)
            yield chunk


async def test_complete_stream_yields_deltas_then_done() -> None:
    provider = OpenAIProvider(api_key="test-key", model="gpt-5.6-luna")
    chunks = [
        _FakeChunk(choices=[_FakeStreamChoice(_FakeDelta(content="Hel"))]),
        _FakeChunk(choices=[_FakeStreamChoice(_FakeDelta(content="lo"), finish_reason="stop")]),
        _FakeChunk(choices=[], usage=_FakeStreamUsage(10, 2, 3)),
    ]
    mock_create = AsyncMock(return_value=_FakeAsyncStream(chunks))

    with patch.object(provider._client.chat.completions, "create", new=mock_create):
        events = [event async for event in provider.complete_stream(_messages(), [], 100)]

    deltas = [e.delta for e in events if e.type == "delta"]
    assert deltas == ["Hel", "lo"]
    done = events[-1]
    assert done.type == "done"
    assert done.response is not None
    assert done.response.text == "Hello"
    assert done.response.finish_reason == "stop"
    assert done.response.usage.cached_input_tokens == 3


async def test_complete_stream_accumulates_tool_call_fragments() -> None:
    provider = OpenAIProvider(api_key="test-key", model="gpt-5.6-luna")
    first_delta = _FakeToolCallDelta(0, id="call_1", name="get_current_date", arguments="")
    second_delta = _FakeToolCallDelta(0, arguments='{"tz"')
    third_delta = _FakeToolCallDelta(0, arguments=': "UTC"}')
    chunks = [
        _FakeChunk(choices=[_FakeStreamChoice(_FakeDelta(tool_calls=[first_delta]))]),
        _FakeChunk(
            choices=[
                _FakeStreamChoice(_FakeDelta(tool_calls=[second_delta]), finish_reason="tool_calls")
            ]
        ),
        _FakeChunk(choices=[_FakeStreamChoice(_FakeDelta(tool_calls=[third_delta]))]),
    ]
    mock_create = AsyncMock(return_value=_FakeAsyncStream(chunks))

    with patch.object(provider._client.chat.completions, "create", new=mock_create):
        events = [event async for event in provider.complete_stream(_messages(), [], 100)]

    done = events[-1].response
    assert done is not None
    assert len(done.tool_calls) == 1
    assert done.tool_calls[0].id == "call_1"
    assert done.tool_calls[0].name == "get_current_date"
    assert done.tool_calls[0].arguments == {"tz": "UTC"}


async def test_complete_stream_handles_truncated_tool_call_json_gracefully() -> None:
    """Regression test: a budget-tier model can emit tool-call arguments
    truncated right at the token limit, producing invalid JSON. This must
    degrade to a tool-result the caller's Pydantic validation naturally
    rejects (Issue #10's "invalid args -> structured error, not an
    exception"), not raise json.JSONDecodeError and 500 the whole turn.
    """
    provider = OpenAIProvider(api_key="test-key", model="gpt-5.6-luna")
    truncated_delta = _FakeToolCallDelta(
        0, id="call_1", name="get_current_date", arguments='{"tz": "UT'  # cut off mid-value
    )
    choice = _FakeStreamChoice(_FakeDelta(tool_calls=[truncated_delta]), finish_reason="tool_calls")
    chunks = [_FakeChunk(choices=[choice])]
    mock_create = AsyncMock(return_value=_FakeAsyncStream(chunks))

    with patch.object(provider._client.chat.completions, "create", new=mock_create):
        events = [event async for event in provider.complete_stream(_messages(), [], 100)]

    done = events[-1].response
    assert done is not None
    assert len(done.tool_calls) == 1
    assert done.tool_calls[0].arguments == {"_malformed_arguments": '{"tz": "UT'}


async def test_complete_stream_sanitizes_error_on_create() -> None:
    provider = OpenAIProvider(api_key="test-key", model="gpt-5.6-luna")
    mock_create = AsyncMock(side_effect=_fake_status_error(500))

    with patch.object(provider._client.chat.completions, "create", new=mock_create):
        from app.agent.providers.base import UpstreamError

        with pytest.raises(UpstreamError) as exc_info:
            async for _ in provider.complete_stream(_messages(), [], 100):
                pass

    assert "sk-dummy-secret-value-12345" not in repr(exc_info.value)


async def test_complete_stream_sanitizes_error_mid_stream() -> None:
    provider = OpenAIProvider(api_key="test-key", model="gpt-5.6-luna")
    chunks = [
        _FakeChunk(choices=[_FakeStreamChoice(_FakeDelta(content="partial"))]),
        _FakeChunk(choices=[_FakeStreamChoice(_FakeDelta(content="more"))]),
    ]
    mock_create = AsyncMock(return_value=_FakeAsyncStream(chunks, raise_after=1))

    with patch.object(provider._client.chat.completions, "create", new=mock_create):
        from app.agent.providers.base import UpstreamError

        with pytest.raises(UpstreamError) as exc_info:
            async for _ in provider.complete_stream(_messages(), [], 100):
                pass

    assert "sk-dummy-secret-value-12345" not in repr(exc_info.value)


async def test_complete_parses_usage_and_cached_tokens() -> None:
    provider = OpenAIProvider(api_key="test-key", model="gpt-5.6-luna")
    mock_create = AsyncMock(return_value=_fake_chat_completion("hello there"))

    with patch.object(provider._client.chat.completions, "create", new=mock_create):
        response = await provider.complete(_messages(), tools=[], max_tokens=100)

    assert response.text == "hello there"
    assert response.usage.input_tokens == 20
    assert response.usage.output_tokens == 4
    assert response.usage.cached_input_tokens == 5


def _fake_chat_completion_with_tool_call(name: str, arguments: str) -> Any:
    class _Function:
        pass

    func = _Function()
    func.name = name  # type: ignore[attr-defined]
    func.arguments = arguments  # type: ignore[attr-defined]

    class _ToolCall:
        pass

    tc = _ToolCall()
    tc.id = "call_1"  # type: ignore[attr-defined]
    tc.type = "function"  # type: ignore[attr-defined]
    tc.function = func  # type: ignore[attr-defined]

    class _Msg:
        content = ""
        tool_calls = [tc]

    class _Choice:
        message = _Msg()
        finish_reason = "tool_calls"

    class _Completion:
        choices = [_Choice()]
        usage = None

    return _Completion()


async def test_complete_handles_truncated_tool_call_json_gracefully() -> None:
    """Non-streaming counterpart of the same regression: a malformed/
    truncated tool-call arguments string must not raise json.JSONDecodeError
    and turn a self-correctable tool-result error into a 500 for the whole
    turn (Issue #10 acceptance criteria).
    """
    provider = OpenAIProvider(api_key="test-key", model="gpt-5.6-luna")
    mock_create = AsyncMock(
        return_value=_fake_chat_completion_with_tool_call("get_current_date", '{"tz": "UT')
    )

    with patch.object(provider._client.chat.completions, "create", new=mock_create):
        response = await provider.complete(_messages(), tools=[], max_tokens=100)

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].arguments == {"_malformed_arguments": '{"tz": "UT'}


async def test_get_provider_resolves_model_per_role() -> None:
    main = get_provider("main")
    classifier = get_provider("classifier")
    summarizer = get_provider("summarizer")

    assert isinstance(main, OpenAIProvider)
    assert main._model == "gpt-5.6-luna"
    assert isinstance(classifier, OpenAIProvider)
    assert classifier._model == "gpt-5.6-luna"
    assert isinstance(summarizer, OpenAIProvider)
    assert summarizer._model == "gpt-5.6-luna"


def test_tool_def_round_trips_to_openai_format() -> None:
    from app.agent.providers.openai import _to_openai_tools

    tools = [ToolDef(name="get_current_date", description="returns today's date", parameters={})]
    result = _to_openai_tools(tools)

    assert result == [
        {
            "type": "function",
            "function": {
                "name": "get_current_date",
                "description": "returns today's date",
                "parameters": {},
            },
        }
    ]


def test_message_with_tool_calls_round_trips_to_openai_format() -> None:
    from app.agent.providers.openai import _to_openai_messages

    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call_1", name="get_current_date", arguments={})],
        )
    ]
    result = _to_openai_messages(messages)

    assert result[0]["tool_calls"][0]["id"] == "call_1"
    assert result[0]["tool_calls"][0]["function"]["name"] == "get_current_date"


def test_provider_modules_import_only_via_base_or_factory() -> None:
    """Acceptance criteria (Issue #4): all other code imports only base.py
    types (plus the get_provider factory) — never a concrete provider module,
    or the openai SDK itself, directly.
    """
    repo_root = Path(__file__).parents[2]
    allowed_files = {
        "app/agent/providers/__init__.py",
        "app/agent/providers/openai.py",
    }

    internal_pattern = r"from app\.agent\.providers\.(openai|fake) import"
    internal_result = subprocess.run(
        ["grep", "-rlnE", "--include=*.py", internal_pattern, "app/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    internal_matches = [
        line
        for line in internal_result.stdout.splitlines()
        if line and line != "app/agent/providers/__init__.py"
    ]
    assert internal_matches == []

    sdk_pattern = r"^\s*(import openai|from openai)"
    sdk_result = subprocess.run(
        ["grep", "-rlnE", "--include=*.py", sdk_pattern, "app/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    sdk_matches = [
        line for line in sdk_result.stdout.splitlines() if line and line not in allowed_files
    ]
    assert sdk_matches == []
