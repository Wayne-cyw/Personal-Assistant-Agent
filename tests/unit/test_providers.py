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


async def test_complete_parses_usage_and_cached_tokens() -> None:
    provider = OpenAIProvider(api_key="test-key", model="gpt-5.6-luna")
    mock_create = AsyncMock(return_value=_fake_chat_completion("hello there"))

    with patch.object(provider._client.chat.completions, "create", new=mock_create):
        response = await provider.complete(_messages(), tools=[], max_tokens=100)

    assert response.text == "hello there"
    assert response.usage.input_tokens == 20
    assert response.usage.output_tokens == 4
    assert response.usage.cached_input_tokens == 5


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
    types (plus the get_provider factory) — never a concrete provider module
    directly.
    """
    repo_root = Path(__file__).parents[2]
    pattern = r"from app\.agent\.providers\.(openai|fake) import"
    result = subprocess.run(
        ["grep", "-rlnE", "--include=*.py", pattern, "app/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    matches = [
        line
        for line in result.stdout.splitlines()
        if line and line not in ("app/agent/providers/__init__.py",)
    ]
    assert matches == []
