"""The primary and only real provider in v1 (Engineering Guide Tech Stack):
wraps the official `openai` Python SDK's async client.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from app.agent.providers.base import (
    LLMResponse,
    Message,
    StreamEvent,
    ToolCall,
    ToolDef,
    UpstreamError,
    Usage,
)

_MAX_RETRIES = 2
_RETRYABLE_STATUS_MIN = 500


def _sanitize(exc: APIStatusError | APIConnectionError) -> UpstreamError:
    """Build a sanitized UpstreamError from only status_code and the
    exception's class name. `exc` (and exc.request/exc.response, which can
    carry the outbound Authorization header) must never be assigned to any
    variable outside this function, logged, or interpolated into the
    message — the message is a fixed generic string (Engineering Guide 4.8).
    """
    status = getattr(exc, "status_code", 599)
    return UpstreamError(
        status_code=status,
        error_type=type(exc).__name__,
        message="Upstream LLM provider error",
    )


class OpenAIProvider:
    def __init__(self, api_key: str, model: str) -> None:
        # max_retries=0: the SDK's own built-in retry loop is disabled so
        # the hand-rolled retry below is the *only* retry mechanism — left
        # enabled, the two would stack and "max 2 retries" (Issue #4) would
        # silently become up to 9 real HTTP attempts.
        self._client = AsyncOpenAI(api_key=api_key, max_retries=0)
        self._model = model

    async def complete(
        self, messages: list[Message], tools: list[ToolDef], max_tokens: int
    ) -> LLMResponse:
        raw = await self._call_with_retry(messages, tools, max_tokens)
        return _parse_response(raw)

    async def complete_stream(
        self, messages: list[Message], tools: list[ToolDef], max_tokens: int
    ) -> AsyncIterator[StreamEvent]:
        # Any: see _to_openai_messages — the SDK's overloaded create() wants
        # its own precise TypedDict shapes; **kwargs-as-Any is how that
        # boundary is crossed cleanly rather than re-modeling the SDK here.
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _to_openai_messages(messages),
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)

        try:
            stream = await self._client.chat.completions.create(**kwargs)

            text_parts: list[str] = []
            tool_call_parts: dict[int, dict[str, str]] = {}
            finish_reason = "stop"
            usage = Usage(input_tokens=0, output_tokens=0)
            async for chunk in stream:
                chunk_finish, delta_text = _consume_chunk(chunk, text_parts, tool_call_parts)
                if chunk_finish is not None:
                    finish_reason = chunk_finish
                if delta_text:
                    yield StreamEvent(type="delta", delta=delta_text)
                if chunk.usage is not None:
                    usage = _usage_from_openai(chunk.usage)
        except (APIStatusError, APIConnectionError) as exc:
            # No retry here: an unknown amount of the stream may already
            # have been yielded to the caller, so silently retrying would
            # risk duplicated/out-of-order output. Sanitize and raise once.
            raise _sanitize(exc) from None

        yield StreamEvent(
            type="done",
            response=LLMResponse(
                text="".join(text_parts),
                tool_calls=_finalize_tool_calls(tool_call_parts),
                usage=usage,
                finish_reason=finish_reason,
            ),
        )

    async def _call_with_retry(
        self, messages: list[Message], tools: list[ToolDef], max_tokens: int
    ) -> ChatCompletion:
        last_err: UpstreamError | None = None
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _to_openai_messages(messages),
            "max_completion_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)
        for attempt in range(_MAX_RETRIES + 1):
            try:
                result: ChatCompletion = await self._client.chat.completions.create(**kwargs)
                return result
            except (APIStatusError, APIConnectionError) as exc:
                status = getattr(exc, "status_code", 599)
                retryable = status == 429 or status >= _RETRYABLE_STATUS_MIN
                err = _sanitize(exc)
                if not retryable or attempt == _MAX_RETRIES:
                    raise err from None
                last_err = err
                await asyncio.sleep(0.5 * 2**attempt)
        assert last_err is not None  # loop always raises or continues; unreachable otherwise
        raise last_err


def _consume_chunk(
    chunk: ChatCompletionChunk,
    text_parts: list[str],
    tool_call_parts: dict[int, dict[str, str]],
) -> tuple[str | None, str | None]:
    if not chunk.choices:
        return None, None
    choice = chunk.choices[0]
    delta_text = choice.delta.content
    if delta_text:
        text_parts.append(delta_text)
    for tc_delta in choice.delta.tool_calls or []:
        if tc_delta.type is not None and tc_delta.type != "function":
            continue  # mirrors _parse_response's non-streaming filter
        entry = tool_call_parts.setdefault(tc_delta.index, {"id": "", "name": "", "arguments": ""})
        if tc_delta.id:
            entry["id"] = tc_delta.id
        if tc_delta.function is not None:
            if tc_delta.function.name:
                entry["name"] = tc_delta.function.name
            if tc_delta.function.arguments:
                entry["arguments"] += tc_delta.function.arguments
    return choice.finish_reason, delta_text


def _finalize_tool_calls(tool_call_parts: dict[int, dict[str, str]]) -> list[ToolCall]:
    return [
        ToolCall(
            id=parts["id"],
            name=parts["name"],
            arguments=json.loads(parts["arguments"] or "{}"),
        )
        for _index, parts in sorted(tool_call_parts.items())
    ]


def _usage_from_openai(usage: object) -> Usage:
    prompt_tokens = getattr(usage, "prompt_tokens", 0)
    completion_tokens = getattr(usage, "completion_tokens", 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    return Usage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cached_input_tokens=cached,
    )


def _to_openai_messages(messages: list[Message]) -> list[Any]:
    # Any: the SDK's create() overloads require its own precise TypedDict
    # union per role; hand-modeling that here would duplicate the SDK's
    # types for no behavioral benefit. This is the one place that boundary
    # is crossed — everything else in this module works with our own types.
    result: list[dict[str, object]] = []
    for m in messages:
        entry: dict[str, object] = {"role": m.role, "content": m.content}
        if m.tool_call_id is not None:
            entry["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        result.append(entry)
    return result


def _to_openai_tools(tools: list[ToolDef]) -> list[Any]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _parse_response(raw: ChatCompletion) -> LLMResponse:
    choice = raw.choices[0]
    tool_calls = []
    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            if tc.type != "function":
                continue
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                )
            )
    usage = _usage_from_openai(raw.usage) if raw.usage is not None else Usage(
        input_tokens=0, output_tokens=0
    )
    return LLMResponse(
        text=choice.message.content or "",
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=choice.finish_reason,
    )
