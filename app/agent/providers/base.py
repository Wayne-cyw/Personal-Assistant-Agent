"""LLM provider interface (Engineering Guide 4.2/4.3): one Protocol, provider-
agnostic message/tool/response types, and a sanitized upstream-error type.

All other code imports only from this module (plus the `get_provider`
factory in `providers/__init__.py`) — never a concrete provider module
directly, so switching providers later touches one file.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol

from pydantic import BaseModel


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, object]


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None  # set when role == "tool"
    tool_calls: list[ToolCall] | None = None  # set on an assistant message that called tools
    # No-op for OpenAI (which caches prompt prefixes automatically); reserved
    # so a future Anthropic provider can map this to cache_control without
    # changing the Message type.
    cache_breakpoint: bool = False


class ToolDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, object]  # JSON Schema


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    tool_calls: list[ToolCall] = []
    usage: Usage
    finish_reason: str


class StreamEvent(BaseModel):
    type: Literal["delta", "done"]
    delta: str | None = None  # set when type == "delta"
    response: LLMResponse | None = None  # set when type == "done"


class UpstreamError(Exception):
    """Raised for any failed LLM provider call. Constructed with only these
    three sanitized fields — never the raw SDK/httpx exception, which can
    carry the outbound request's Authorization header (Issue #6).
    """

    def __init__(self, status_code: int, error_type: str, message: str) -> None:
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        super().__init__(message)


class LLMProvider(Protocol):
    async def complete(
        self, messages: list[Message], tools: list[ToolDef], max_tokens: int
    ) -> LLMResponse: ...

    def complete_stream(
        self, messages: list[Message], tools: list[ToolDef], max_tokens: int
    ) -> AsyncIterator[StreamEvent]: ...
