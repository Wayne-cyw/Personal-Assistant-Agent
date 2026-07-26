"""A scripted provider for tests — never calls a real API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

from app.agent.providers.base import LLMResponse, Message, StreamEvent, ToolDef


class FakeProvider:
    def __init__(
        self,
        responses: list[LLMResponse],
        stream_events: list[list[StreamEvent]] | None = None,
    ) -> None:
        self._responses: Iterator[LLMResponse] = iter(responses)
        self._streams: Iterator[list[StreamEvent]] = iter(stream_events or [])
        self.calls: list[list[Message]] = []

    async def complete(
        self, messages: list[Message], tools: list[ToolDef], max_tokens: int
    ) -> LLMResponse:
        self.calls.append(messages)
        return next(self._responses)

    async def complete_stream(
        self, messages: list[Message], tools: list[ToolDef], max_tokens: int
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(messages)
        for event in next(self._streams):
            yield event
