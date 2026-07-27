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
        # Copy, not a reference: a caller that reuses/mutates the same list
        # across multiple calls (e.g. the orchestration loop's tool-call
        # round trips, Issue #10) would otherwise leave every entry in
        # self.calls pointing at the same, later-mutated list — silently
        # corrupting any assertion against an earlier call's exact contents.
        self.calls.append(list(messages))
        return next(self._responses)

    async def complete_stream(
        self, messages: list[Message], tools: list[ToolDef], max_tokens: int
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(list(messages))
        for event in next(self._streams):
            yield event
