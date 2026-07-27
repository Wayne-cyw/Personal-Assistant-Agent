"""The hand-written agent orchestration loop (Engineering Guide 4.2,
Issue #10) — the component that turns "an LLM call" into "an agent".

Scope note: context assembly (system prompt + history + user message) stays
the caller's job, same as Issues #5/#9 — this function owns only the
tool-call round-trip loop. Issue #11 upgrades context assembly to the full
pinned-profile + summary + token-budgeted-window memory system; tool
availability isn't yet gated by booking state (no state machine exists
until #18), so every registered tool is always offered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.agent.providers.base import LLMProvider, Message
from app.tools.registry import TOOL_DEFS, execute_tool

FALLBACK_MESSAGE = (
    "I'm having trouble completing that right now — could you rephrase, or "
    "let me know if there's something else I can help with?"
)


@dataclass
class AgentResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    hit_iteration_cap: bool = False


async def run_agent(
    messages: list[Message],
    provider: LLMProvider,
    max_tokens: int,
    *,
    max_iterations: int,
) -> AgentResult:
    """Run the tool-call round-trip loop per 4.2's pseudocode: call the LLM,
    execute any tool calls it makes, append the results, and repeat until it
    returns a final answer or `max_iterations` is hit (a hard cap — the loop
    cannot run away). `messages` must already include the system prompt and
    conversation history; it is not mutated (a copy is extended internally).
    """
    working_messages = list(messages)
    total_input_tokens = 0
    total_output_tokens = 0

    for _ in range(max_iterations):
        response = await provider.complete(
            messages=working_messages, tools=TOOL_DEFS, max_tokens=max_tokens
        )
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        if not response.tool_calls:
            return AgentResult(
                text=response.text,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            )

        working_messages.append(
            Message(role="assistant", content=response.text, tool_calls=response.tool_calls)
        )
        for call in response.tool_calls:
            result = await execute_tool(call)
            working_messages.append(
                Message(role="tool", content=json.dumps(result), tool_call_id=call.id)
            )

    return AgentResult(
        text=FALLBACK_MESSAGE,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        hit_iteration_cap=True,
    )
