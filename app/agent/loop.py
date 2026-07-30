"""The hand-written agent orchestration loop (Engineering Guide 4.2,
Issue #10) — the component that turns "an LLM call" into "an agent".

Scope note: context assembly (system prompt + history + user message) stays
the caller's job — app/agent/memory.py (Issue #11) builds that list; this
function owns only the tool-call round-trip loop. Tool availability isn't
yet gated by booking state (no state machine exists until #18), so every
registered tool is always offered.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.agent.memory import ToolEvent
from app.agent.providers.base import LLMProvider, Message
from app.tools.context import ToolContext
from app.tools.registry import TOOL_DEFS, execute_tool, persist_receipt_for

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "I'm having trouble completing that right now — could you rephrase, or "
    "let me know if there's something else I can help with?"
)


@dataclass
class AgentResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Subset of input_tokens billed at the provider's discounted cached
    # rate (Engineering Guide 4.3) — accumulated for
    # app/agent/tokens.py's effective_tokens (Issue #12's session budget
    # accounting), not used anywhere within this module itself.
    cached_input_tokens: int = 0
    hit_iteration_cap: bool = False
    tool_events: list[ToolEvent] = field(default_factory=list)


async def run_agent(
    messages: list[Message],
    provider: LLMProvider,
    max_tokens: int,
    *,
    max_iterations: int,
    tool_context: ToolContext,
) -> AgentResult:
    """Run the tool-call round-trip loop per 4.2's pseudocode: call the LLM,
    execute any tool calls it makes, append the results, and repeat until it
    returns a final answer or `max_iterations` is hit (a hard cap — the loop
    cannot run away). `messages` must already include the system prompt and
    conversation history; it is not mutated (a copy is extended internally).
    Every tool call made along the way is collected into `AgentResult.
    tool_events` (Issue #11) so the caller can persist a receipt of each to
    the messages log via app/agent/memory.py's persist_turn.
    """
    working_messages = list(messages)
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_input_tokens = 0
    tool_events: list[ToolEvent] = []

    for _ in range(max_iterations):
        response = await provider.complete(
            messages=working_messages, tools=TOOL_DEFS, max_tokens=max_tokens
        )
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        total_cached_input_tokens += response.usage.cached_input_tokens

        if not response.tool_calls:
            return AgentResult(
                text=response.text,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cached_input_tokens=total_cached_input_tokens,
                tool_events=tool_events,
            )

        working_messages.append(
            Message(role="assistant", content=response.text, tool_calls=response.tool_calls)
        )
        for call in response.tool_calls:
            result = await execute_tool(call, tool_context)
            tool_events.append(
                ToolEvent(
                    name=call.name, result=result, persist_receipt=persist_receipt_for(call.name)
                )
            )
            working_messages.append(
                Message(role="tool", content=json.dumps(result), tool_call_id=call.id)
            )

    # 4.2's hard-cap contract: "if the cap is hit ... the incident is
    # logged" — this is the only signal that Luna is looping on tool calls
    # instead of reaching a final answer within budget.
    logger.warning(
        "agent hit the %d-iteration cap without a final answer; returning fallback",
        max_iterations,
    )
    return AgentResult(
        text=FALLBACK_MESSAGE,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        cached_input_tokens=total_cached_input_tokens,
        hit_iteration_cap=True,
        tool_events=tool_events,
    )
