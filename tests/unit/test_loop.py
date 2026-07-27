import logging

import pytest

from app.agent.loop import FALLBACK_MESSAGE, run_agent
from app.agent.providers.base import LLMResponse, Message, ToolCall, Usage
from app.agent.providers.fake import FakeProvider


def _usage(input_tokens: int = 10, output_tokens: int = 5) -> Usage:
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)


def _messages() -> list[Message]:
    return [
        Message(role="system", content="you are a helpful assistant"),
        Message(role="user", content="what day is it?"),
    ]


async def test_tool_call_then_final_answer() -> None:
    """(a) tool call -> result -> final answer."""
    tool_call_response = LLMResponse(
        text="",
        tool_calls=[ToolCall(id="call_1", name="get_current_date", arguments={})],
        usage=_usage(),
        finish_reason="tool_calls",
    )
    final_response = LLMResponse(
        text="Today is the date the tool returned.",
        usage=_usage(),
        finish_reason="stop",
    )
    provider = FakeProvider(responses=[tool_call_response, final_response])

    result = await run_agent(_messages(), provider, max_tokens=100, max_iterations=5)

    assert result.text == "Today is the date the tool returned."
    assert result.hit_iteration_cap is False
    assert len(provider.calls) == 2  # one call that triggered the tool, one with the result

    # The second call's message list must include the tool result.
    second_call_messages = provider.calls[1]
    tool_result_messages = [m for m in second_call_messages if m.role == "tool"]
    assert len(tool_result_messages) == 1
    assert tool_result_messages[0].tool_call_id == "call_1"
    assert "date" in tool_result_messages[0].content

    # Regression: FakeProvider.calls must store a snapshot per call, not a
    # reference to the loop's mutated-in-place working list — otherwise the
    # first call's recorded messages would retroactively "gain" the tool
    # call/result that were only appended after it was made.
    first_call_messages = provider.calls[0]
    assert not any(m.role == "tool" for m in first_call_messages)
    assert len(first_call_messages) < len(second_call_messages)


async def test_multiple_tool_calls_in_one_turn_all_get_matching_results() -> None:
    """Regression test: when the model makes several tool calls in a single
    response, each must get its own tool-result message with the matching
    tool_call_id, in order — an ordering/ID-matching bug here would be easy
    to introduce and easy to miss without a test exercising more than one
    call per response.
    """
    multi_call_response = LLMResponse(
        text="",
        tool_calls=[
            ToolCall(id="call_1", name="get_current_date", arguments={}),
            ToolCall(id="call_2", name="get_current_date", arguments={}),
            ToolCall(id="call_3", name="get_current_date", arguments={}),
        ],
        usage=_usage(),
        finish_reason="tool_calls",
    )
    final_response = LLMResponse(text="all done", usage=_usage(), finish_reason="stop")
    provider = FakeProvider(responses=[multi_call_response, final_response])

    result = await run_agent(_messages(), provider, max_tokens=100, max_iterations=5)

    assert result.text == "all done"
    second_call_messages = provider.calls[1]
    tool_result_messages = [m for m in second_call_messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_result_messages] == ["call_1", "call_2", "call_3"]
    assert all("date" in m.content for m in tool_result_messages)

    # Exactly one assistant message carrying all three tool_calls, followed
    # immediately by the three tool results, in order — the shape the
    # OpenAI provider's message serialization expects.
    assistant_msg = second_call_messages[-4]
    assert assistant_msg.role == "assistant"
    assert assistant_msg.tool_calls is not None
    assert [tc.id for tc in assistant_msg.tool_calls] == ["call_1", "call_2", "call_3"]


async def test_token_usage_accumulates_across_iterations() -> None:
    tool_call_response = LLMResponse(
        text="",
        tool_calls=[ToolCall(id="call_1", name="get_current_date", arguments={})],
        usage=_usage(input_tokens=10, output_tokens=2),
        finish_reason="tool_calls",
    )
    final_response = LLMResponse(
        text="done",
        usage=_usage(input_tokens=15, output_tokens=3),
        finish_reason="stop",
    )
    provider = FakeProvider(responses=[tool_call_response, final_response])

    result = await run_agent(_messages(), provider, max_tokens=100, max_iterations=5)

    assert result.input_tokens == 25
    assert result.output_tokens == 5


async def test_infinite_tool_loop_hits_cap_and_returns_fallback() -> None:
    """(b) an infinite-tool-loop script hits the cap and returns the
    fallback message.
    """
    always_calls_tool = LLMResponse(
        text="",
        tool_calls=[ToolCall(id="call_x", name="get_current_date", arguments={})],
        usage=_usage(),
        finish_reason="tool_calls",
    )
    provider = FakeProvider(responses=[always_calls_tool] * 5)

    result = await run_agent(_messages(), provider, max_tokens=100, max_iterations=5)

    assert result.text == FALLBACK_MESSAGE
    assert result.hit_iteration_cap is True
    assert len(provider.calls) == 5  # never exceeds max_iterations


async def test_iteration_cap_hit_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Engineering Guide 4.2: "if the cap is hit ... the incident is
    logged" — this is the only production signal that the model is looping
    on tool calls instead of reaching a final answer within budget.
    """
    always_calls_tool = LLMResponse(
        text="",
        tool_calls=[ToolCall(id="call_x", name="get_current_date", arguments={})],
        usage=_usage(),
        finish_reason="tool_calls",
    )
    provider = FakeProvider(responses=[always_calls_tool] * 3)

    with caplog.at_level(logging.WARNING):
        await run_agent(_messages(), provider, max_tokens=100, max_iterations=3)

    assert any(
        record.levelno == logging.WARNING and "iteration" in record.getMessage().lower()
        for record in caplog.records
    )


async def test_unknown_tool_name_surfaces_as_tool_result_not_exception() -> None:
    """(c) invalid args surface as a tool-result error, not an exception —
    an unknown tool name is the loop-level analogue of "invalid" (the
    per-field schema-validation case is covered directly against
    execute_tool in tests/unit/test_tool_registry.py, since get_current_date
    has no fields and so cannot itself produce a validation failure).
    """
    unknown_tool_response = LLMResponse(
        text="",
        tool_calls=[ToolCall(id="call_1", name="not_a_real_tool", arguments={})],
        usage=_usage(),
        finish_reason="tool_calls",
    )
    final_response = LLMResponse(text="recovered", usage=_usage(), finish_reason="stop")
    provider = FakeProvider(responses=[unknown_tool_response, final_response])

    result = await run_agent(_messages(), provider, max_tokens=100, max_iterations=5)

    assert result.text == "recovered"
    tool_result = provider.calls[1][-1]  # last message before the 2nd LLM call is the tool result
    assert tool_result.role == "tool"
    assert "error" in tool_result.content


async def test_final_answer_with_no_tool_calls_returns_immediately() -> None:
    response = LLMResponse(text="hi!", usage=_usage(), finish_reason="stop")
    provider = FakeProvider(responses=[response])

    result = await run_agent(_messages(), provider, max_tokens=100, max_iterations=5)

    assert result.text == "hi!"
    assert len(provider.calls) == 1


async def test_original_messages_list_not_mutated() -> None:
    original = _messages()
    original_len = len(original)
    response = LLMResponse(text="hi", usage=_usage(), finish_reason="stop")
    provider = FakeProvider(responses=[response])

    await run_agent(original, provider, max_tokens=100, max_iterations=5)

    assert len(original) == original_len
