"""Smoke test for the LLM provider interface (Issue #4 acceptance criteria):
sends "say hello" through the configured real provider and prints the
completion. Requires a valid OPENAI_API_KEY — not run as part of `pytest`.

Usage: python scripts/smoke_llm.py
"""

from __future__ import annotations

import asyncio

from app.agent.providers import get_provider
from app.agent.providers.base import Message


async def main() -> None:
    provider = get_provider("main")
    response = await provider.complete(
        messages=[Message(role="user", content="say hello")],
        tools=[],
        max_tokens=100,
    )
    print(response.text)
    print(f"usage: {response.usage}")


if __name__ == "__main__":
    asyncio.run(main())
