"""LLM provider interface and concrete provider implementations.

`get_provider()` is the only supported way to obtain a provider — call
sites depend on this factory plus the `base` module's types, never a
concrete provider module directly (Issue #4 acceptance criteria).
"""

from __future__ import annotations

from typing import Literal

from app.agent.providers.base import LLMProvider
from app.agent.providers.openai import OpenAIProvider
from app.config import LLMProvider as LLMProviderSetting
from app.config import settings

Role = Literal["main", "classifier", "summarizer"]

# One provider instance per role, reused for the life of the process: each
# OpenAIProvider owns an AsyncOpenAI client with its own httpx connection
# pool, and this app runs as a single long-lived worker (Engineering Guide
# 4.3) — building a fresh client per call would leak connections over time.
_providers: dict[Role, LLMProvider] = {}


def get_provider(role: Role = "main") -> LLMProvider:
    if role in _providers:
        return _providers[role]

    model = {
        "main": settings.llm_model,
        "classifier": settings.classifier_model,
        "summarizer": settings.summarizer_model,
    }[role]

    if settings.llm_provider != LLMProviderSetting.OPENAI:
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

    provider = OpenAIProvider(api_key=settings.openai_api_key, model=model)
    _providers[role] = provider
    return provider
