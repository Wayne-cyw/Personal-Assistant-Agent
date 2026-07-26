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


def get_provider(role: Role = "main") -> LLMProvider:
    model = {
        "main": settings.llm_model,
        "classifier": settings.classifier_model,
        "summarizer": settings.summarizer_model,
    }[role]

    if settings.llm_provider == LLMProviderSetting.OPENAI:
        return OpenAIProvider(api_key=settings.openai_api_key, model=model)

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
