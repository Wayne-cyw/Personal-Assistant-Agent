"""Pydantic Settings — the single typed source of truth for all configuration.

No other module may read os.environ directly; add new settings here as later
issues introduce them.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # LLM provider — OpenAI is the chosen primary provider for v1 (Engineering
    # Guide Tech Stack); all three LLM roles are pinned to gpt-5.6-luna.
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, alias="LLM_PROVIDER")
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-5.6-luna", alias="LLM_MODEL")
    classifier_model: str = Field(default="gpt-5.6-luna", alias="CLASSIFIER_MODEL")
    summarizer_model: str = Field(default="gpt-5.6-luna", alias="SUMMARIZER_MODEL")

    # Database — async driver required (Engineering Guide 4.7 rule 1): SQLite
    # (aiosqlite) for local dev/tests, Postgres (asyncpg) in production.
    database_url: str = Field(
        default="sqlite+aiosqlite:///./agent.db", alias="DATABASE_URL"
    )

    # Token / conversation-window budgets (Engineering Guide 4.3)
    max_tokens_per_turn: int = Field(default=2048, alias="MAX_TOKENS_PER_TURN")
    window_high_tokens: int = Field(default=3000, alias="WINDOW_HIGH_TOKENS")
    window_low_tokens: int = Field(default=1500, alias="WINDOW_LOW_TOKENS")

    # Orchestration loop (Engineering Guide 4.2, Issue #10): hard cap on
    # tool-call round-trips per turn before falling back to a static message.
    # ge=1: a misconfigured 0 would silently skip every LLM call and always
    # return the fallback message, with no startup-time error to catch it.
    max_iterations: int = Field(default=5, ge=1, alias="MAX_ITERATIONS")

    # CORS (Engineering Guide 4.8): empty in v1, comma-separated when set.
    # NoDecode stops pydantic-settings from JSON-decoding the raw env string
    # before our validator runs (list-typed fields are decoded as JSON by
    # default, which would reject a plain comma-separated value).
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=list, alias="ALLOWED_ORIGINS"
    )

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        # logging.Logger.setLevel requires exact-case names ("INFO", not
        # "info"); normalizing here means every consumer gets a valid level
        # regardless of how the env var was cased.
        if isinstance(value, str):
            return value.upper()
        return value


settings = Settings()  # type: ignore[call-arg]  # values come from env, not call args
