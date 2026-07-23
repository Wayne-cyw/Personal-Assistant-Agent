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

    # LLM provider
    llm_provider: LLMProvider = Field(alias="LLM_PROVIDER")
    llm_api_key: str = Field(alias="LLM_API_KEY")
    llm_model: str | None = Field(default=None, alias="LLM_MODEL")

    # Database
    database_url: str = Field(default="sqlite:///./personal_agent.db", alias="DATABASE_URL")

    # Token / conversation-window budgets (Engineering Guide 4.3)
    max_tokens_per_turn: int = Field(default=2048, alias="MAX_TOKENS_PER_TURN")
    window_high_tokens: int = Field(default=3000, alias="WINDOW_HIGH_TOKENS")
    window_low_tokens: int = Field(default=1500, alias="WINDOW_LOW_TOKENS")

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


settings = Settings()  # type: ignore[call-arg]  # values come from env, not call args
