import pytest
from pydantic import ValidationError

from app.config import LLMProvider, Settings


def _settings(**overrides: str) -> Settings:
    """Build a Settings instance isolated from the real process env and any
    local .env file, so tests are deterministic regardless of what's set on
    the machine running them.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_missing_llm_provider_raises_naming_the_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(ValidationError) as exc_info:
        _settings(LLM_API_KEY="key")
    assert "LLM_PROVIDER" in str(exc_info.value)


def test_missing_llm_api_key_raises_naming_the_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValidationError) as exc_info:
        _settings(LLM_PROVIDER="anthropic")
    assert "LLM_API_KEY" in str(exc_info.value)


def test_missing_both_required_vars_names_both(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValidationError) as exc_info:
        _settings()
    message = str(exc_info.value)
    assert "LLM_PROVIDER" in message
    assert "LLM_API_KEY" in message


def test_invalid_llm_provider_value_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(LLM_PROVIDER="not-a-real-provider", LLM_API_KEY="key")


def test_defaults_applied_when_only_required_vars_given() -> None:
    settings = _settings(LLM_PROVIDER="anthropic", LLM_API_KEY="key")

    assert settings.llm_provider == LLMProvider.ANTHROPIC
    assert settings.llm_model is None
    assert settings.database_url == "sqlite:///./personal_agent.db"
    assert settings.max_tokens_per_turn == 2048
    assert settings.window_high_tokens == 3000
    assert settings.window_low_tokens == 1500
    assert settings.allowed_origins == []
    assert settings.log_level == "INFO"


def test_allowed_origins_parses_comma_separated_string() -> None:
    settings = _settings(
        LLM_PROVIDER="anthropic",
        LLM_API_KEY="key",
        ALLOWED_ORIGINS="https://a.example.com, https://b.example.com",
    )

    assert settings.allowed_origins == ["https://a.example.com", "https://b.example.com"]


def test_numeric_overrides_are_coerced_to_int() -> None:
    settings = _settings(
        LLM_PROVIDER="anthropic",
        LLM_API_KEY="key",
        MAX_TOKENS_PER_TURN="512",
        WINDOW_HIGH_TOKENS="4000",
        WINDOW_LOW_TOKENS="2000",
    )

    assert settings.max_tokens_per_turn == 512
    assert settings.window_high_tokens == 4000
    assert settings.window_low_tokens == 2000
