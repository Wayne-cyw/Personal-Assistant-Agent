from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import LLMProvider, Settings


def _settings(**overrides: str) -> Settings:
    """Build a Settings instance isolated from the real process env and any
    local .env file, so tests are deterministic regardless of what's set on
    the machine running them.
    """
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_missing_openai_api_key_raises_naming_the_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError) as exc_info:
        _settings()
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_missing_owner_contact_email_raises_naming_the_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OWNER_CONTACT_EMAIL", raising=False)
    with pytest.raises(ValidationError) as exc_info:
        _settings(OPENAI_API_KEY="key")
    assert "OWNER_CONTACT_EMAIL" in str(exc_info.value)


def test_missing_google_credentials_raise_naming_the_var(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
        with pytest.raises(ValidationError) as exc_info:
            _settings(OPENAI_API_KEY="key")
        assert var in str(exc_info.value)
        monkeypatch.setenv(var, "restored-for-next-iteration")


def test_invalid_llm_provider_value_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(LLM_PROVIDER="not-a-real-provider", OPENAI_API_KEY="key")


def test_defaults_applied_when_only_required_vars_given() -> None:
    settings = _settings(OPENAI_API_KEY="key")

    assert settings.llm_provider == LLMProvider.OPENAI
    assert settings.llm_model == "gpt-5.6-luna"
    assert settings.classifier_model == "gpt-5.6-luna"
    assert settings.summarizer_model == "gpt-5.6-luna"
    assert settings.database_url == "sqlite+aiosqlite:///./agent.db"
    assert settings.max_tokens_per_turn == 2048
    assert settings.window_high_tokens == 3000
    assert settings.window_low_tokens == 1500
    assert settings.allowed_origins == []
    assert settings.log_level == "INFO"
    assert settings.chat_max_output_tokens == 500
    assert settings.session_token_budget == 50_000
    assert settings.google_calendar_id == "primary"


def test_allowed_origins_parses_comma_separated_string() -> None:
    settings = _settings(
        OPENAI_API_KEY="key",
        ALLOWED_ORIGINS="https://a.example.com, https://b.example.com",
    )

    assert settings.allowed_origins == ["https://a.example.com", "https://b.example.com"]


def test_allowed_origins_from_real_env_var_not_json_decoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: pydantic-settings JSON-decodes list-typed fields
    sourced from the real process env / .env file *before* any
    field_validator runs, unless the field opts out via NoDecode. A plain
    comma-separated string like "https://a.com,https://b.com" is not valid
    JSON, so without NoDecode this raises SettingsError at import/construction
    time instead of being parsed. Passing ALLOWED_ORIGINS as a Settings(...)
    kwarg (as the test above does) bypasses that source entirely and would
    not have caught this.
    """
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.example.com,https://b.example.com")

    settings = Settings(_env_file=None, OPENAI_API_KEY="key")  # type: ignore[call-arg]

    assert settings.allowed_origins == ["https://a.example.com", "https://b.example.com"]


def test_allowed_origins_empty_string_from_real_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "")

    settings = Settings(_env_file=None, OPENAI_API_KEY="key")  # type: ignore[call-arg]

    assert settings.allowed_origins == []


def test_env_example_leaves_optional_vars_at_their_python_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: .env.example must not bind LLM_MODEL/DATABASE_URL to
    an explicit empty string, since an explicitly-set-but-blank key (e.g.
    `DATABASE_URL=`) overrides the Python default with "" rather than being
    treated as absent — unlike a fully commented-out/omitted key. Also
    confirms inline "KEY=  # comment" no longer corrupts the value: python-
    dotenv only strips inline comments when a non-empty value precedes them,
    so a comment on a blank-value line was previously taken literally as
    part of the value.
    """
    optional_vars = (
        "LLM_MODEL",
        "CLASSIFIER_MODEL",
        "SUMMARIZER_MODEL",
        "DATABASE_URL",
        "MAX_ITERATIONS",
        "SUMMARY_MAX_TOKENS",
        "PINNED_FACTS_MAX",
        "SUMMARY_AUDIT_INTERVAL",
        "ENFORCE_SINGLE_WORKER",
        "WEB_CONCURRENCY",
        "CHAT_MAX_OUTPUT_TOKENS",
        "SESSION_TOKEN_BUDGET",
        "GOOGLE_CALENDAR_ID",
    )
    for var in optional_vars:
        monkeypatch.delenv(var, raising=False)
    env_example = Path(__file__).parents[2] / ".env.example"

    settings = Settings(_env_file=env_example, OPENAI_API_KEY="key")  # type: ignore[call-arg]

    assert settings.llm_model == "gpt-5.6-luna"
    assert settings.classifier_model == "gpt-5.6-luna"
    assert settings.summarizer_model == "gpt-5.6-luna"
    assert settings.database_url == "sqlite+aiosqlite:///./agent.db"
    assert settings.max_iterations == 5
    assert settings.summary_max_tokens == 150
    assert settings.pinned_facts_max == 5
    assert settings.summary_audit_interval == 3
    assert settings.enforce_single_worker is False
    assert settings.web_concurrency is None
    assert settings.chat_max_output_tokens == 500
    assert settings.session_token_budget == 50_000


def test_web_concurrency_parses_numeric_env_var() -> None:
    settings = _settings(OPENAI_API_KEY="key", WEB_CONCURRENCY="3")
    assert settings.web_concurrency == 3


def test_web_concurrency_malformed_value_ignored_not_fatal() -> None:
    """Regression test: WEB_CONCURRENCY is set by external tooling (gunicorn/
    uvicorn process managers) this app doesn't control the format of — a
    malformed value must not crash startup (app/main.py's guard treats a
    missing signal as a no-op, not an error).
    """
    settings = _settings(OPENAI_API_KEY="key", WEB_CONCURRENCY="not-a-number")
    assert settings.web_concurrency is None


def test_max_iterations_zero_or_negative_rejected() -> None:
    """Regression test: MAX_ITERATIONS=0 must fail fast at startup rather
    than silently making every turn immediately return the fallback message
    (range(0) never executes the loop body).
    """
    with pytest.raises(ValidationError):
        _settings(OPENAI_API_KEY="key", MAX_ITERATIONS="0")
    with pytest.raises(ValidationError):
        _settings(OPENAI_API_KEY="key", MAX_ITERATIONS="-1")


def test_chat_max_output_tokens_and_session_token_budget_zero_or_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(OPENAI_API_KEY="key", CHAT_MAX_OUTPUT_TOKENS="0")
    with pytest.raises(ValidationError):
        _settings(OPENAI_API_KEY="key", SESSION_TOKEN_BUDGET="0")


def test_max_tokens_per_turn_too_low_for_summarizer_rejected() -> None:
    """Regression test: MAX_TOKENS_PER_TURN below the summarizer's minimum
    floor (app/agent/memory.py's _summarize clamps its own call to
    min(its budget, MAX_TOKENS_PER_TURN)) would make every eviction fail
    and retry forever, burning real tokens with no way to ever succeed —
    caught at startup instead.
    """
    with pytest.raises(ValidationError, match="MAX_TOKENS_PER_TURN"):
        _settings(OPENAI_API_KEY="key", MAX_TOKENS_PER_TURN="100")


def test_numeric_overrides_are_coerced_to_int() -> None:
    settings = _settings(
        OPENAI_API_KEY="key",
        MAX_TOKENS_PER_TURN="512",
        WINDOW_HIGH_TOKENS="4000",
        WINDOW_LOW_TOKENS="2000",
    )

    assert settings.max_tokens_per_turn == 512
    assert settings.window_high_tokens == 4000
    assert settings.window_low_tokens == 2000


def test_every_credential_and_model_name_is_env_overridable() -> None:
    """Acceptance criterion (Issue #2): every credential, connection string,
    and model name is overridable purely via env var, with no code change.
    """
    settings = _settings(
        OPENAI_API_KEY="sk-test-arbitrary",
        DATABASE_URL="postgresql+asyncpg://user:pw@host/db",
        LLM_MODEL="gpt-5.6-terra",
        CLASSIFIER_MODEL="gpt-5.6-luna-classifier-test",
        SUMMARIZER_MODEL="gpt-5.6-luna-summarizer-test",
        OWNER_CONTACT_EMAIL="owner-test@example.com",
        GOOGLE_CLIENT_ID="test-client-id-override",
        GOOGLE_CLIENT_SECRET="test-client-secret-override",
        GOOGLE_REFRESH_TOKEN="test-refresh-token-override",
        GOOGLE_CALENDAR_ID="owner@example.com",
    )

    assert settings.openai_api_key == "sk-test-arbitrary"
    assert settings.database_url == "postgresql+asyncpg://user:pw@host/db"
    assert settings.llm_model == "gpt-5.6-terra"
    assert settings.classifier_model == "gpt-5.6-luna-classifier-test"
    assert settings.summarizer_model == "gpt-5.6-luna-summarizer-test"
    assert settings.owner_contact_email == "owner-test@example.com"
    assert settings.google_client_id == "test-client-id-override"
    assert settings.google_client_secret == "test-client-secret-override"
    assert settings.google_refresh_token == "test-refresh-token-override"
    assert settings.google_calendar_id == "owner@example.com"


def test_log_level_normalized_to_uppercase() -> None:
    """Regression test: logging.Logger.setLevel requires an exact-case name
    ("INFO", not "info"); a lowercase LOG_LEVEL env var must not reach any
    consumer un-normalized, or it crashes at import/setup time.
    """
    import logging

    settings = _settings(LOG_LEVEL="debug")
    assert settings.log_level == "DEBUG"
    logging.getLogger("test_log_level_normalized").setLevel(settings.log_level)  # must not raise


def test_os_environ_read_only_in_config_module() -> None:
    """Grep audit (Issue #2 acceptance criteria): grep -r "os.environ" app/
    must return only config.py — no other module may read the process
    environment directly.
    """
    import subprocess

    repo_root = Path(__file__).parents[2]
    result = subprocess.run(
        ["grep", "-rl", "--include=*.py", "os.environ", "app/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    matches = [line for line in result.stdout.splitlines() if line]
    assert matches == ["app/config.py"]
