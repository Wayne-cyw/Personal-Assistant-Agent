"""Pydantic Settings — the single typed source of truth for all configuration.

No other module may read os.environ directly; add new settings here as later
issues introduce them.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, field_validator, model_validator
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

    # Conversation memory (Engineering Guide 4.3, Issue #11).
    summary_max_tokens: int = Field(default=150, ge=1, alias="SUMMARY_MAX_TOKENS")
    pinned_facts_max: int = Field(default=5, ge=0, alias="PINNED_FACTS_MAX")
    summary_audit_interval: int = Field(default=3, ge=1, alias="SUMMARY_AUDIT_INTERVAL")
    # 4.3: the per-session asyncio.Lock (app/agent/session_lock.py) is only
    # correct with exactly one worker process. False (default): a detected
    # second worker only logs a prominent warning. True: refuse to start.
    # Full deploy-config enforcement is Issue #34 — this is the in-app guard
    # 4.3 also calls for, using WEB_CONCURRENCY (the common gunicorn/uvicorn-
    # worker-manager env var) as the best available signal a bare `uvicorn
    # --workers N` invocation has no other way to surface to the process.
    # Read here (not via os.environ in app/main.py) per Issue #2's rule that
    # only this module reads the process environment directly.
    enforce_single_worker: bool = Field(default=False, alias="ENFORCE_SINGLE_WORKER")
    web_concurrency: int | None = Field(default=None, alias="WEB_CONCURRENCY")

    # Cost-control guardrails (Engineering Guide 4.6 layer 4, Issue #12).
    # A tighter, chat-reply-specific cap than max_tokens_per_turn — pairs
    # with #8's brevity rule (2-4 sentences by default). max_tokens_per_turn
    # remains the outer ceiling enforced on every provider call, including
    # this one (app/api/chat.py takes min(chat_max_output_tokens,
    # max_tokens_per_turn)); this is the tighter, ordinary-chat-turn value
    # in the common case where it's the smaller of the two.
    chat_max_output_tokens: int = Field(default=500, ge=1, alias="CHAT_MAX_OUTPUT_TOKENS")
    # Per-session lifetime token budget. No number is given in the
    # Engineering Guide; 50,000 is a deliberately generous default (dozens
    # of ordinary turns plus a few evictions) that a real deployment should
    # tune against observed cost, not a value with any special significance.
    session_token_budget: int = Field(default=50_000, ge=1, alias="SESSION_TOKEN_BUDGET")
    # Required, no default (same pattern as openai_api_key): the wrap-up
    # message shown once a session hits its budget names this address
    # explicitly, so shipping a placeholder here would leak into a real
    # visitor-facing response.
    owner_contact_email: str = Field(alias="OWNER_CONTACT_EMAIL")

    # Google Calendar (Engineering Guide Tech Stack, Issue #17). Required, no
    # default (same pattern as openai_api_key) — obtained via the one-time
    # local OAuth flow, scripts/gcal_auth.py; see README's setup section.
    google_client_id: str = Field(alias="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field(alias="GOOGLE_CLIENT_SECRET")
    google_refresh_token: str = Field(alias="GOOGLE_REFRESH_TOKEN")
    # "primary" is the Google Calendar API's own alias for the account's
    # default calendar — a sensible non-secret default; override only if the
    # owner books off a dedicated secondary calendar.
    google_calendar_id: str = Field(default="primary", alias="GOOGLE_CALENDAR_ID")

    # Soft-hold duration on a selected-but-not-yet-confirmed slot (Issue #21,
    # 4.5's slot_selected step). DB-only — never written to the calendar.
    # 10 minutes is the issue's own stated default: long enough to type a
    # name/email, short enough that an abandoned flow doesn't block a real
    # slot for long.
    hold_minutes: int = Field(default=10, ge=1, alias="HOLD_MINUTES")

    # Booking-specific rate limits (Issue #23, PRD: "booking abuse is
    # costlier than Q&A abuse" — stricter and separate from #24's general
    # chat-message limits). Per-session is a lifetime cap (no reset window
    # — bounded by the session's own natural lifetime, not a rolling
    # period); per-IP is explicitly "per day" in the issue text. Neither
    # number is authoritative from the issue text as written: it names
    # BOOKING_ATTEMPTS_PER_SESSION=3, but "attempt" = one calendar_find_slots
    # call (the natural unit, since that's the tool's own entry point into
    # the flow) means a single legitimate negotiation — initial proposal,
    # one re-propose, one widen, one more call that lands on the
    # email-fallback conclusion (app/booking/state.py's negotiation cap,
    # Issue #19/#20) — already takes 4 calls on its own. 3 would rate-limit
    # that already-shipped, already-tested flow on its own legitimate last
    # step; 4 is the smallest cap that doesn't (user-confirmed after this
    # was discovered while wiring the check into calendar_find_slots). No
    # default is specified for the per-IP cap in the Engineering Guide
    # either; 10 is deliberately looser than the per-session cap so a
    # shared office/NAT IP with a few genuine visitors in one day isn't
    # false-positived by one visitor's own legitimate attempts.
    booking_attempts_per_session: int = Field(default=4, ge=1, alias="BOOKING_ATTEMPTS_PER_SESSION")
    booking_attempts_per_ip_per_day: int = Field(
        default=10, ge=1, alias="BOOKING_ATTEMPTS_PER_IP_PER_DAY"
    )

    # General per-session/per-IP message rate limits (Issue #24) — apply to
    # all chat traffic, sitting underneath #23's stricter booking-attempt
    # caps. Both defaults are the issue's own stated numbers.
    messages_per_session_per_min: int = Field(
        default=10, ge=1, alias="MESSAGES_PER_SESSION_PER_MIN"
    )
    messages_per_ip_per_min: int = Field(default=20, ge=1, alias="MESSAGES_PER_IP_PER_MIN")
    # Whether to trust the X-Forwarded-For header for per-IP rate limiting
    # (Issue #24: "honor X-Forwarded-For only from the trusted host
    # proxy"). The issue doesn't name a mechanism for identifying that
    # proxy, and no specific IP/CIDR range is documented for any of the
    # Tech Stack's candidate hosts (Render/Railway/Fly.io) — user-confirmed
    # design call: a plain on/off switch rather than an IP allowlist,
    # since this app always runs as exactly one uvicorn worker behind
    # whichever platform's own edge proxy is the only way in (Engineering
    # Guide 4.3's single-worker rule). There's no scenario where the
    # deployed app receives a direct, unproxied connection, so "are we
    # running in an environment where this header can be trusted at all"
    # is the only question that actually needs answering, not "which
    # specific hop sent it." Default False (untrusted, use the raw TCP
    # peer) so local dev/tests are never fooled by a spoofed header;
    # deploy config is what should set this True in production.
    trust_x_forwarded_for: bool = Field(default=False, alias="TRUST_X_FORWARDED_FOR")

    # Owner notification (Issue #22, app/notify.py). Optional: unset just
    # means notifications no-op with a logged warning rather than crashing
    # startup — a dev environment without a configured webhook shouldn't be
    # blocked from booking end-to-end, only from the notification side effect.
    owner_notify_webhook_url: str | None = Field(default=None, alias="OWNER_NOTIFY_WEBHOOK_URL")

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

    @field_validator("web_concurrency", mode="before")
    @classmethod
    def _ignore_malformed_web_concurrency(cls, value: object) -> object:
        # WEB_CONCURRENCY is set by external tooling (gunicorn/uvicorn
        # process managers) this app doesn't control the format of — a
        # malformed value is a signal worth logging (app/main.py's startup
        # guard), not grounds for crashing the whole app at construction.
        if isinstance(value, str) and not value.strip().lstrip("-").isdigit():
            return None
        return value

    @model_validator(mode="after")
    def _check_max_tokens_per_turn_supports_summarization(self) -> Settings:
        # Mirrors app/agent/memory.py's _SUMMARIZER_MAX_TOKENS_FLOOR
        # (duplicated here, not imported, to avoid a config<->memory
        # circular import — keep the two in sync if either changes).
        # _summarize() clamps its own call to
        # min(its own computed budget, max_tokens_per_turn) — Issue #12's
        # "enforce MAX_TOKENS_PER_TURN on every provider call" rule. Below
        # this floor, that clamp guarantees every summarizer call is too
        # small to emit valid JSON, so every eviction fails, retries once,
        # and burns real tokens with no way to ever succeed (a fail-fast
        # startup error here is cheap insurance against that silent,
        # unbounded-retry cost).
        min_reasonable = 300
        if self.max_tokens_per_turn < min_reasonable:
            raise ValueError(
                f"MAX_TOKENS_PER_TURN={self.max_tokens_per_turn} is too low for the memory "
                f"summarizer to ever produce valid JSON (needs at least {min_reasonable}) — "
                "every eviction would fail and retry indefinitely. Raise MAX_TOKENS_PER_TURN."
            )
        return self


settings = Settings()  # type: ignore[call-arg]  # values come from env, not call args
