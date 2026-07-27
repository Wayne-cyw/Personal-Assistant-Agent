"""Token counting for the memory system's window budget (Engineering Guide
4.3, Issue #11): "count window tokens with the provider tokenizer (or a
calibrated estimate)."

`gpt-5.6-luna` is a pinned model name that doesn't exist in tiktoken's model
registry, so `tiktoken.encoding_for_model()` would raise — this uses
`cl100k_base` directly instead: a fixed, always-resolvable encoding that
tiktoken ships without needing a per-model lookup. It's not guaranteed to be
byte-for-byte what the real provider counts, but it's a stable, calibrated
estimate, which is exactly what this budget check needs (a consistent
signal to trigger eviction on, not a billing-accurate count — the real,
billed token counts always come from `LLMResponse.usage`).

Falls back to a chars-per-token heuristic if tiktoken can't load its
encoding data at all (e.g. no network on first use in an offline
environment, and no local cache yet) — the eviction check must never crash
a turn over a tokenizer being unavailable.
"""

from __future__ import annotations

import logging

from app.agent.providers.base import Usage

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN_ESTIMATE = 4

_encoding = None
_encoding_load_failed = False


def _get_encoding() -> object | None:
    global _encoding, _encoding_load_failed
    if _encoding is not None or _encoding_load_failed:
        return _encoding
    try:
        import tiktoken

        _encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.warning(
            "tiktoken encoding unavailable; falling back to a chars/%d token estimate",
            _CHARS_PER_TOKEN_ESTIMATE,
        )
        _encoding_load_failed = True
    return _encoding


def count_tokens(text: str) -> int:
    encoding = _get_encoding()
    if encoding is None:
        return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE) if text else 0
    return len(encoding.encode(text))  # type: ignore[attr-defined]


# Engineering Guide 4.3: "cached input is billed at roughly 10% of the
# standard rate, automatically." Used to weight sessions.token_budget_used
# (Issue #12) by real cost rather than raw count, so a heavily-cached turn
# doesn't eat into the budget as fast as an uncached one of the same size.
CACHED_INPUT_DISCOUNT = 0.1


def effective_tokens(*, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> int:
    """A cost-weighted token count for budget accounting (Issue #12) —
    distinct from count_tokens (a pre-call estimate for the eviction
    trigger), this consumes real, provider-reported LLMResponse.usage
    figures. `cached_input_tokens` is assumed to already be a subset of
    `input_tokens` (matching Usage's field semantics in
    app/agent/providers/base.py), so it's discounted rather than added on
    top of the full input count.
    """
    uncached_input = input_tokens - cached_input_tokens
    weighted_cached = cached_input_tokens * CACHED_INPUT_DISCOUNT
    return round(uncached_input + weighted_cached + output_tokens)


def effective_tokens_from_usage(usage: Usage) -> int:
    """Convenience wrapper over effective_tokens for the common case of
    already holding a provider Usage object (every summarizer call site in
    app/agent/memory.py) rather than loose ints.
    """
    return effective_tokens(
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        output_tokens=usage.output_tokens,
    )
