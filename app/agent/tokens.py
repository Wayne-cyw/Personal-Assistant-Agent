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
