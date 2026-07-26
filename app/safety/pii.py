"""PII validation and log-scrubbing helpers.

`redact()` is built out starting in Issue #6 as the defense-in-depth safety
net behind every log sink; PII validation (name/LinkedIn/email handling) is
built out in Issue #26.
"""

from __future__ import annotations

import re

# Key-shaped patterns, not an exhaustive secret-detection engine: this is
# the safety net behind "nothing that touches a log call is ever built from
# a raw secret value" (Issue #6) — by-construction sanitization is the
# primary control, this only catches a future mistake.
_SIMPLE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{10,}"),  # OpenAI-style API keys
    re.compile(r"AIza[A-Za-z0-9_-]{10,}"),  # Google-style API keys
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{10,}", re.IGNORECASE),  # bearer tokens
]

# DB connection-string credentials: keeps the scheme and host visible,
# redacts only the user:password portion.
_DB_CREDENTIALS_PATTERN = re.compile(
    r"(postgresql(?:\+\w+)?|postgres)://[^:/\s]+:[^@/\s]+@"
)


def redact(text: str) -> str:
    """Regex-scrub key-shaped substrings in `text`, replacing them with
    "[REDACTED]". Applied to every log sink (the JSON request logger, and
    exception/traceback output) before anything is written.
    """
    redacted = _DB_CREDENTIALS_PATTERN.sub(r"\1://[REDACTED]@", text)
    for pattern in _SIMPLE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted
