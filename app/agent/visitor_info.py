"""Server-side extraction of a volunteered visitor name / LinkedIn URL from
free text (Issue #9).

This is a conservative, regex-based placeholder for the LLM tool-based
extraction that lands in Issue #11 (`save_visitor_info`) — this module never
calls the LLM and is deliberately narrow: false negatives (missing a name
that was phrased unusually) are fine, false positives (capturing the wrong
thing as someone's name) are not, since a wrong pinned name is hard to
recover from once persisted.
"""

from __future__ import annotations

import re

_NAME_WORD = r"[A-Z][a-zA-Z'-]+"
_NAME_GROUP = rf"({_NAME_WORD}(?:\s+{_NAME_WORD}){{0,2}})"
_NAME_PATTERNS = [
    # (?i:...) scopes case-insensitivity to the trigger phrase only ("My
    # name is" and "my name is" both match) — the captured name itself
    # stays case-sensitive via _NAME_WORD, so a lowercase word right after
    # the trigger (e.g. "I'm not sure") is correctly not captured.
    re.compile(rf"\b(?i:my name(?:'s| is))\s+{_NAME_GROUP}"),
    re.compile(rf"\b(?i:i(?:'m| am))\s+{_NAME_GROUP}\b"),
]

# linkedin.com/in/<slug> only — the one shape Issue #9 asks to validate.
_LINKEDIN_PATTERN = re.compile(
    r"https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+/?", re.IGNORECASE
)


def extract_name(message: str) -> str | None:
    for pattern in _NAME_PATTERNS:
        match = pattern.search(message)
        if match:
            return match.group(1).strip()
    return None


def extract_linkedin_url(message: str) -> str | None:
    match = _LINKEDIN_PATTERN.search(message)
    return match.group(0) if match else None
