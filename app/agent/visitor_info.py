"""Server-side extraction of a volunteered visitor name / LinkedIn URL from
free text (Issue #9).

This is a conservative, regex-based placeholder for the LLM tool-based
extraction that lands in Issue #11 (`save_visitor_info`) — this module never
calls the LLM and is deliberately narrow: false negatives (missing a name
that was phrased unusually) are fine, false positives (capturing the wrong
thing as someone's name) are not, since a wrong pinned name is hard to
recover from once persisted (the caller in app/api/chat.py never overwrites
a name once set — see its docstring for why, and the known limitation that
follows from it).

Only "my name is X" / "my name's X" is matched — an earlier version also
matched "I'm X" / "I am X", but that pattern has an unacceptably high false
positive rate in ordinary conversation on exactly this site ("I'm
interested in booking a call" -> "Interested"; "I'm curious about..." ->
"Curious"), since "I'm <capitalized word>" is just as often an adjective or
verb as a name. "my name is X" carries almost no such ambiguity.
"""

from __future__ import annotations

import re

_NAME_WORD = r"[A-Z][a-zA-Z'-]+"
_NAME_GROUP = rf"({_NAME_WORD}(?:\s+{_NAME_WORD}){{0,2}})"

# (?i:...) scopes case-insensitivity to the trigger phrase only ("My name
# is" and "my name is" both match) — the captured name itself stays
# case-sensitive via _NAME_WORD, so a lowercase word right after the
# trigger is correctly not captured.
_NAME_PATTERN = re.compile(rf"\b(?i:my name(?:'s| is))\s+{_NAME_GROUP}")

# linkedin.com/in/<slug>, with or without a scheme, with or without a
# 2-3-letter locale subdomain (uk./ca./www.), optionally followed by a
# query string. Two separate patterns rather than one optional-scheme
# pattern: without a scheme present, the match must start at the beginning
# of the message or right after whitespace, so "evil.com/linkedin.com/in/x"
# (where "linkedin.com" is embedded as a path segment of another domain,
# preceded by "/" — neither start-of-string nor whitespace) is correctly
# rejected, while a bare pasted "linkedin.com/in/jane-doe" is accepted.
_SLUG = r"[A-Za-z0-9\-_%]+/?(?:\?\S*)?"
_LINKEDIN_PATTERNS = [
    re.compile(rf"https?://(?:[a-z]{{2,3}}\.)?linkedin\.com/in/{_SLUG}", re.IGNORECASE),
    re.compile(rf"(?:(?<=^)|(?<=\s))(?:[a-z]{{2,3}}\.)?linkedin\.com/in/{_SLUG}", re.IGNORECASE),
]


def extract_name(message: str) -> str | None:
    match = _NAME_PATTERN.search(message)
    return match.group(1).strip() if match else None


def extract_linkedin_url(message: str) -> str | None:
    for pattern in _LINKEDIN_PATTERNS:
        match = pattern.search(message)
        if match:
            return match.group(0)
    return None
