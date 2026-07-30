"""Deterministic confirmation-reply classification (Issue #22, Engineering
Guide 4.5's `confirmed` step): "Only an affirmative reply advances." Like
slot selection (app/booking/selection.py), this is never left to the LLM's
own judgment — the visitor's raw reply is classified in code before the
agent loop runs, and only a classification of "affirmative" makes
calendar_create_booking's actual effect (a real calendar write) legitimate.

The risk profile here is the inverse of slot selection's: there, the unsafe
direction was silently matching a slot the visitor didn't mean (so the
guard had to hunt down every way to say "not that one"). Here, "unclear"
is always the safe default — nothing advances, no event gets created — so
`_AFFIRMATIVE_RE` only needs to recognize a small, curated set of
unambiguous phrases anchored at the start of the reply, rather than
needing to catch every possible way of saying "no" or "wait."
"""

from __future__ import annotations

import re
from typing import Literal

ConfirmationClassification = Literal["affirmative", "negative", "unclear"]

# A clearly negative/change-of-mind reply — checked before affirmative
# words, so "no, actually let's do a different time" doesn't fall through
# to anything else.
_NEGATIVE_RE = re.compile(
    r"\bno\b|\bnope\b|\bcancel\b|\bnever\s?mind\b|\bchanged?\s+my\s+mind\b"
    r"|\bdifferent\s+time\b|\banother\s+time\b|\bnot\s+that\b|\bnot\s+anymore\b"
)
# A question, hesitation, or hedge — the visitor needs something answered
# or hasn't decided yet; the reply is treated as unclear even if it also
# contains an affirmative-looking word ("sure, but does the invite include
# a video link?" is not a clear yes).
_HESITATION_RE = re.compile(
    r"\?|\bwait\b|\bhold\s+on\b|\bhmm+\b|\bmaybe\b|\bactually\b|\bnot\s+sure\b|\bnot\s+yet\b"
)
# Anchored at the start of the (stripped, lowercased) reply — a short list
# of unambiguous affirmations, not an attempt to recognize every possible
# phrasing (the safe default for anything not on this list is "unclear").
_AFFIRMATIVE_RE = re.compile(
    r"^(yes|yep|yeah|yup|confirm(?:ed)?|book\s+it|let'?s\s+do\s+it|go\s+ahead"
    r"|sounds\s+good|that\s+works|perfect|great|sure|ok(?:ay)?|correct)\b"
)


def classify_confirmation_reply(message: str) -> ConfirmationClassification:
    text = message.strip().lower()
    if _NEGATIVE_RE.search(text):
        return "negative"
    if _HESITATION_RE.search(text):
        return "unclear"
    if _AFFIRMATIVE_RE.match(text):
        return "affirmative"
    return "unclear"
