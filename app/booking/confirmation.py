"""Deterministic confirmation-reply classification (Issue #22, Engineering
Guide 4.5's `confirmed` step): "Only an affirmative reply advances." Like
slot selection (app/booking/selection.py), this is never left to the LLM's
own judgment — the visitor's raw reply is classified in code before the
agent loop runs, and only a classification of "affirmative" makes
calendar_create_booking's actual effect (a real, external calendar write)
legitimate.

The risk profile here is asymmetric in a way slot selection's isn't: a
false "unclear" just makes the agent ask again (mildly annoying); a false
"affirmative" creates a live Google Calendar event the visitor never
actually agreed to (a real external side effect with no automatic undo).
That asymmetry is why `_AFFIRMATIVE_RE` requires the *entire* cleaned
reply (not just its first word) to match a short, curated list of
unambiguous phrases: a prefix-anchored version of this check was tried
first and found, on review, to also match hedges and deferrals that merely
*start* with an affirmative-sounding word — "sure, I'll confirm once I
check my calendar", "perfect, I'll get back to you", "yeah, I guess",
"great question! let me think about it" all previously misclassified as
"affirmative" this way. Requiring a full match closes that whole class at
once, at the cost of also rejecting a few genuinely-clear compound replies
("yes, that works for me!") — an acceptable trade given the asymmetry
above; those just cost one extra clarifying turn instead of a wrong
booking.
"""

from __future__ import annotations

import re
from typing import Literal

ConfirmationClassification = Literal["affirmative", "negative", "unclear"]

# A clearly negative/change-of-mind reply — checked before affirmative
# words, so "no, actually let's do a different time" doesn't fall through
# to anything else. `\bno\b` excludes "no problem"/"no worries" (common
# ways of *agreeing*, not declining) via the trailing negative lookahead —
# those then fall through to the strict affirmative fullmatch below, which
# correctly leaves them "unclear" (safe) rather than misreading them either
# way.
_NEGATIVE_RE = re.compile(
    r"\bno\b(?!\s+(?:problem|worries|worry))|\bnope\b|\bcancel\b|\bnever\s?mind\b"
    r"|\bchanged?\s+my\s+mind\b|\bdifferent\s+time\b|\banother\s+time\b|\bnot\s+that\b"
    r"|\bnot\s+anymore\b|\brather\s+not\b"
)
# A question, hesitation, or hedge — the visitor needs something answered
# or hasn't decided yet. Belt-and-suspenders alongside the strict
# fullmatch below (which already rejects most hedges structurally, since
# they add trailing text past the allowed phrase), not the primary guard.
_HESITATION_RE = re.compile(
    r"\?|\bwait\b|\bhold\s+on\b|\bhmm+\b|\bmaybe\b|\bactually\b|\bnot\s+sure\b|\bnot\s+yet\b"
    r"|\bi\s+guess\b|\blet\s+me\s+(?:think|check)\b|\bget\s+back\s+to\s+you\b"
)
# The *entire* cleaned reply must match one of these, optionally followed
# by minimal trailing punctuation/politeness — not just its first word
# (see module docstring for why a prefix match was rejected).
_AFFIRMATIVE_RE = re.compile(
    r"^(yes|yep|yeah|yup|confirm(?:ed)?|book\s+it|let'?s\s+do\s+it|go\s+ahead"
    r"|sounds\s+good|that\s+works|perfect|great|sure|ok(?:ay)?|correct)"
    r"[\s!.,]*(please|thanks|thank\s+you)?[\s!.,]*$"
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
