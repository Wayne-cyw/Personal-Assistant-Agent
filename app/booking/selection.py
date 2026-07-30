"""Deterministic slot-selection matching (Issue #20, Engineering Guide 4.5):
"Selection is matched deterministically... The LLM never free-associates a
datetime." The visitor's raw reply is matched against the stored proposal
list in code, or not at all — a non-match means the agent clarifies and
nothing advances (4.5: "no match -> the agent clarifies, counts as the same
round").

Slots are presented to the visitor as a 1-indexed numbered list (Issue #20's
system-prompt guidance: "present the returned slots verbatim, numbered"), so
match order here must mirror `proposed_slots` order exactly.
"""

from __future__ import annotations

import re

_ORDINAL_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
# A single combined pattern (not "try digits, then fall back to ordinal
# words") so the leftmost mention wins regardless of which form it takes —
# e.g. "third one, not the second" must resolve to "third" (it appears
# first), not "second" (an earlier bug here iterated _ORDINAL_WORDS in
# fixed dict order, which happened to check "second" before "third"
# regardless of which one actually appeared first in the text). A bare 1-5
# digit excludes a longer number (so "slot 12" or a year like "2026" doesn't
# false-match) and the hour of a clock time (so "2:00" doesn't get read as
# index 2) via negative lookaround on both sides.
_INDEX_RE = re.compile(
    r"(?<!\d)(?P<digit>[1-5])(?![\d:])|\b(?P<ordinal>first|second|third|fourth|fifth)\b"
)
# Labels render weekdays via strftime("%a") — "Mon", "Tue", "Wed", etc.
# (app/booking/slots.py's _format_label) — so matching has to normalize a
# visitor's full-name-or-abbreviation mention ("Wednesday", "Wed", "wed") to
# that same 3-letter key before comparing against the label.
_WEEKDAY_RE = re.compile(
    r"\b(mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?"
    r"|sun(?:day)?)\b"
)
# HH:MM, optionally with am/pm attached or space-separated (label times are
# always zero-padded, e.g. "9:00", "2:30").
_CLOCK_TIME_RE = re.compile(r"\b(\d{1,2}:\d{2})\s*(am|pm)?\b")
# A visitor rejecting every proposed slot ("none of those work", "2:00
# doesn't work for me") can still incidentally reference a number/weekday/
# time — without this guard, "2:00 doesn't work" would misread as *choosing*
# the 2:00 slot instead of rejecting it. Checked before any positive match
# is attempted, deliberately over the acceptance-criteria phrase ("none of
# those work") plus the most common rejection phrasings; this is not full
# negation-scope parsing (e.g. "not sure, but I'll take the 2nd one" isn't
# handled) — just closing the obvious false-positive-selection risk.
_REJECTION_PHRASES = (
    "none of",
    "not any of",
    "doesn't work",
    "does not work",
    "don't work",
    "do not work",
    "won't work",
    "will not work",
    "no good",
)


def match_selection(
    message: str, proposed_slots: list[dict[str, object]]
) -> dict[str, object] | None:
    """Returns the matched slot dict, or None if `message` cannot be
    resolved to exactly one of `proposed_slots` unambiguously. Tries, in
    order: an exact slot_id substring, a 1-based number/ordinal reference to
    presentation order, then a weekday-name or clock-time reference against
    each slot's rendered label (only resolved if it narrows to exactly one
    candidate — an ambiguous label match is treated the same as no match).
    """
    if not proposed_slots:
        return None

    text = message.strip().lower()

    if any(phrase in text for phrase in _REJECTION_PHRASES):
        return None

    for slot in proposed_slots:
        slot_id = str(slot.get("slot_id", ""))
        if slot_id and slot_id in message:
            return slot

    index = _extract_index(text)
    if index is not None and 1 <= index <= len(proposed_slots):
        return proposed_slots[index - 1]

    label_matches = [
        slot for slot in proposed_slots if _label_matches(text, str(slot.get("label", "")))
    ]
    if len(label_matches) == 1:
        return label_matches[0]

    return None


def _extract_index(text: str) -> int | None:
    match = _INDEX_RE.search(text)
    if match is None:
        return None
    if match.group("digit"):
        return int(match.group("digit"))
    return _ORDINAL_WORDS[match.group("ordinal")]


def _label_matches(text: str, label: str) -> bool:
    """True if `text` mentions a weekday and/or a clock time that's present
    in `label` — and mentions at least one of the two at all (an empty
    reply matches nothing, not every slot).
    """
    label_lower = label.lower()

    mentioned_weekdays = {m.group(1)[:3] for m in _WEEKDAY_RE.finditer(text)}
    weekday_ok = not mentioned_weekdays or any(day in label_lower for day in mentioned_weekdays)

    mentioned_times = {m.group(1) for m in _CLOCK_TIME_RE.finditer(text)}
    time_ok = not mentioned_times or any(t in label_lower for t in mentioned_times)

    if not mentioned_weekdays and not mentioned_times:
        return False
    return weekday_ok and time_ok
