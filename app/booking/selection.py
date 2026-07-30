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
# A bare 1-5 digit excludes a longer number (so "slot 12" or a year like
# "2026" doesn't false-match) and the hour of a clock time (so "2:00"
# doesn't get read as index 2) via negative lookaround on both sides.
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

# A visitor rejecting a proposed slot can phrase it in effectively unbounded
# ways, and the rejection cue can precede, follow, or sit several words away
# from whichever number/ordinal/weekday/time it's actually about: "skip the
# second", "the second is impossible", "not going to work for the second
# one", "ruling out the second", "2:00 doesn't work for me". Earlier
# versions of this guard tried to be clever about *which* reference a
# negation applied to — excluding only the nearby candidate, so a message
# like "third one, not the second" could still resolve to "third" — but
# every refinement of that approach (position-based, then adjacency-based,
# then an enumerated list of whole rejection *phrases*) kept finding a new
# real-world phrasing that slipped through and caused a *wrong* selection,
# not just a missed one.
#
# This module's whole design premise is that a missed match (the agent asks
# for clarification) is always safer than a wrong one (the agent locks in a
# hold on a slot the visitor just rejected) — so this guard is deliberately
# blunt: if the message contains *any* recognizable rejection cue *word*
# anywhere, the whole message is treated as unresolved, even if it also
# names what looks like an intended slot elsewhere ("anything but the
# second, let's do the third" still returns None, not "third"). This is not
# full negation-scope parsing; it trades "sometimes asks to clarify when a
# human reader could have figured out the real choice" for "never silently
# selects a slot the visitor just rejected."
#
# "no", "but", "out", and "pass" alone are deliberately NOT included —
# they're too common in ordinary affirmative replies ("no problem, the
# second works", "I like it, but is Tuesday possible too?", "I'm out and
# about but the second works") to trigger on safely as bare words. A few
# genuinely unambiguous idioms that use them ("anything but", "no good",
# "no-go") are still matched as fixed multi-word phrases instead. This
# means some colloquial rejections still slip through uncaught — "the
# second is a hard no", "I'll pass on the second", "the second is out for
# me" all still (wrongly) select the referenced slot rather than punting to
# clarification. Known, accepted, not chased further: the space of ways to
# reject a slot in English is unbounded, each phrase added here closes one
# case while risking a new false-positive elsewhere, and this is
# deliberately not full negation-scope parsing (see the module docstring).
# If a wrong-selection case shows up in real traffic, add it as a new
# fixed phrase rather than reaching for a broader bare word.
_REJECTION_CUE_RE = re.compile(
    r"\bnot\b"
    r"|\bnone\b"
    r"|\b\w+n't\b"  # any contraction ending in n't: don't, won't, can't, doesn't, ...
    r"|\bcannot\b"
    r"|\bimpossible\b"
    r"|\bunable\b"
    r"|\bavoid(?:ing)?\b"
    r"|\bskip(?:ping)?\b"
    r"|\bexcept\b"
    r"|\banything\s+but\b"
    r"|\bno[- ]good\b"
    r"|\bno[- ]go\b"
    r"|\brul(?:e|ing)\s+out\b"
    r"|\bis\s+a\s+no\b"
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

    if _REJECTION_CUE_RE.search(text):
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
    """A single, unambiguous digit/ordinal mention resolves; more than one
    (or none) is ambiguous. No per-candidate negation check is needed here
    — _REJECTION_CUE_RE above has already ruled out any negated message by
    the time this runs.
    """
    matches = list(_INDEX_RE.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    digit = match.group("digit")
    return int(digit) if digit else _ORDINAL_WORDS[match.group("ordinal")]


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
