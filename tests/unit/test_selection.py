import pytest

from app.booking.selection import match_selection

SLOTS: list[dict[str, object]] = [
    {
        "slot_id": "11111111-1111-1111-1111-111111111111",
        "start_iso": "2026-08-03T09:00:00-04:00",
        "end_iso": "2026-08-03T09:30:00-04:00",
        "label": "Mon Aug 3, 9:00–9:30 AM EDT",
    },
    {
        "slot_id": "22222222-2222-2222-2222-222222222222",
        "start_iso": "2026-08-04T14:00:00-04:00",
        "end_iso": "2026-08-04T14:30:00-04:00",
        "label": "Tue Aug 4, 2:00–2:30 PM EDT",
    },
    {
        "slot_id": "33333333-3333-3333-3333-333333333333",
        "start_iso": "2026-08-05T09:00:00-04:00",
        "end_iso": "2026-08-05T09:30:00-04:00",
        "label": "Wed Aug 5, 9:00–9:30 AM EDT",
    },
]


def test_empty_proposal_list_never_matches() -> None:
    assert match_selection("2", []) is None


def test_matches_by_bare_number() -> None:
    assert match_selection("2", SLOTS) == SLOTS[1]


def test_matches_number_within_a_sentence() -> None:
    assert match_selection("let's do option 2 please", SLOTS) == SLOTS[1]


def test_number_is_one_indexed_to_presentation_order() -> None:
    assert match_selection("1", SLOTS) == SLOTS[0]
    assert match_selection("3", SLOTS) == SLOTS[2]


def test_number_out_of_range_does_not_match() -> None:
    assert match_selection("9", SLOTS) is None


def test_clock_time_hour_does_not_false_match_as_index() -> None:
    """Regression: "2:00" must not be read as index 2 via its leading
    digit — it should fall through to weekday/time label matching instead.
    """
    assert match_selection("2:00 doesn't work for me", SLOTS) is None


def test_year_like_number_does_not_false_match() -> None:
    """A 4-digit number (e.g. a year mentioned in passing) must not be
    truncated/parsed as a 1-5 index.
    """
    assert match_selection("works for 2026", SLOTS) is None


def test_matches_ordinal_word() -> None:
    assert match_selection("the second one works", SLOTS) == SLOTS[1]


def test_a_negated_reference_alongside_another_does_not_match_either() -> None:
    """A message mentioning both a rejected and (seemingly) an intended
    slot is treated as fully unresolved, not "smartly" resolved to the
    non-negated one. Several rounds of trying to be clever about *which*
    reference a negation applies to (positional, then adjacency-based)
    each fixed one phrasing while breaking or missing another — seven
    concrete real-world phrasings were still found selecting the wrong
    slot after that. Punting the whole message to clarification is
    deliberately blunt but never silently wrong.
    """
    assert match_selection("third one, not the second", SLOTS) is None
    assert match_selection("not the first, I meant 3", SLOTS) is None


def test_only_negated_references_present_does_not_match() -> None:
    assert match_selection("not the first, not the second either", SLOTS) is None


def test_matches_exact_slot_id() -> None:
    assert match_selection("33333333-3333-3333-3333-333333333333", SLOTS) == SLOTS[2]


def test_matches_unambiguous_weekday() -> None:
    assert match_selection("Tuesday works great", SLOTS) == SLOTS[1]


def test_weekday_matching_zero_slots_does_not_match() -> None:
    assert match_selection("Friday please", SLOTS) is None


def test_ambiguous_time_across_multiple_days_does_not_match() -> None:
    """9:00 appears in two different slots' labels (Mon and Wed) — a bare
    time reference with no weekday must not resolve to either.
    """
    assert match_selection("9:00 works for me", SLOTS) is None


def test_weekday_plus_time_disambiguates() -> None:
    assert match_selection("Wednesday at 9:00 works", SLOTS) == SLOTS[2]


def test_no_reference_at_all_does_not_match() -> None:
    assert match_selection("can we do a video call instead?", SLOTS) is None


def test_weekday_and_time_referring_to_different_slots_does_not_match() -> None:
    """Monday mentioned, but the time given belongs to Tuesday's slot —
    contradictory reference, must not silently pick either.
    """
    assert match_selection("Monday at 2:00 works", SLOTS) is None


def test_rejection_phrase_overrides_an_incidental_time_reference() -> None:
    """Regression: "2:00 doesn't work for me" must not be read as choosing
    the 2:00 slot — the visitor is rejecting it.
    """
    assert match_selection("2:00 doesn't work for me", SLOTS) is None


def test_rejection_phrase_overrides_an_incidental_number() -> None:
    assert match_selection("2 doesn't work, got anything else?", SLOTS) is None


def test_no_brainer_idiom_is_not_mistaken_for_a_rejection() -> None:
    """Regression: the "is a no" cue (for "second one is a no for me") must
    not fire on the unrelated affirmative idiom "no-brainer" — \\b matches
    right before the hyphen too, so a naive \\bis\\s+a\\s+no\\b would.
    """
    assert match_selection("this is a no-brainer, book the second", SLOTS) == SLOTS[1]
    assert match_selection("the second is a no-brainer for me", SLOTS) == SLOTS[1]


def test_rejection_cue_overrides_a_weekday_reference() -> None:
    """The cue-word guard is checked before label-based (weekday/time)
    matching, not just before number/ordinal matching — a rejected weekday
    reference must not resolve via the label-matching path either.
    """
    assert match_selection("Tuesday doesn't work for me, how about Wednesday", SLOTS) is None


def test_rejection_cue_overrides_an_exact_slot_id_reference() -> None:
    """The cue-word guard runs before the slot_id exact-match check too —
    an unusual combination in practice (slot_ids are normally passed by a
    UI, not typed by a visitor alongside rejection language), but the
    guard's placement as the very first check in match_selection means it
    applies uniformly regardless of which resolution path would otherwise
    have matched.
    """
    slot_id = "22222222-2222-2222-2222-222222222222"
    assert match_selection(f"not {slot_id}", SLOTS) is None


def test_none_of_those_work_phrase_does_not_match() -> None:
    assert match_selection("none of those work for me", SLOTS) is None


@pytest.mark.parametrize(
    "message",
    [
        "anything but the second, let us do that one",
        "skip the second, whatever else",
        "I cannot make the second one",
        "the second is impossible for me, find something else",
        "let's not do the second",
        "not going to work for the second one",
        "the second doesn't fit my schedule",
        "second one is a no for me",
        "the second is a no-go, what else you got",
        "can we avoid the second",
        "the second won't fly for me",
        "ruling out the second",
    ],
)
def test_alternate_rejection_phrasings_do_not_wrongly_select_the_rejected_slot(
    message: str,
) -> None:
    """Regression: successive review passes each found real-world
    rejection phrasings that resolved to the explicitly rejected slot
    anyway — first a handful of specific idioms, then several more with
    the negation cue trailing or several words away from the reference.
    The generic-cue-word design (any of "not"/an n't-contraction/"cannot"/
    "impossible"/etc. anywhere in the message) closes the whole class
    rather than each phrasing individually.
    """
    assert match_selection(message, SLOTS) is None
