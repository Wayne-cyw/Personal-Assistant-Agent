import pytest

from app.booking.confirmation import classify_confirmation_reply


@pytest.mark.parametrize(
    "message",
    [
        "yes",
        "Yes",
        "yes!",
        "yep",
        "yeah",
        "yup",
        "confirm",
        "confirmed",
        "book it",
        "book it!",
        "let's do it",
        "lets do it",
        "go ahead",
        "sounds good",
        "that works",
        "perfect",
        "great",
        "sure",
        "ok",
        "okay",
        "correct",
        "yes please",
        "yes, thanks!",
        "confirm, thank you",
    ],
)
def test_clear_affirmatives(message: str) -> None:
    assert classify_confirmation_reply(message) == "affirmative"


@pytest.mark.parametrize(
    "message",
    [
        "no",
        "nope",
        "no, let's pick a different time",
        "cancel",
        "never mind",
        "nevermind",
        "actually, I changed my mind",
        "can we do another time instead",
        "not that one",
        "not anymore, sorry",
    ],
)
def test_clear_negatives(message: str) -> None:
    assert classify_confirmation_reply(message) == "negative"


@pytest.mark.parametrize(
    "message",
    [
        "does this include a video call link?",
        "wait, what timezone is this in?",
        "hmm, let me think about it",
        "maybe",
        "not sure yet",
        "sure, but can you confirm the duration?",
        "actually hold on",
        "can you tell me more first",
        "what happens after I confirm",
        "",
        "   ",
        "I think so",
    ],
)
def test_unclear_replies_do_not_advance(message: str) -> None:
    assert classify_confirmation_reply(message) == "unclear"


@pytest.mark.parametrize(
    "message",
    [
        "Correct me if I'm wrong, but I'd rather not.",
        "Great question! Let me think about it",
        "Perfect, I'll get back to you",
        "yeah, I guess",
        "ok, let me get back to you",
        "sure, I'll confirm once I check my calendar",
    ],
)
def test_hedges_starting_with_an_affirmative_word_do_not_wrongly_confirm(message: str) -> None:
    """Regression test: a review pass found the prior prefix-anchored
    affirmative regex misclassified every one of these as "affirmative" —
    each starts with a word from the allowlist but is actually a hedge or
    deferral, which would have created a real, unwanted calendar event.
    The fullmatch-based redesign requires the *entire* reply to be one of
    the allowlisted phrases (plus minimal trailing punctuation/politeness),
    so any of these — which all have substantial content after the leading
    word — correctly fall through to "unclear" instead.
    """
    assert classify_confirmation_reply(message) != "affirmative"


@pytest.mark.parametrize(
    "message",
    [
        "no problem, let's do it",
        "no worries, book it",
    ],
)
def test_no_problem_and_no_worries_are_not_misread_as_declining(message: str) -> None:
    """Regression test: bare \\bno\\b previously matched inside these
    common ways of *agreeing*, wrongly firing a real CONFIRMATION_DECLINED
    state transition. Excluding "no problem"/"no worries" from the
    negative check is enough on its own — neither example fullmatches the
    strict affirmative allowlist either (both have substantial leading
    text before the affirmative-shaped tail), so both land on the safe
    "unclear" default rather than being misread in either direction.
    """
    assert classify_confirmation_reply(message) == "unclear"


def test_negative_takes_priority_over_an_incidental_affirmative_word() -> None:
    assert classify_confirmation_reply("sure, actually no, let's not") == "negative"


def test_hesitation_overrides_a_leading_affirmative_word() -> None:
    assert classify_confirmation_reply("yes, wait, is this the right timezone?") == "unclear"
