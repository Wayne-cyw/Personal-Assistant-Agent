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
        "yes, that works for me!",
        "yes please, book it",
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


def test_negative_takes_priority_over_an_incidental_affirmative_word() -> None:
    assert classify_confirmation_reply("sure, actually no, let's not") == "negative"


def test_hesitation_overrides_a_leading_affirmative_word() -> None:
    assert classify_confirmation_reply("yes, wait, is this the right timezone?") == "unclear"
