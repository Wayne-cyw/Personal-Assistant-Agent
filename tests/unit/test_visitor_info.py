from app.agent.visitor_info import extract_linkedin_url, extract_name


def test_extract_name_my_name_is() -> None:
    assert extract_name("my name is Sam") == "Sam"


def test_extract_name_my_names() -> None:
    assert extract_name("my name's Sam Rivera") == "Sam Rivera"


def test_extract_name_im() -> None:
    assert extract_name("I'm Alex") == "Alex"


def test_extract_name_i_am() -> None:
    assert extract_name("I am Jordan Lee") == "Jordan Lee"


def test_extract_name_none_when_absent() -> None:
    assert extract_name("what can you help with?") is None


def test_extract_name_does_not_match_call_me() -> None:
    """Deliberately narrow: "call me X" is not one of the recognized
    patterns — conservative false-negatives are fine (Issue #9).
    """
    assert extract_name("call me Sam") is None


def test_extract_name_lowercase_word_not_captured() -> None:
    # "i'm not sure" — "not" isn't capitalized, so it wouldn't look like a name anyway,
    # but this also guards against matching filler words after "I'm".
    assert extract_name("i'm not sure what you mean") is None


def test_extract_linkedin_url_basic() -> None:
    text = "here's my profile: https://www.linkedin.com/in/jane-doe"
    assert extract_linkedin_url(text) == "https://www.linkedin.com/in/jane-doe"


def test_extract_linkedin_url_without_www() -> None:
    assert extract_linkedin_url("https://linkedin.com/in/jdoe123") == "https://linkedin.com/in/jdoe123"


def test_extract_linkedin_url_trailing_slash() -> None:
    assert extract_linkedin_url("https://www.linkedin.com/in/jane-doe/") == (
        "https://www.linkedin.com/in/jane-doe/"
    )


def test_extract_linkedin_url_none_for_other_linkedin_paths() -> None:
    # /company/... is not a personal profile — Issue #9 asks specifically
    # for LinkedIn *profile* URL validation.
    assert extract_linkedin_url("https://www.linkedin.com/company/acme") is None


def test_extract_linkedin_url_none_when_absent() -> None:
    assert extract_linkedin_url("just saying hi") is None


def test_extract_linkedin_url_rejects_non_linkedin_domain() -> None:
    assert extract_linkedin_url("https://evil.com/linkedin.com/in/fake") is None
