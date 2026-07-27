from app.agent.visitor_info import extract_linkedin_url, extract_name


def test_extract_name_my_name_is() -> None:
    assert extract_name("my name is Sam") == "Sam"


def test_extract_name_my_names() -> None:
    assert extract_name("my name's Sam Rivera") == "Sam Rivera"


def test_extract_name_none_when_absent() -> None:
    assert extract_name("what can you help with?") is None


def test_extract_name_does_not_match_call_me() -> None:
    """Deliberately narrow: "call me X" is not one of the recognized
    patterns — conservative false-negatives are fine (Issue #9).
    """
    assert extract_name("call me Sam") is None


def test_extract_name_does_not_match_im_x() -> None:
    """"I'm X" is deliberately NOT matched — it has an unacceptably high
    false-positive rate in ordinary conversation on this exact site (see
    the module docstring): "I'm interested in booking a call" would
    otherwise capture "Interested" as a name.
    """
    assert extract_name("I'm Alex") is None
    assert extract_name("I am Jordan Lee") is None


def test_extract_name_common_im_phrasings_do_not_produce_false_positives() -> None:
    false_positive_prone = [
        "I'm Interested in booking a call",
        "I'm Curious about your projects",
        "I'm New here, just exploring",
        "I'm Looking for a backend engineer",
        "I'm Currently Exploring Options",
    ]
    for message in false_positive_prone:
        assert extract_name(message) is None, f"false positive on: {message!r}"


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


def test_extract_linkedin_url_scheme_less_paste_at_start() -> None:
    assert extract_linkedin_url("linkedin.com/in/jane-doe") == "linkedin.com/in/jane-doe"


def test_extract_linkedin_url_scheme_less_paste_after_whitespace() -> None:
    text = "here's my profile linkedin.com/in/jane-doe thanks"
    assert extract_linkedin_url(text) == "linkedin.com/in/jane-doe"


def test_extract_linkedin_url_locale_subdomain() -> None:
    assert extract_linkedin_url("https://uk.linkedin.com/in/jane-doe") == (
        "https://uk.linkedin.com/in/jane-doe"
    )


def test_extract_linkedin_url_with_query_string() -> None:
    text = "https://www.linkedin.com/in/jane-doe?originalSubdomain=uk"
    assert extract_linkedin_url(text) == text


def test_extract_linkedin_url_scheme_less_still_rejects_embedded_spoof() -> None:
    """The scheme-less pattern only matches at start-of-message or after
    whitespace — "evil.com/linkedin.com/in/fake" has "linkedin.com"
    preceded by "/", not whitespace or start, so it must not match.
    """
    assert extract_linkedin_url("visit evil.com/linkedin.com/in/fake now") is None
