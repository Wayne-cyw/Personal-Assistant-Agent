import app.agent.tokens as tokens_module
from app.agent.tokens import count_tokens


def test_count_tokens_empty_string_is_zero() -> None:
    assert count_tokens("") == 0


def test_count_tokens_nonempty_text_is_positive() -> None:
    assert count_tokens("hello world, this is a test") > 0


def test_count_tokens_longer_text_counts_more_tokens() -> None:
    short = count_tokens("hello")
    long = count_tokens("hello " * 200)
    assert long > short


def test_count_tokens_falls_back_when_tiktoken_unavailable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Regression test: a tokenizer load failure (e.g. no network on first
    use) must degrade to the chars/4 estimate, not crash the caller.
    """
    monkeypatch.setattr(tokens_module, "_encoding", None)
    monkeypatch.setattr(tokens_module, "_encoding_load_failed", True)

    assert count_tokens("") == 0
    assert count_tokens("abcd") == 1
    assert count_tokens("a" * 40) == 10
