import app.agent.tokens as tokens_module
from app.agent.tokens import count_tokens, effective_tokens


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


def test_effective_tokens_with_no_caching_is_a_plain_sum() -> None:
    result = effective_tokens(input_tokens=100, cached_input_tokens=0, output_tokens=50)
    assert result == 150


def test_effective_tokens_discounts_cached_input_at_ten_percent() -> None:
    # 100 input tokens, 80 of them cached: 20 uncached (full price) + 80 *
    # 0.1 (discounted) + 50 output = 20 + 8 + 50 = 78.
    result = effective_tokens(input_tokens=100, cached_input_tokens=80, output_tokens=50)
    assert result == 78


def test_effective_tokens_fully_cached_input_is_mostly_discounted() -> None:
    result = effective_tokens(input_tokens=1000, cached_input_tokens=1000, output_tokens=0)
    assert result == 100  # 1000 * 0.1
