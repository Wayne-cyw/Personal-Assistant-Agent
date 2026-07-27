from pathlib import Path

import pytest

from app.agent.intro import INTRO_MESSAGE, MAX_LENGTH, _load_intro_message


def test_intro_message_matches_file_contents() -> None:
    raw = Path(__file__).parents[2] / "knowledge" / "intro.md"
    assert INTRO_MESSAGE == raw.read_text(encoding="utf-8").strip()


def test_intro_message_is_nonempty_and_under_cap() -> None:
    assert len(INTRO_MESSAGE) > 0
    assert len(INTRO_MESSAGE) <= MAX_LENGTH


def test_missing_file_fails_loudly(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.md"
    with pytest.raises(RuntimeError, match="Missing required file"):
        _load_intro_message(missing)


def test_empty_file_fails_loudly(tmp_path: Path) -> None:
    empty = tmp_path / "intro.md"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty"):
        _load_intro_message(empty)


def test_oversized_file_fails_loudly(tmp_path: Path) -> None:
    oversized = tmp_path / "intro.md"
    oversized.write_text("x" * (MAX_LENGTH + 1), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exceeding"):
        _load_intro_message(oversized)


def test_file_at_exactly_the_cap_is_accepted(tmp_path: Path) -> None:
    at_cap = tmp_path / "intro.md"
    at_cap.write_text("x" * MAX_LENGTH, encoding="utf-8")
    assert _load_intro_message(at_cap) == "x" * MAX_LENGTH
