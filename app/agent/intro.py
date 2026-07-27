"""The deterministic turn-zero prefix message (Issue #9).

Loaded verbatim from `knowledge/intro.md` at import time — not generated,
not templated by the LLM. Loading here (module import, triggered from
`app/api/chat.py`, which is imported before the app starts serving) means a
missing/invalid file fails the whole process at startup, per the issue's
"startup fails loudly if missing" requirement.
"""

from __future__ import annotations

from pathlib import Path

MAX_LENGTH = 1200

_INTRO_PATH = Path(__file__).resolve().parents[2] / "knowledge" / "intro.md"


def _load_intro_message(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(
            f"Missing required file: {path}. The owner must author "
            "knowledge/intro.md (the turn-zero prefix message) before the "
            "app can start."
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"{path} is empty. The turn-zero prefix message cannot be blank.")
    if len(text) > MAX_LENGTH:
        raise RuntimeError(
            f"{path} is {len(text)} characters, exceeding the {MAX_LENGTH}-character "
            "cap for the turn-zero prefix message."
        )
    return text


INTRO_MESSAGE = _load_intro_message(_INTRO_PATH)
