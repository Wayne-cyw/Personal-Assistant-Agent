import logging

import pytest

from app.config import settings
from app.main import _check_single_worker


def test_no_web_concurrency_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "web_concurrency", None)
    _check_single_worker()  # must not raise


def test_web_concurrency_one_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "web_concurrency", 1)
    _check_single_worker()  # must not raise


def test_multiple_workers_logs_warning_by_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "web_concurrency", 4)
    monkeypatch.setattr(settings, "enforce_single_worker", False)

    with caplog.at_level(logging.ERROR):
        _check_single_worker()  # must not raise when enforcement is off

    assert any("WEB_CONCURRENCY=4" in r.getMessage() for r in caplog.records)


def test_multiple_workers_raises_when_enforcement_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "web_concurrency", 2)
    monkeypatch.setattr(settings, "enforce_single_worker", True)

    with pytest.raises(RuntimeError, match="WEB_CONCURRENCY"):
        _check_single_worker()
