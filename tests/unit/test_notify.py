"""app/notify.py — the owner-notification interface (Issue #22).

Monkeypatches httpx.AsyncClient.post directly to intercept the outbound
call rather than hitting a real network endpoint, mirroring how
app/tools/calendar.py's tests mock the Google API client instead of
making real calls.
"""

import logging

import httpx
import pytest

from app.config import settings
from app.notify import WebhookNotifier, get_notifier


async def test_notify_posts_subject_and_body_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "owner_notify_webhook_url", "https://example.com/hook")
    calls: list[tuple[str, dict[str, object]]] = []

    async def fake_post(
        self: httpx.AsyncClient, url: str, *, json: dict[str, object]
    ) -> httpx.Response:
        calls.append((url, json))
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    await WebhookNotifier().notify("New booking", "Priya booked Tue Aug 4, 2:00 PM EDT")

    assert calls == [
        (
            "https://example.com/hook",
            {"subject": "New booking", "body": "Priya booked Tue Aug 4, 2:00 PM EDT"},
        )
    ]


async def test_notify_no_ops_when_webhook_url_not_configured(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "owner_notify_webhook_url",None)
    called = False

    async def fake_post(self: httpx.AsyncClient, *args: object, **kwargs: object) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with caplog.at_level(logging.WARNING):
        await WebhookNotifier().notify("subject", "body")

    assert called is False
    assert any("no webhook url configured" in r.getMessage().lower() for r in caplog.records)


async def test_notify_failure_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "owner_notify_webhook_url","https://example.com/hook")

    async def failing_post(
        self: httpx.AsyncClient, *args: object, **kwargs: object
    ) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", failing_post)

    with caplog.at_level(logging.ERROR):
        await WebhookNotifier().notify("subject", "body")  # must not raise

    assert any("webhook" in r.getMessage().lower() for r in caplog.records)


async def test_notify_failure_on_non_2xx_status_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "owner_notify_webhook_url","https://example.com/hook")

    async def error_post(
        self: httpx.AsyncClient, *args: object, **kwargs: object
    ) -> httpx.Response:
        return httpx.Response(500, request=httpx.Request("POST", "https://example.com/hook"))

    monkeypatch.setattr(httpx.AsyncClient, "post", error_post)

    with caplog.at_level(logging.ERROR):
        await WebhookNotifier().notify("subject", "body")  # must not raise

    assert any("webhook" in r.getMessage().lower() for r in caplog.records)


def test_get_notifier_returns_the_same_instance_across_calls() -> None:
    first = get_notifier()
    second = get_notifier()
    assert first is second
