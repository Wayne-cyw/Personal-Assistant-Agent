"""Owner notification (Issue #22, Engineering Guide 4.5/PRD: "Notify the
owner when a booking is created" — this same interface also covers the
flagged-conversation notification Issue #25's abusive-message handling
needs).

A narrow `Notifier` Protocol, mirroring `app/agent/providers`' and
`app/tools/calendar.py`'s "one factory, one Protocol, never the concrete
implementation directly" pattern — so the transport (a webhook today,
maybe email later) is swappable without touching call sites. Webhook, not
email: "whichever is simpler" per the issue text, and `httpx` is already a
dependency with no SMTP credentials/templating needed.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class Notifier(Protocol):
    async def notify(self, subject: str, body: str) -> None: ...


class WebhookNotifier:
    """POSTs `{"subject", "body"}` as JSON to `settings.owner_notify_
    webhook_url`. A notification failure is logged, never raised — by the
    time this runs, the event it's reporting on (a booking, a flagged
    session) is already durably persisted, so a dead webhook must not turn
    into a failure for the turn that triggered it.
    """

    async def notify(self, subject: str, body: str) -> None:
        if not settings.owner_notify_webhook_url:
            logger.warning("owner notification skipped: no webhook URL configured")
            return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    settings.owner_notify_webhook_url, json={"subject": subject, "body": body}
                )
            response.raise_for_status()
        except httpx.HTTPError:
            logger.error("owner notification webhook call failed")


_notifier: Notifier | None = None


def get_notifier() -> Notifier:
    """The only supported way to obtain a notifier — call sites depend on
    this factory plus this module's types, never `WebhookNotifier`
    directly. One instance reused for the life of the process.
    """
    global _notifier
    if _notifier is None:
        _notifier = WebhookNotifier()
    return _notifier
