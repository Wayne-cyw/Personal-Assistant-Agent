"""The Google Calendar client wrapper (Engineering Guide Tech Stack, Issue
#17) — authenticated, narrowly-scoped calendar access behind a Protocol the
rest of the code depends on, never the raw Google SDK directly.

The `calendar_find_slots`/`calendar_create_booking` *tools* that call this
wrapper are a separate concern, built out starting in Issues #19/#22 once
the booking state machine (#18) exists to gate them.

`google-api-python-client` is synchronous (blocking HTTP calls) — there is
no official async client. Every `GoogleCalendarClient` method below is
`async def` but runs the blocking call via `asyncio.to_thread(...)` so it
never blocks the event loop, the same non-blocking-I/O discipline
Engineering Guide 4.7 rule 1 applies to the database.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from google.auth.exceptions import GoogleAuthError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.config import settings

logger = logging.getLogger(__name__)

# calendar.events: create/read/update/delete events on the one configured
# calendar — deliberately not the broader `calendar` scope (Issue #17's
# "scope minimally" task), since this app never needs to read the owner's
# event details/attendee list, only free/busy and its own bookings.
_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
_TOKEN_URI = "https://oauth2.googleapis.com/token"


class CalendarError(Exception):
    """Raised for any failed Calendar API call. Constructed with only a
    fixed, sanitized message (and, where available, an HTTP status code) —
    never the raw google-api-python-client/google-auth exception, which can
    carry request/response details (Engineering Guide 4.8's rule: no raw
    exception content ever reaches a log or a caller).
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


@dataclass
class BusyInterval:
    start: datetime
    end: datetime


@dataclass
class Attendee:
    name: str
    email: str


class CalendarClient(Protocol):
    async def get_free_busy(self, start: datetime, end: datetime) -> list[BusyInterval]: ...

    async def create_event(
        self, start: datetime, end: datetime, attendee: Attendee, description: str
    ) -> str: ...

    async def delete_event(self, event_id: str) -> None: ...


def _status_code_from(exc: HttpError) -> int | None:
    try:
        return int(exc.resp.status)
    except (AttributeError, TypeError, ValueError):
        return None


def _require_aware(value: datetime, name: str) -> None:
    # Same naive-datetime guard app/db/models.py applies to DB writes
    # (Engineering Guide 4.7 rule 2) — a naive datetime silently means
    # different absolute instants depending on which timezone the caller
    # assumed, and the booking flow (Issues #19-22) that will call this
    # wrapper is timezone-sensitive by design.
    if value.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware datetime, got a naive one")


class GoogleCalendarClient:
    """The real implementation, wrapping `google-api-python-client`."""

    def _credentials(self) -> Credentials:
        # token=None with a refresh_token/client_id/client_secret/token_uri
        # set is enough for the SDK to fetch a fresh access token on first
        # use and re-refresh automatically whenever it expires — the "auto
        # token refresh" requirement, with no extra code needed here.
        return Credentials(  # type: ignore[no-untyped-call]
            token=None,
            refresh_token=settings.google_refresh_token,
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            token_uri=_TOKEN_URI,
            scopes=_SCOPES,
        )

    def _build_service(self) -> object:
        # cache_discovery=False: the discovery-doc file cache this library
        # defaults to writes to disk and warns on read-only filesystems
        # (exactly this app's production deploy target, Engineering Guide
        # Tech Stack) — unneeded here since the API surface used is tiny.
        return build("calendar", "v3", credentials=self._credentials(), cache_discovery=False)

    def _get_free_busy_sync(self, start: datetime, end: datetime) -> list[BusyInterval]:
        service = self._build_service()
        body = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "items": [{"id": settings.google_calendar_id}],
        }
        try:
            response = service.freebusy().query(body=body).execute()  # type: ignore[attr-defined]
        except HttpError as exc:
            status = _status_code_from(exc)
            logger.warning("free/busy query failed (status=%s)", status)
            raise CalendarError("free/busy query failed", status_code=status) from None
        except GoogleAuthError:
            logger.warning("calendar authentication failed during free/busy query")
            raise CalendarError("calendar authentication failed") from None
        try:
            busy_raw = response["calendars"][settings.google_calendar_id]["busy"]
            return [
                BusyInterval(
                    start=datetime.fromisoformat(item["start"]),
                    end=datetime.fromisoformat(item["end"]),
                )
                for item in busy_raw
            ]
        except (KeyError, TypeError, ValueError):
            # A per-calendar "errors" entry (permission/notFound) instead
            # of "busy" hits this — a malformed/unexpected response shape
            # is a failure the same as an HTTP error, not something that
            # should propagate as a raw KeyError past this module's own
            # CalendarError contract.
            logger.warning(
                "free/busy response for calendar %s was missing or malformed",
                settings.google_calendar_id,
            )
            raise CalendarError("free/busy response was missing or malformed") from None

    async def get_free_busy(self, start: datetime, end: datetime) -> list[BusyInterval]:
        _require_aware(start, "start")
        _require_aware(end, "end")
        return await asyncio.to_thread(self._get_free_busy_sync, start, end)

    def _create_event_sync(
        self, start: datetime, end: datetime, attendee: Attendee, description: str
    ) -> str:
        service = self._build_service()
        body = {
            "summary": f"Call with {attendee.name}",
            "description": description,
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
            "attendees": [{"email": attendee.email, "displayName": attendee.name}],
            "status": "tentative",
        }
        try:
            event = (
                service.events()  # type: ignore[attr-defined]
                .insert(calendarId=settings.google_calendar_id, body=body)
                .execute()
            )
        except HttpError as exc:
            status = _status_code_from(exc)
            logger.warning("event creation failed (status=%s)", status)
            raise CalendarError("event creation failed", status_code=status) from None
        except GoogleAuthError:
            logger.warning("calendar authentication failed during event creation")
            raise CalendarError("calendar authentication failed") from None
        try:
            event_id: str = event["id"]
        except (KeyError, TypeError):
            logger.warning("event creation response was missing an id")
            raise CalendarError("event creation response was missing an id") from None
        return event_id

    async def create_event(
        self, start: datetime, end: datetime, attendee: Attendee, description: str
    ) -> str:
        _require_aware(start, "start")
        _require_aware(end, "end")
        return await asyncio.to_thread(self._create_event_sync, start, end, attendee, description)

    def _delete_event_sync(self, event_id: str) -> None:
        service = self._build_service()
        try:
            service.events().delete(  # type: ignore[attr-defined]
                calendarId=settings.google_calendar_id, eventId=event_id
            ).execute()
        except HttpError as exc:
            status = _status_code_from(exc)
            logger.warning("event deletion failed (status=%s)", status)
            raise CalendarError("event deletion failed", status_code=status) from None
        except GoogleAuthError:
            logger.warning("calendar authentication failed during event deletion")
            raise CalendarError("calendar authentication failed") from None

    async def delete_event(self, event_id: str) -> None:
        await asyncio.to_thread(self._delete_event_sync, event_id)


_client: CalendarClient | None = None


def get_calendar_client() -> CalendarClient:
    """The only supported way to obtain a calendar client — call sites
    depend on this factory plus this module's types, never
    GoogleCalendarClient directly, mirroring app/agent/providers'
    get_provider() pattern. One instance reused for the life of the
    process (this app runs as a single long-lived worker, Engineering
    Guide 4.3).
    """
    global _client
    if _client is None:
        _client = GoogleCalendarClient()
    return _client
