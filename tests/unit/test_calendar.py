import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

import app.tools.calendar as calendar_module
from app.tools.calendar import (
    Attendee,
    CalendarError,
    GoogleCalendarClient,
    get_calendar_client,
)


def _dt(hour: int) -> datetime:
    return datetime(2026, 8, 3, hour, 0, tzinfo=UTC)


def test_calendar_error_carries_message_and_optional_status_code() -> None:
    err = CalendarError("free/busy query failed", status_code=503)
    assert err.message == "free/busy query failed"
    assert err.status_code == 503
    assert str(err) == "free/busy query failed"


def test_calendar_error_status_code_defaults_to_none() -> None:
    err = CalendarError("calendar authentication failed")
    assert err.status_code is None


def test_get_calendar_client_returns_a_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(calendar_module, "_client", None)
    first = get_calendar_client()
    second = get_calendar_client()
    assert first is second


class _FakeHttpResponse:
    def __init__(self, status: int) -> None:
        self.status = status
        self.reason = "error"


def _http_error(status: int) -> HttpError:
    return HttpError(resp=_FakeHttpResponse(status), content=b"{}")  # type: ignore[arg-type]


@pytest.fixture
def client() -> GoogleCalendarClient:
    return GoogleCalendarClient()


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(calendar_module, "_client", None)


async def test_get_free_busy_parses_response_into_intervals(
    client: GoogleCalendarClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_service = MagicMock()
    mock_service.freebusy.return_value.query.return_value.execute.return_value = {
        "calendars": {
            "primary": {
                "busy": [
                    {"start": "2026-08-03T09:00:00+00:00", "end": "2026-08-03T09:30:00+00:00"},
                    {"start": "2026-08-03T14:00:00+00:00", "end": "2026-08-03T15:00:00+00:00"},
                ]
            }
        }
    }
    monkeypatch.setattr(client, "_build_service", lambda: mock_service)

    result = await client.get_free_busy(_dt(9), _dt(17))

    assert len(result) == 2
    assert result[0].start == datetime.fromisoformat("2026-08-03T09:00:00+00:00")
    assert result[0].end == datetime.fromisoformat("2026-08-03T09:30:00+00:00")


async def test_get_free_busy_http_error_raises_calendar_error_with_status(
    client: GoogleCalendarClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_service = MagicMock()
    mock_service.freebusy.return_value.query.return_value.execute.side_effect = _http_error(503)
    monkeypatch.setattr(client, "_build_service", lambda: mock_service)

    with pytest.raises(CalendarError) as exc_info:
        await client.get_free_busy(_dt(9), _dt(17))
    assert exc_info.value.status_code == 503


async def test_get_free_busy_auth_error_raises_calendar_error_without_leaking_details(
    client: GoogleCalendarClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: a revoked/expired refresh token raises a
    google-auth RefreshError, which must never propagate raw (it can carry
    request details) — only the sanitized CalendarError.
    """
    mock_service = MagicMock()
    mock_service.freebusy.return_value.query.return_value.execute.side_effect = RefreshError(
        "invalid_grant: token has been expired or revoked"
    )
    monkeypatch.setattr(client, "_build_service", lambda: mock_service)

    with pytest.raises(CalendarError) as exc_info:
        await client.get_free_busy(_dt(9), _dt(17))
    assert "expired or revoked" not in exc_info.value.message
    assert "invalid_grant" not in exc_info.value.message


async def test_create_event_returns_id_and_sets_tentative_status(
    client: GoogleCalendarClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_service = MagicMock()
    mock_service.events.return_value.insert.return_value.execute.return_value = {
        "id": "event-abc123"
    }
    monkeypatch.setattr(client, "_build_service", lambda: mock_service)

    event_id = await client.create_event(
        _dt(10), _dt(11), Attendee(name="Priya", email="priya@example.com"), "Intro call"
    )

    assert event_id == "event-abc123"
    _, kwargs = mock_service.events.return_value.insert.call_args
    assert kwargs["body"]["status"] == "tentative"
    assert kwargs["body"]["attendees"] == [
        {"email": "priya@example.com", "displayName": "Priya"}
    ]


async def test_create_event_http_error_raises_calendar_error(
    client: GoogleCalendarClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_service = MagicMock()
    mock_service.events.return_value.insert.return_value.execute.side_effect = _http_error(409)
    monkeypatch.setattr(client, "_build_service", lambda: mock_service)

    with pytest.raises(CalendarError) as exc_info:
        await client.create_event(
            _dt(10), _dt(11), Attendee(name="Priya", email="priya@example.com"), "Intro call"
        )
    assert exc_info.value.status_code == 409


async def test_delete_event_calls_the_right_api_method(
    client: GoogleCalendarClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_service = MagicMock()
    monkeypatch.setattr(client, "_build_service", lambda: mock_service)

    await client.delete_event("event-abc123")

    mock_service.events.return_value.delete.assert_called_once()
    _, kwargs = mock_service.events.return_value.delete.call_args
    assert kwargs["eventId"] == "event-abc123"


async def test_delete_event_http_error_raises_calendar_error(
    client: GoogleCalendarClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_service = MagicMock()
    mock_service.events.return_value.delete.return_value.execute.side_effect = _http_error(404)
    monkeypatch.setattr(client, "_build_service", lambda: mock_service)

    with pytest.raises(CalendarError) as exc_info:
        await client.delete_event("does-not-exist")
    assert exc_info.value.status_code == 404


# --- naive-datetime rejection -----------------------------------------------


async def test_get_free_busy_rejects_naive_start(client: GoogleCalendarClient) -> None:
    naive = datetime(2026, 8, 3, 9, 0)  # no tzinfo
    with pytest.raises(ValueError, match="timezone-aware"):
        await client.get_free_busy(naive, _dt(17))


async def test_get_free_busy_rejects_naive_end(client: GoogleCalendarClient) -> None:
    naive = datetime(2026, 8, 3, 17, 0)  # no tzinfo
    with pytest.raises(ValueError, match="timezone-aware"):
        await client.get_free_busy(_dt(9), naive)


async def test_create_event_rejects_naive_datetimes(client: GoogleCalendarClient) -> None:
    naive = datetime(2026, 8, 3, 10, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        await client.create_event(
            naive, _dt(11), Attendee(name="Priya", email="priya@example.com"), "Intro call"
        )


# --- malformed responses degrade to CalendarError, not a raw KeyError -------


async def test_get_free_busy_missing_busy_key_raises_calendar_error(
    client: GoogleCalendarClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-calendar "errors" entry (permission/notFound) instead of
    "busy" is a real Google API response shape — must not surface as a raw
    KeyError past this module's own CalendarError contract.
    """
    mock_service = MagicMock()
    mock_service.freebusy.return_value.query.return_value.execute.return_value = {
        "calendars": {"primary": {"errors": [{"domain": "global", "reason": "notFound"}]}}
    }
    monkeypatch.setattr(client, "_build_service", lambda: mock_service)

    with pytest.raises(CalendarError):
        await client.get_free_busy(_dt(9), _dt(17))


async def test_create_event_missing_id_in_response_raises_calendar_error(
    client: GoogleCalendarClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_service = MagicMock()
    mock_service.events.return_value.insert.return_value.execute.return_value = {
        "status": "confirmed"  # no "id"
    }
    monkeypatch.setattr(client, "_build_service", lambda: mock_service)

    with pytest.raises(CalendarError):
        await client.create_event(
            _dt(10), _dt(11), Attendee(name="Priya", email="priya@example.com"), "Intro call"
        )


# --- non-blocking (Engineering Guide 4.7 rule 1, applied to this SDK) ------


async def test_get_free_busy_does_not_block_the_event_loop(
    client: GoogleCalendarClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole sync call chain (service construction through .execute())
    must run in a worker thread via asyncio.to_thread — a concurrent, fast
    asyncio task must not be stalled behind a slow (blocking, synchronous)
    Calendar API call.
    """

    def _slow_build_service() -> MagicMock:
        time.sleep(0.2)  # simulates a slow blocking network call
        mock_service = MagicMock()
        mock_service.freebusy.return_value.query.return_value.execute.return_value = {
            "calendars": {"primary": {"busy": []}}
        }
        return mock_service

    monkeypatch.setattr(client, "_build_service", _slow_build_service)

    async def fast_task() -> float:
        start = time.monotonic()
        await asyncio.sleep(0.01)
        return time.monotonic() - start

    _, fast_duration = await asyncio.gather(client.get_free_busy(_dt(9), _dt(17)), fast_task())

    assert fast_duration < 0.1  # nowhere near the 0.2s blocking call's duration
