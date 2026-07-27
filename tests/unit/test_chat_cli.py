import json

import httpx

from scripts.chat_cli import (
    _build_payload,
    _format_error,
    _format_response,
    _render_data,
    send_turn,
)


def test_build_payload_without_timezone() -> None:
    assert _build_payload("sess-1", "hello", None) == {
        "session_id": "sess-1",
        "message": "hello",
    }


def test_build_payload_with_timezone() -> None:
    assert _build_payload("sess-1", "hello", "America/Toronto") == {
        "session_id": "sess-1",
        "message": "hello",
        "timezone": "America/Toronto",
    }


def test_format_response_plain_message() -> None:
    body: dict[str, object] = {"reply": "Hi there!", "type": "message", "data": None}
    assert _format_response(body) == "Hi there!"


def test_render_data_booking_proposal_as_numbered_list() -> None:
    slot1 = {
        "slot_id": "1",
        "start_iso": "2026-08-01T10:00:00",
        "end_iso": "2026-08-01T10:30:00",
        "label": "Aug 1, 10:00 AM",
    }
    slot2 = {
        "slot_id": "2",
        "start_iso": "2026-08-01T14:00:00",
        "end_iso": "2026-08-01T14:30:00",
        "label": "Aug 1, 2:00 PM",
    }
    data = {"slots": [slot1, slot2], "round": 1}
    rendered = _render_data("booking_proposal", data)
    assert rendered == "Available times:\n  1. Aug 1, 10:00 AM\n  2. Aug 1, 2:00 PM"


def test_render_data_booking_confirmed_includes_slot_and_timezone() -> None:
    data: dict[str, object] = {
        "booking_id": "42",
        "slot": {"start_iso": "2026-08-01T10:00:00", "label": "Aug 1, 10:00 AM"},
        "timezone": "America/Toronto",
        "next_steps": "Check your email.",
    }
    rendered = _render_data("booking_confirmed", data)
    assert rendered == (
        "Booking confirmed for Aug 1, 10:00 AM (America/Toronto), id: 42. Check your email."
    )


def test_render_data_booking_confirmation_request_uses_slot_label_not_raw_dict() -> None:
    data: dict[str, object] = {
        "slot": {
            "slot_id": "1",
            "start_iso": "2026-08-01T10:00:00",
            "end_iso": "2026-08-01T10:30:00",
            "label": "Aug 1, 10:00 AM",
        },
        "timezone": "America/Toronto",
        "name": "Sam",
        "email": "sam@example.com",
    }
    rendered = _render_data("booking_confirmation_request", data)
    assert rendered is not None
    assert "{'slot_id'" not in rendered  # must not print a raw dict repr
    assert "Aug 1, 10:00 AM" in rendered
    assert "Sam" in rendered
    assert "sam@example.com" in rendered


def test_render_data_none_for_plain_message() -> None:
    assert _render_data("message", None) is None
    assert _render_data("refusal", None) is None


def test_format_response_includes_rendered_data() -> None:
    body: dict[str, object] = {
        "reply": "Here are some times:",
        "type": "booking_proposal",
        "data": {"slots": [{"start_iso": "x", "end_iso": "y"}]},
    }
    formatted = _format_response(body)
    assert formatted.startswith("Here are some times:\n")
    assert "1. x - y" in formatted


def test_format_error_known_code() -> None:
    body: dict[str, object] = {"error": {"code": "rate_limited", "message": "Too many requests."}}
    assert _format_error(body) == "error [rate_limited]: Too many requests."


def test_format_error_missing_fields_falls_back() -> None:
    assert _format_error({}) == "error [unknown]: An error occurred."


def test_send_turn_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat"
        return httpx.Response(200, json={"reply": "hello back", "type": "message", "data": None})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    result = send_turn(client, "sess-1", "hi", None)
    assert result == "hello back"


def test_send_turn_sends_timezone_when_provided() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"reply": "ok", "type": "message", "data": None})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    send_turn(client, "sess-1", "hi", "America/Toronto")
    assert captured["timezone"] == "America/Toronto"


def test_send_turn_error_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422, json={"error": {"code": "invalid_request", "message": "The request was invalid."}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    result = send_turn(client, "sess-1", "", None)
    assert result == "error [invalid_request]: The request was invalid."


def test_send_turn_connect_error_formatted_not_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    result = send_turn(client, "sess-1", "hi", None)  # must not raise
    assert "could not connect" in result
    assert "is the server running" in result


def test_send_turn_timeout_formatted_not_raised() -> None:
    """A read timeout (e.g. a free-tier cold start per the Tech Stack table)
    must not kill the whole REPL session.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    result = send_turn(client, "sess-1", "hi", None)  # must not raise
    assert result.startswith("error: request failed")


def test_send_turn_non_json_response_formatted_not_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, content=b"<html>Bad Gateway</html>")

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    result = send_turn(client, "sess-1", "hi", None)  # must not raise
    assert "non-JSON response" in result
    assert "502" in result


def test_send_turn_missing_reply_field_does_not_crash() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"type": "message", "data": None})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    result = send_turn(client, "sess-1", "hi", None)  # must not raise KeyError
    assert result == ""
