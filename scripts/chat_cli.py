"""Interactive terminal REPL client — the stand-in frontend for manual
testing and demos (Issue #7).

Usage:
    python scripts/chat_cli.py [--session SESSION_ID] [--timezone TZ] [--url BASE_URL]
"""

from __future__ import annotations

import argparse
import uuid

import httpx

_TIMEOUT_SECONDS = 90.0  # generous: covers documented free-tier cold starts (Tech Stack)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _render_slots(slots: list[dict[str, object]]) -> str:
    lines = []
    for i, slot in enumerate(slots, start=1):
        label = slot.get("label") or f"{slot.get('start_iso')} - {slot.get('end_iso')}"
        lines.append(f"  {i}. {label}")
    return "\n".join(lines)


def _render_slot_label(slot: object) -> str:
    if isinstance(slot, dict):
        return str(slot.get("label") or f"{slot.get('start_iso')} - {slot.get('end_iso')}")
    return str(slot)


def _render_data(response_type: str, data: dict[str, object] | None) -> str | None:
    """Render the `data` payload for booking-related response types (shapes
    per Engineering Guide 4.8). Returns None when there's nothing extra to
    show (plain messages/refusals).
    """
    if not data:
        return None

    if response_type == "booking_proposal" and isinstance(data.get("slots"), list):
        return "Available times:\n" + _render_slots(data["slots"])  # type: ignore[arg-type]

    if response_type == "booking_confirmation_request":
        return (
            f"Please confirm: {_render_slot_label(data.get('slot'))} ({data.get('timezone')})\n"
            f"  Name: {data.get('name')}  Email: {data.get('email')}"
        )

    if response_type == "booking_confirmed":
        next_steps = data.get("next_steps", "")
        return (
            f"Booking confirmed for {_render_slot_label(data.get('slot'))} "
            f"({data.get('timezone')}), id: {data.get('booking_id')}. {next_steps}"
        ).strip()

    return None


def _build_payload(session_id: str, message: str, timezone: str | None) -> dict[str, object]:
    payload: dict[str, object] = {"session_id": session_id, "message": message}
    if timezone:
        payload["timezone"] = timezone
    return payload


def _format_response(body: dict[str, object]) -> str:
    lines = [str(body.get("reply", ""))]
    data = body.get("data")
    extra = _render_data(
        str(body.get("type", "message")), data if isinstance(data, dict) else None
    )
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def _format_error(body: dict[str, object]) -> str:
    error = body.get("error", {})
    if not isinstance(error, dict):
        error = {}
    code = error.get("code", "unknown")
    message = error.get("message", "An error occurred.")
    return f"error [{code}]: {message}"


def send_turn(client: httpx.Client, session_id: str, message: str, timezone: str | None) -> str:
    """POST one turn to /v1/chat and return the text to print. Every
    failure mode (connection refused, timeout — including the free-tier
    cold starts the Tech Stack table documents, non-JSON response body) is
    caught and formatted here rather than raised, so one bad turn never
    kills the whole REPL session.
    """
    payload = _build_payload(session_id, message, timezone)
    try:
        response = client.post("/v1/chat", json=payload)
    except httpx.ConnectError:
        return f"error: could not connect to {client.base_url} — is the server running?"
    except httpx.HTTPError as exc:
        return f"error: request failed ({type(exc).__name__}: {exc})"

    try:
        body = response.json()
    except ValueError:  # json.JSONDecodeError subclasses ValueError
        return f"error: server returned a non-JSON response (status {response.status_code})"

    if response.status_code >= 400:
        return _format_error(body)
    return _format_response(body)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive chat client for the Personal AI Assistant Agent"
    )
    parser.add_argument("--session", default=None, help="Resume an existing session_id")
    parser.add_argument("--timezone", default=None, help="IANA timezone, e.g. America/Toronto")
    parser.add_argument(
        "--url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the running server (default: {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args()

    session_id = args.session or str(uuid.uuid4())
    print(f"session_id: {session_id}")
    print("Type your message and press Enter. Ctrl+D or 'exit' to quit.\n")

    with httpx.Client(base_url=args.url, timeout=_TIMEOUT_SECONDS) as client:
        while True:
            try:
                message = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not message:
                continue
            if message.lower() in ("exit", "quit"):
                break

            print(send_turn(client, session_id, message, args.timezone))


if __name__ == "__main__":
    main()
