from app.safety.pii import redact


def test_redacts_openai_style_key() -> None:
    assert redact("key is sk-abcdefghijklmnop123") == "key is [REDACTED]"


def test_redacts_google_style_key() -> None:
    assert redact("key is AIzaSyAbCdEfGhIjKlMnOpQrSt123") == "key is [REDACTED]"


def test_redacts_bearer_token() -> None:
    assert redact("Authorization: Bearer abc123def456ghi789") == "Authorization: [REDACTED]"


def test_redacts_bearer_token_case_insensitive() -> None:
    assert redact("bearer abc123def456ghi789") == "[REDACTED]"


def test_redacts_db_connection_string_credentials_keeping_host_visible() -> None:
    result = redact("connecting to postgresql+asyncpg://user:hunter2@db.example.com/agent")
    assert "hunter2" not in result
    assert "user" not in result
    assert "db.example.com" in result
    assert result == "connecting to postgresql+asyncpg://[REDACTED]@db.example.com/agent"


def test_leaves_ordinary_text_unchanged() -> None:
    text = "GET /v1/chat returned 200 in 45ms for session sess-abc-123"
    assert redact(text) == text


def test_redacts_multiple_occurrences() -> None:
    text = "key1=sk-aaaaaaaaaaaaaaaaaa key2=sk-bbbbbbbbbbbbbbbbbb"
    result = redact(text)
    assert "sk-aaaaaaaaaaaaaaaaaa" not in result
    assert "sk-bbbbbbbbbbbbbbbbbb" not in result
