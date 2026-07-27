"""Shared test fixtures.

app.config.settings is validated at import time, so the required env vars
must exist before any test module imports app.config for collection to
succeed. Individual tests that exercise config validation itself construct
Settings directly (with _env_file=None) rather than relying on this.
"""

import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OWNER_CONTACT_EMAIL", "owner@example.com")


@pytest.fixture(autouse=True)
def _reset_session_locks() -> None:
    """app.agent.session_lock._locks is process-global, keyed by session_id
    — and different test *files* reuse the same session ids (e.g. "sess-1")
    across their own independent event loops (pytest-asyncio, function
    scope). An asyncio.Lock created under one test's loop raises
    RuntimeError if reused under a later test's loop, so this clears the
    registry before every test in the whole suite. Real deployments never
    hit this — the process has exactly one event loop for its entire
    lifetime (Engineering Guide 4.3) — it's purely a test-isolation
    artifact of reusing session ids across tests.
    """
    import app.agent.session_lock as session_lock_module

    session_lock_module._locks.clear()
