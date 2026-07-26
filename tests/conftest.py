"""Shared test fixtures.

app.config.settings is validated at import time, so the required env vars
must exist before any test module imports app.config for collection to
succeed. Individual tests that exercise config validation itself construct
Settings directly (with _env_file=None) rather than relying on this.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
