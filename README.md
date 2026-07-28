# Personal AI Assistant Agent

A backend-only conversational AI agent that acts as a digital stand-in for its owner: answers questions about the owner via RAG, books calls on the owner's Google Calendar through a deterministic state machine, and refuses everything else.

Full architecture, data model, API contract, and build plan: [`Personal_AI_Assistant_Agent_Engineering_Guide.md`](Personal_AI_Assistant_Agent_Engineering_Guide.md). Product requirements: [`Personal_AI_Assistant_Agent_PRD_v2.md`](Personal_AI_Assistant_Agent_PRD_v2.md).

## Development

```
pip install -e ".[dev]"
pre-commit install
make lint
make test
make run
```

Copy `.env.example` to `.env` and fill in values before running locally.

### Testing against local Postgres

Unit tests run against a temp SQLite file by default. `test_tables_land_as_postgres_native_types` (dialect parity — the real CI parity job lands in Issue #31) is skipped unless `POSTGRES_TEST_URL` is set:

```
docker run --rm -d --name personal-agent-pg \
  -e POSTGRES_USER=agent -e POSTGRES_PASSWORD=agent -e POSTGRES_DB=agent \
  -p 5432:5432 postgres:16

POSTGRES_TEST_URL=postgresql+asyncpg://agent:agent@localhost:5432/agent make test
```

### Google Calendar setup

The calendar wrapper (`app/tools/calendar.py`, Issue #17) needs OAuth credentials for the owner's own Google account — this is the fiddliest part of the whole build, so follow these steps in order.

1. **Create a Google Cloud project.** [console.cloud.google.com](https://console.cloud.google.com) → new project (any name).
2. **Enable the Calendar API.** In the project, go to "APIs & Services" → "Library" → search "Google Calendar API" → Enable.
3. **Configure the OAuth consent screen.** "APIs & Services" → "OAuth consent screen". User type "External" is fine for a single-user tool like this (you'll be the only person who ever authorizes it) — Google will show an "unverified app" warning during the auth flow below; that's expected and safe to click through, since you're authorizing your own app to access your own account. Add your own Google account under "Test users" (required while the app is in "Testing" publish status, which is fine indefinitely at this scale — no need to submit for verification).
4. **Create an OAuth Client ID.** "APIs & Services" → "Credentials" → "Create Credentials" → "OAuth client ID" → Application type **"Desktop app"**. Note the client ID and client secret it gives you.
5. **Run the one-time local auth flow** to get a refresh token:
   ```
   python scripts/gcal_auth.py --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET
   ```
   This opens a browser, has you sign in with the Google account whose calendar the agent should book on, and shows an "unverified app" warning (click "Advanced" → "Go to (app name) (unsafe)" — safe here, since it's your own OAuth client). On success it prints a refresh token.
6. **Set the three env vars** in `.env`: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` (from step 4), `GOOGLE_REFRESH_TOKEN` (from step 5). `GOOGLE_CALENDAR_ID` defaults to `primary` (the signed-in account's main calendar) — only set it if booking against a different, dedicated calendar.
7. **Verify it works:**
   ```
   python scripts/gcal_smoke.py
   ```
   Prints real free/busy for the next 7 days, then creates and deletes a test tentative event. `GET /health` also reports `calendar: "ok"` once these are configured (cached 5 minutes — revoking access in your [Google account permissions](https://myaccount.google.com/permissions) flips it to `"error"` within that window).

**Scope note:** the OAuth client only ever requests `calendar.events` (not full calendar access) — the app can read free/busy and create/delete events it made, nothing else.

### Model notes

**`gpt-5.6-luna` function-calling reliability (Issue #10):** the orchestration loop (`app/agent/loop.py`) and its automated tests (`tests/unit/test_loop.py`) are fully verified against `FakeProvider`, which doesn't exercise real model behavior. No `OPENAI_API_KEY` was available in the environment this issue was built in, so Luna's actual tool-calling reliability at this budget tier has **not yet been observed against a live model** — the CLI acceptance check ("what day is it?" → tool call → correct date) and the config-only escalation path to `gpt-5.6-terra` (`LLM_MODEL=gpt-5.6-terra` for the main loop only, per the Tech Stack table) are both implemented and ready to exercise, but still need a real run before this note can be replaced with actual observations. Whoever runs that first live session should update this note with what they saw (successful/malformed tool calls, retry behavior, anything else worth flagging) rather than leave it as a placeholder.

**Conversation memory (Issue #11):** same `FakeProvider`-only caveat applies to the summarizer (`app/agent/memory.py`) — the eviction/summarization prompt has never run against a live model, so its actual JSON-compliance rate, summary quality, and retry-then-skip frequency at this budget tier are unobserved. Two things are deliberately out of scope for this issue, not oversights:

- **Reconciliation trigger 1** (explicit correction detected via the input classifier's label set, Engineering Guide 4.3) is deferred to Issue #25, which isn't a dependency of #11 and isn't built yet. Triggers 2 (model self-flagged `summary_conflict`, via the `flag_summary_conflict` tool) and 3 (the scheduled drift audit) are both fully implemented.
- **Cache-hit verification against real usage numbers** ("cached input tokens dominate on non-eviction turns," 4.3's acceptance criteria) needs a live `OPENAI_API_KEY` and several real turns to observe `LLMResponse.usage.cached_input_tokens` — prefix *stability* is verified directly in `tests/unit/test_memory.py`, but the actual provider-side cache hit rate is not.

**Google Calendar wrapper (Issue #17):** no Google Cloud project/OAuth credentials were available in the environment this issue was built in, so `app/tools/calendar.py`'s real API calls have **not been exercised against a live calendar** — `tests/unit/test_calendar.py` mocks the `google-api-python-client` service object directly (request/response shape, error mapping, status-code extraction), which verifies the wrapper's own logic but not real API behavior (rate limits, actual free/busy response shapes for edge cases like all-day events, OAuth token refresh timing). `scripts/gcal_smoke.py` and the `/health` calendar check are both implemented and ready to exercise per the "Google Calendar setup" section above, but still need a real run before this note can be replaced with actual observations.
