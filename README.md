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

### Model notes

**`gpt-5.6-luna` function-calling reliability (Issue #10):** the orchestration loop (`app/agent/loop.py`) and its automated tests (`tests/unit/test_loop.py`) are fully verified against `FakeProvider`, which doesn't exercise real model behavior. No `OPENAI_API_KEY` was available in the environment this issue was built in, so Luna's actual tool-calling reliability at this budget tier has **not yet been observed against a live model** — the CLI acceptance check ("what day is it?" → tool call → correct date) and the config-only escalation path to `gpt-5.6-terra` (`LLM_MODEL=gpt-5.6-terra` for the main loop only, per the Tech Stack table) are both implemented and ready to exercise, but still need a real run before this note can be replaced with actual observations. Whoever runs that first live session should update this note with what they saw (successful/malformed tool calls, retry behavior, anything else worth flagging) rather than leave it as a placeholder.
