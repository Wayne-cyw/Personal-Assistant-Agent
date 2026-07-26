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
