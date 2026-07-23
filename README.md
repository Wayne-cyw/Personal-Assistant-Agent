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
