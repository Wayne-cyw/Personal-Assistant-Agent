"""GET /health handler.

Static status for now; real component checks (llm, calendar) land in
Issues #17 and later.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "db": "ok", "llm": "unchecked"}
