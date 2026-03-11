"""FastAPI dependency injection helpers."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from openqueryagent.core.agent import QueryAgent


def get_agent(request: Request) -> QueryAgent:
    """Return the shared ``QueryAgent`` stored on ``app.state``."""
    agent: QueryAgent = request.app.state.agent  # type: ignore[assignment]
    return agent


def get_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())
