"""Request / response models for the REST API.

Response models re-use the existing ``core.types`` Pydantic models so
serialization stays consistent between embedded and server modes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from openqueryagent.core.types import (
    AggregationResponse,
    AskResponse,
    CollectionSchema,
    SearchResponse,
)

# Re-export response types so consumers can import from server.models
__all__ = [
    "AskRequest",
    "SearchRequest",
    "AggregateRequest",
    "AskResponse",
    "SearchResponse",
    "AggregationResponse",
    "CollectionsListResponse",
    "AdapterHealth",
    "HealthResponse",
    "ErrorResponse",
]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """Request body for ``POST /v1/ask``."""

    query: str = Field(..., min_length=1, max_length=10_000, description="Natural language question")
    session_id: str | None = Field(default=None, description="Session ID for conversation continuity")
    stream: bool = Field(default=False, description="If true, use WebSocket streaming instead")


class SearchRequest(BaseModel):
    """Request body for ``POST /v1/search``."""

    query: str = Field(..., min_length=1, max_length=10_000, description="Natural language search query")
    limit: int = Field(default=10, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, le=100_000, description="Pagination offset")
    filters: dict[str, Any] | None = Field(default=None, description="Filter expression dict")


class AggregateRequest(BaseModel):
    """Request body for ``POST /v1/aggregate``."""

    query: str = Field(..., min_length=1, max_length=10_000, description="Natural language aggregation query")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class CollectionsListResponse(BaseModel):
    """Response for ``GET /v1/collections``."""

    collections: list[str] = Field(default_factory=list)
    schemas: dict[str, CollectionSchema] = Field(default_factory=dict)


class AdapterHealth(BaseModel):
    """Health status of a single adapter."""

    adapter_id: str
    status: str  # "healthy" | "unhealthy"
    latency_ms: float = 0.0
    error: str | None = None


class HealthResponse(BaseModel):
    """Response for ``GET /v1/health``."""

    status: str  # "healthy" | "degraded" | "unhealthy"
    version: str
    adapters: list[AdapterHealth] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str
    detail: str = ""
    request_id: str | None = None
