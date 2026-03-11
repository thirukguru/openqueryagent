"""FastAPI application for OpenQueryAgent REST API.

Usage::

    from openqueryagent.server import create_app
    app = create_app()

Or run directly::

    python -m openqueryagent.server
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from openqueryagent.core.agent import QueryAgent
from openqueryagent.core.exceptions import (
    AdapterConnectionError,
    AdapterQueryError,
    FilterCompilationError,
    OpenQueryAgentError,
    PlannerError,
    QueryTimeoutError,
    RateLimitError,
    SchemaError,
    SynthesisError,
)
from openqueryagent.server.config import ServerConfig
from openqueryagent.server.dependencies import get_agent, get_request_id
from openqueryagent.server.middleware import (
    APIKeyMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
)
from openqueryagent.server.models import (
    AdapterHealth,
    AggregateRequest,
    AskRequest,
    CollectionsListResponse,
    ErrorResponse,
    HealthResponse,
    SearchRequest,
)
from openqueryagent.server.websocket import ask_stream_handler

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = structlog.get_logger(__name__)

# Graceful shutdown tracking
_SHUTDOWN_TIMEOUT = 30  # seconds to wait for in-flight requests


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize QueryAgent on startup, tear down on shutdown."""
    config: ServerConfig = app.state.server_config  # type: ignore[attr-defined]
    adapters = await _build_adapters(config)

    agent = QueryAgent(adapters=adapters)
    await agent.initialize()

    app.state.agent = agent
    logger.info("server_started", adapters=list(adapters.keys()))

    yield

    # Shutdown: drain in-flight requests
    logger.info("server_shutting_down", msg="waiting for in-flight requests")
    import asyncio
    in_flight: int = getattr(app.state, "in_flight_count", 0)
    for _ in range(_SHUTDOWN_TIMEOUT):
        in_flight = getattr(app.state, "in_flight_count", 0)
        if in_flight <= 0:
            break
        await asyncio.sleep(1)

    # Disconnect adapters
    for adapter in adapters.values():
        try:
            await adapter.disconnect()  # type: ignore[union-attr]
        except Exception:
            pass
    logger.info("server_stopped")


async def _build_adapters(config: ServerConfig) -> dict[str, Any]:
    """Construct adapters from server configuration.

    For each configured adapter URL/DSN, import the adapter class,
    instantiate it, and connect.
    """
    adapters: dict[str, Any] = {}

    if config.qdrant_url:
        try:
            from openqueryagent.adapters.qdrant import QdrantAdapter

            adapter = QdrantAdapter(url=config.qdrant_url)
            await adapter.connect()
            adapters["qdrant"] = adapter
        except Exception as exc:
            logger.error("adapter_init_failed", adapter="qdrant", error=str(exc))

    if config.pgvector_dsn:
        try:
            from openqueryagent.adapters.pgvector import PgvectorAdapter

            adapter = PgvectorAdapter(dsn=config.pgvector_dsn)
            await adapter.connect()
            adapters["pgvector"] = adapter
        except Exception as exc:
            logger.error("adapter_init_failed", adapter="pgvector", error=str(exc))

    if config.milvus_url:
        try:
            from openqueryagent.adapters.milvus import MilvusAdapter

            adapter = MilvusAdapter(url=config.milvus_url)
            await adapter.connect()
            adapters["milvus"] = adapter
        except Exception as exc:
            logger.error("adapter_init_failed", adapter="milvus", error=str(exc))

    if config.weaviate_url:
        try:
            from openqueryagent.adapters.weaviate import WeaviateAdapter

            adapter = WeaviateAdapter(url=config.weaviate_url)
            await adapter.connect()
            adapters["weaviate"] = adapter
        except Exception as exc:
            logger.error("adapter_init_failed", adapter="weaviate", error=str(exc))

    if config.chroma_url:
        try:
            from openqueryagent.adapters.chroma import ChromaAdapter

            adapter = ChromaAdapter(url=config.chroma_url)
            await adapter.connect()
            adapters["chroma"] = adapter
        except Exception as exc:
            logger.error("adapter_init_failed", adapter="chroma", error=str(exc))

    if config.elasticsearch_url:
        try:
            from openqueryagent.adapters.elasticsearch import ElasticsearchAdapter

            adapter = ElasticsearchAdapter(url=config.elasticsearch_url)
            await adapter.connect()
            adapters["elasticsearch"] = adapter
        except Exception as exc:
            logger.error("adapter_init_failed", adapter="elasticsearch", error=str(exc))

    if config.pinecone_api_key:
        try:
            from openqueryagent.adapters.pinecone import PineconeAdapter

            adapter = PineconeAdapter(api_key=config.pinecone_api_key)
            await adapter.connect()
            adapters["pinecone"] = adapter
        except Exception as exc:
            logger.error("adapter_init_failed", adapter="pinecone", error=str(exc))

    if config.s3vectors_region and config.s3vectors_bucket:
        try:
            from openqueryagent.adapters.s3vectors import S3VectorsAdapter

            adapter = S3VectorsAdapter(
                region=config.s3vectors_region, bucket=config.s3vectors_bucket
            )
            await adapter.connect()
            adapters["s3vectors"] = adapter
        except Exception as exc:
            logger.error("adapter_init_failed", adapter="s3vectors", error=str(exc))

    return adapters


# ---------------------------------------------------------------------------
# Exception → HTTP mapping
# ---------------------------------------------------------------------------

_ERROR_STATUS: dict[type[OpenQueryAgentError], int] = {
    AdapterConnectionError: 503,
    AdapterQueryError: 502,
    PlannerError: 500,
    FilterCompilationError: 422,
    SynthesisError: 500,
    QueryTimeoutError: 504,
    SchemaError: 404,
    RateLimitError: 429,
}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Server configuration. If ``None``, loads from env vars.

    Returns:
        Configured FastAPI application.
    """
    config = config or ServerConfig()

    app = FastAPI(
        title="OpenQueryAgent",
        description="Database-agnostic query agent for vector databases",
        version="0.3.0",
        lifespan=_lifespan,
    )

    # Store config on app state for lifespan access
    app.state.server_config = config  # type: ignore[attr-defined]

    # ---- Middleware (order matters: outermost first) ----
    app.add_middleware(RequestIDMiddleware)

    if config.rate_limit:
        app.add_middleware(RateLimitMiddleware, rate_limit=config.rate_limit)

    if config.api_key:
        app.add_middleware(APIKeyMiddleware, api_key=config.api_key)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Exception handlers ----
    @app.exception_handler(OpenQueryAgentError)
    async def _oqa_error_handler(request: Request, exc: OpenQueryAgentError) -> JSONResponse:
        status = _ERROR_STATUS.get(type(exc), 500)
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(
                error=type(exc).__name__,
                detail=str(exc),
                request_id=request_id,
            ).model_dump(),
        )

    @app.exception_handler(ValueError)
    async def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="ValidationError",
                detail=str(exc),
                request_id=request_id,
            ).model_dump(),
        )

    # ---- Routes ----
    _register_routes(app)

    # ---- WebSocket ----
    app.websocket("/v1/ask/stream")(ask_stream_handler)

    return app


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    """Register all REST endpoints."""

    @app.post("/v1/ask", tags=["Query"])
    async def ask(
        body: AskRequest,
        agent: QueryAgent = Depends(get_agent),
        request_id: str = Depends(get_request_id),
    ) -> Any:
        """Ask a natural language question and get an answer with citations."""
        start = time.monotonic()
        response = await agent.ask(body.query)
        latency = (time.monotonic() - start) * 1000
        logger.info("ask_completed", request_id=request_id, latency_ms=round(latency, 1))
        return response

    @app.post("/v1/search", tags=["Query"])
    async def search(
        body: SearchRequest,
        agent: QueryAgent = Depends(get_agent),
        request_id: str = Depends(get_request_id),
    ) -> Any:
        """Search for documents without answer synthesis."""
        start = time.monotonic()
        response = await agent.search(body.query, limit=body.limit)
        latency = (time.monotonic() - start) * 1000
        logger.info("search_completed", request_id=request_id, latency_ms=round(latency, 1))
        return response

    @app.post("/v1/aggregate", tags=["Query"])
    async def aggregate(
        body: AggregateRequest,
        agent: QueryAgent = Depends(get_agent),
        request_id: str = Depends(get_request_id),
    ) -> Any:
        """Execute an aggregation query."""
        start = time.monotonic()
        response = await agent.aggregate(body.query)
        latency = (time.monotonic() - start) * 1000
        logger.info("aggregate_completed", request_id=request_id, latency_ms=round(latency, 1))
        return response

    @app.get("/v1/collections", tags=["Schema"])
    async def list_collections(
        agent: QueryAgent = Depends(get_agent),
    ) -> CollectionsListResponse:
        """List available collections and their schemas."""
        schema_map = await agent.schema_inspector.get_schema_map()
        return CollectionsListResponse(
            collections=list(schema_map.collections.keys()),
            schemas=schema_map.collections,
        )

    @app.get("/v1/collections/{name}/schema", tags=["Schema"])
    async def get_collection_schema(
        name: str,
        agent: QueryAgent = Depends(get_agent),
    ) -> Any:
        """Get the schema of a specific collection."""
        schema_map = await agent.schema_inspector.get_schema_map()
        if name not in schema_map.collections:
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(
                    error="NotFound",
                    detail=f"Collection '{name}' not found",
                ).model_dump(),
            )
        return schema_map.collections[name]

    @app.get("/v1/health", tags=["System"])
    async def health_check(request: Request) -> HealthResponse:
        """Health check with adapter connectivity status."""
        from openqueryagent import __version__

        adapters_health: list[AdapterHealth] = []
        overall = "healthy"

        if hasattr(request.app.state, "agent"):
            agent: QueryAgent = request.app.state.agent
            for adapter_id, adapter in agent._adapters.items():
                start = time.monotonic()
                try:
                    await adapter.health_check()
                    latency = (time.monotonic() - start) * 1000
                    adapters_health.append(
                        AdapterHealth(
                            adapter_id=adapter_id,
                            status="healthy",
                            latency_ms=round(latency, 1),
                        )
                    )
                except Exception as exc:
                    latency = (time.monotonic() - start) * 1000
                    overall = "degraded"
                    adapters_health.append(
                        AdapterHealth(
                            adapter_id=adapter_id,
                            status="unhealthy",
                            latency_ms=round(latency, 1),
                            error=str(exc),
                        )
                    )

        if all(a.status == "unhealthy" for a in adapters_health) and adapters_health:
            overall = "unhealthy"

        return HealthResponse(
            status=overall,
            version=__version__,
            adapters=adapters_health,
        )

    @app.get("/v1/metrics", tags=["System"])
    async def metrics() -> Any:
        """Prometheus metrics endpoint."""
        from starlette.responses import Response as StarletteResponse

        from openqueryagent.observability.metrics import get_metrics

        data = get_metrics().generate_latest()
        return StarletteResponse(
            content=data,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
