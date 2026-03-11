"""Qdrant adapter — connects to Qdrant via qdrant-client.

Implements the VectorStoreAdapter protocol for Qdrant vector database.
Requires ``qdrant-client`` (install with ``pip install openqueryagent[qdrant]``).
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from openqueryagent.adapters.base import (
    ConnectionConfig,
    FilterCompiler,
    HealthStatus,
)
from openqueryagent.adapters.qdrant_filters import QdrantFilterCompiler
from openqueryagent.core.exceptions import (
    AdapterConnectionError,
    AdapterQueryError,
    SchemaError,
)
from openqueryagent.core.types import (
    AggregationQuery,
    AggregationResult,
    CollectionSchema,
    DataType,
    DistanceMetric,
    Document,
    PropertySchema,
    SearchResult,
    SearchType,
    VectorConfig,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Connection Config
# ---------------------------------------------------------------------------


class QdrantConnectionConfig(ConnectionConfig):
    """Connection configuration for Qdrant."""

    url: str = "http://localhost:6333"
    api_key: str | None = None
    grpc_port: int | None = None
    prefer_grpc: bool = True
    https: bool = False


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

# Qdrant distance metric mapping
_DISTANCE_MAP: dict[str, DistanceMetric] = {
    "Cosine": DistanceMetric.COSINE,
    "Euclid": DistanceMetric.EUCLIDEAN,
    "Dot": DistanceMetric.DOT_PRODUCT,
}

# Qdrant payload type mapping
_TYPE_MAP: dict[str, DataType] = {
    "keyword": DataType.TEXT,
    "text": DataType.TEXT,
    "integer": DataType.INT,
    "float": DataType.FLOAT,
    "bool": DataType.BOOL,
    "datetime": DataType.DATE,
    "geo": DataType.GEO,
}


class QdrantAdapter:
    """Qdrant vector store adapter.

    Uses ``qdrant_client.AsyncQdrantClient`` for async operations.
    """

    def __init__(self, adapter_id: str = "qdrant") -> None:
        self._adapter_id = adapter_id
        self._client: Any = None
        self._filter_compiler = QdrantFilterCompiler()

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_name(self) -> str:
        return "qdrant"

    @property
    def supports_native_aggregation(self) -> bool:
        return False  # Qdrant doesn't have native aggregation

    async def connect(self, config: ConnectionConfig) -> None:
        """Connect to Qdrant server."""
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError as e:
            raise AdapterConnectionError(
                "qdrant-client is not installed. Install with: pip install openqueryagent[qdrant]",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

        if not isinstance(config, QdrantConnectionConfig):
            config = QdrantConnectionConfig(**config.model_dump())

        try:
            self._client = AsyncQdrantClient(
                url=config.url,
                api_key=config.api_key,
                grpc_port=config.grpc_port or 6334,
                prefer_grpc=config.prefer_grpc,
                https=config.https,
                timeout=int(config.timeout_seconds),
            )
            logger.info("qdrant_connected", url=config.url)
        except Exception as e:
            raise AdapterConnectionError(
                f"Failed to connect to Qdrant: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

    async def disconnect(self) -> None:
        """Disconnect from Qdrant server."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("qdrant_disconnected")

    async def health_check(self) -> HealthStatus:
        """Check Qdrant server health."""
        self._ensure_connected()
        start = time.monotonic()
        try:
            # qdrant_client health check returns True/raises
            await self._client.get_collections()
            latency = (time.monotonic() - start) * 1000
            return HealthStatus(
                healthy=True,
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return HealthStatus(
                healthy=False,
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                message=str(e),
                latency_ms=latency,
            )

    async def get_collections(self) -> list[str]:
        """List all Qdrant collections."""
        self._ensure_connected()
        try:
            response = await self._client.get_collections()
            return [c.name for c in response.collections]
        except Exception as e:
            raise AdapterQueryError(
                f"Failed to list collections: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection="",
            ) from e

    async def get_schema(self, collection: str) -> CollectionSchema:
        """Introspect collection schema from Qdrant."""
        self._ensure_connected()
        try:
            info = await self._client.get_collection(collection)
        except Exception as e:
            raise SchemaError(
                f"Failed to get schema for '{collection}': {e}",
                collection=collection,
                adapter_id=self._adapter_id,
            ) from e

        # Parse vector config
        vectors_config = info.config.params.vectors
        vector_cfg = None
        if hasattr(vectors_config, "size"):
            # Single unnamed vector
            distance_str = str(vectors_config.distance)
            vector_cfg = VectorConfig(
                dimensions=vectors_config.size,
                distance_metric=_DISTANCE_MAP.get(distance_str, DistanceMetric.COSINE),
            )

        # Parse payload schema
        properties: list[PropertySchema] = []
        if info.payload_schema:
            for name, field_info in info.payload_schema.items():
                data_type_str = str(field_info.data_type) if field_info.data_type else "text"
                data_type = _TYPE_MAP.get(data_type_str, DataType.TEXT)
                properties.append(PropertySchema(
                    name=name,
                    data_type=data_type,
                    filterable=True,
                ))

        return CollectionSchema(
            name=collection,
            adapter_id=self._adapter_id,
            total_objects=info.points_count or 0,
            vector_config=vector_cfg,
            properties=properties,
        )

    async def search(
        self,
        collection: str,
        query_vector: list[float] | None = None,
        query_text: str | None = None,
        filters: Any | None = None,
        limit: int = 10,
        offset: int = 0,
        search_type: SearchType = SearchType.HYBRID,
        search_params: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Execute search on Qdrant collection."""
        self._ensure_connected()
        from qdrant_client.models import Filter

        try:
            query_filter = Filter(**filters) if isinstance(filters, dict) and filters else None

            if search_type == SearchType.VECTOR and query_vector:
                results = await self._client.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    offset=offset,
                )
            elif search_type == SearchType.KEYWORD and query_text:
                # Keyword search via scroll with text match filter
                results_scroll = await self._client.scroll(
                    collection_name=collection,
                    scroll_filter=query_filter,
                    limit=limit,
                    offset=offset,
                )
                results = results_scroll[0]  # (points, next_offset)
            else:
                # Hybrid or fallback to vector if available
                if query_vector:
                    results = await self._client.search(
                        collection_name=collection,
                        query_vector=query_vector,
                        query_filter=query_filter,
                        limit=limit,
                        offset=offset,
                    )
                else:
                    results = await self._client.scroll(
                        collection_name=collection,
                        scroll_filter=query_filter,
                        limit=limit,
                    )
                    results = results[0]

            # Convert results to Documents
            documents: list[Document] = []
            for point in results:
                payload = point.payload or {}
                score = getattr(point, "score", None)
                documents.append(Document(
                    id=str(point.id),
                    content=payload.get("content", payload.get("text", "")),
                    metadata={k: v for k, v in payload.items() if k not in ("content", "text")},
                    score=score if score is not None else 0.0,
                ))

            return SearchResult(
                documents=documents,
                total_count=len(documents),
            )
        except Exception as e:
            raise AdapterQueryError(
                f"Search failed on '{collection}': {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection=collection,
            ) from e

    async def aggregate(
        self,
        collection: str,
        aggregation: AggregationQuery,
        filters: Any | None = None,
    ) -> AggregationResult:
        """Aggregate via client-side scroll (Qdrant has no native aggregation)."""
        self._ensure_connected()
        from qdrant_client.models import Filter

        try:
            query_filter = Filter(**filters) if isinstance(filters, dict) and filters else None

            # Scroll all matching points
            all_values: list[Any] = []
            next_offset = None
            scroll_limit = 100

            while True:
                points, next_offset = await self._client.scroll(
                    collection_name=collection,
                    scroll_filter=query_filter,
                    limit=scroll_limit,
                    offset=next_offset,
                )
                for point in points:
                    payload = point.payload or {}
                    if aggregation.field and aggregation.field in payload:
                        all_values.append(payload[aggregation.field])
                if next_offset is None or len(points) < scroll_limit:
                    break

            # Compute aggregation client-side
            result_values: dict[str, Any] = {}
            op = aggregation.operation.lower()

            if op == "count":
                result_values["count"] = len(all_values)
            elif op == "sum" and all_values:
                result_values["sum"] = sum(float(v) for v in all_values)
            elif op == "avg" and all_values:
                total = sum(float(v) for v in all_values)
                result_values["avg"] = total / len(all_values)
            elif op == "min" and all_values:
                result_values["min"] = min(all_values)
            elif op == "max" and all_values:
                result_values["max"] = max(all_values)

            return AggregationResult(
                values=result_values,
                is_approximate=True,
            )
        except Exception as e:
            raise AdapterQueryError(
                f"Aggregation failed on '{collection}': {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection=collection,
            ) from e

    async def get_by_ids(self, collection: str, ids: list[str]) -> list[Document]:
        """Retrieve documents by ID from Qdrant."""
        self._ensure_connected()
        try:
            points = await self._client.retrieve(
                collection_name=collection,
                ids=ids,
            )
            documents: list[Document] = []
            for point in points:
                payload = point.payload or {}
                documents.append(Document(
                    id=str(point.id),
                    content=payload.get("content", payload.get("text", "")),
                    metadata={k: v for k, v in payload.items() if k not in ("content", "text")},
                ))
            return documents
        except Exception as e:
            raise AdapterQueryError(
                f"get_by_ids failed on '{collection}': {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection=collection,
            ) from e

    def get_filter_compiler(self) -> FilterCompiler:
        """Get the Qdrant filter compiler."""
        return self._filter_compiler

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if self._client is None:
            raise AdapterConnectionError(
                "Not connected to Qdrant. Call connect() first.",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            )
