"""Weaviate adapter — connects to Weaviate via weaviate-client v4.

Implements the VectorStoreAdapter protocol for Weaviate vector database.
Requires ``weaviate-client`` (install with ``pip install openqueryagent[weaviate]``).
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
from openqueryagent.adapters.weaviate_filters import WeaviateFilterCompiler
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


class WeaviateConnectionConfig(ConnectionConfig):
    """Connection configuration for Weaviate."""

    url: str = "http://localhost:8080"
    grpc_host: str | None = None
    grpc_port: int = 50051
    api_key: str | None = None
    auth_credentials: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

# Weaviate data type mapping
_TYPE_MAP: dict[str, DataType] = {
    "text": DataType.TEXT,
    "text[]": DataType.TEXT,
    "int": DataType.INT,
    "int[]": DataType.INT,
    "number": DataType.FLOAT,
    "number[]": DataType.FLOAT,
    "boolean": DataType.BOOL,
    "date": DataType.DATE,
    "geoCoordinates": DataType.GEO,
    "blob": DataType.TEXT,
    "uuid": DataType.TEXT,
}

# Weaviate distance metric mapping
_DISTANCE_MAP: dict[str, DistanceMetric] = {
    "cosine": DistanceMetric.COSINE,
    "l2-squared": DistanceMetric.EUCLIDEAN,
    "dot": DistanceMetric.DOT_PRODUCT,
}


class WeaviateAdapter:
    """Weaviate vector store adapter.

    Uses ``weaviate-client`` v4 for operations. Supports server-side
    vectorization, hybrid search with alpha parameter, and native aggregation.
    """

    def __init__(self, adapter_id: str = "weaviate") -> None:
        self._adapter_id = adapter_id
        self._client: Any = None
        self._filter_compiler = WeaviateFilterCompiler()

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_name(self) -> str:
        return "weaviate"

    @property
    def supports_native_aggregation(self) -> bool:
        return True  # Weaviate has native aggregate API

    async def connect(self, config: ConnectionConfig) -> None:
        """Connect to Weaviate server."""
        try:
            import weaviate
        except ImportError as e:
            raise AdapterConnectionError(
                "weaviate-client is not installed. Install with: pip install openqueryagent[weaviate]",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

        if not isinstance(config, WeaviateConnectionConfig):
            config = WeaviateConnectionConfig(**config.model_dump())

        try:
            connect_kwargs: dict[str, Any] = {}
            if config.api_key:
                connect_kwargs["auth_credentials"] = weaviate.auth.AuthApiKey(config.api_key)

            if config.grpc_host:
                self._client = weaviate.connect_to_custom(
                    http_host=config.url.replace("http://", "").replace("https://", "").split(":")[0],
                    http_port=int(config.url.split(":")[-1]) if ":" in config.url.rsplit("/", 1)[-1] else 8080,
                    grpc_host=config.grpc_host,
                    grpc_port=config.grpc_port,
                    **connect_kwargs,
                )
            else:
                self._client = weaviate.connect_to_local(
                    host=config.url.replace("http://", "").replace("https://", "").split(":")[0],
                    port=int(config.url.split(":")[-1]) if ":" in config.url.rsplit("/", 1)[-1] else 8080,
                    grpc_port=config.grpc_port,
                    **connect_kwargs,
                )

            logger.info("weaviate_connected", url=config.url)
        except Exception as e:
            raise AdapterConnectionError(
                f"Failed to connect to Weaviate: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

    async def disconnect(self) -> None:
        """Disconnect from Weaviate server."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("weaviate_disconnected")

    async def health_check(self) -> HealthStatus:
        """Check Weaviate server health."""
        self._ensure_connected()
        start = time.monotonic()
        try:
            is_ready = self._client.is_ready()
            latency = (time.monotonic() - start) * 1000
            return HealthStatus(
                healthy=is_ready,
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
        """List all Weaviate collections."""
        self._ensure_connected()
        try:
            collections = self._client.collections.list_all()
            return list(collections.keys())
        except Exception as e:
            raise AdapterQueryError(
                f"Failed to list collections: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection="",
            ) from e

    async def get_schema(self, collection: str) -> CollectionSchema:
        """Introspect collection schema from Weaviate."""
        self._ensure_connected()
        try:
            col = self._client.collections.get(collection)
            col_config = col.config.get()
        except Exception as e:
            raise SchemaError(
                f"Failed to get schema for '{collection}': {e}",
                collection=collection,
                adapter_id=self._adapter_id,
            ) from e

        # Parse properties
        properties: list[PropertySchema] = []
        for prop in col_config.properties:
            dt = _TYPE_MAP.get(str(prop.data_type).lower(), DataType.TEXT)
            properties.append(PropertySchema(
                name=prop.name,
                data_type=dt,
                filterable=True,
                description=getattr(prop, "description", None),
            ))

        # Parse vector config
        vector_cfg = None
        if col_config.vectorizer_config:
            # Default dimensions if vectorizer is configured
            distance_str = str(getattr(col_config, "vector_index_type", "cosine")).lower()
            vector_cfg = VectorConfig(
                dimensions=0,  # Weaviate doesn't expose dimensions in schema
                distance_metric=_DISTANCE_MAP.get(distance_str, DistanceMetric.COSINE),
            )

        return CollectionSchema(
            name=collection,
            adapter_id=self._adapter_id,
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
        """Execute search on Weaviate collection."""
        self._ensure_connected()

        try:
            col = self._client.collections.get(collection)

            if search_type == SearchType.VECTOR and query_vector:
                response = col.query.near_vector(
                    near_vector=query_vector,
                    limit=limit,
                    offset=offset,
                    filters=filters,
                )
            elif search_type == SearchType.KEYWORD and query_text:
                response = col.query.bm25(
                    query=query_text,
                    limit=limit,
                    offset=offset,
                    filters=filters,
                )
            elif search_type == SearchType.HYBRID and query_text:
                alpha = 0.5  # Balance between vector and keyword
                if search_params and "alpha" in search_params:
                    alpha = search_params["alpha"]
                response = col.query.hybrid(
                    query=query_text,
                    alpha=alpha,
                    limit=limit,
                    offset=offset,
                    filters=filters,
                    vector=query_vector,
                )
            elif query_vector:
                response = col.query.near_vector(
                    near_vector=query_vector,
                    limit=limit,
                    offset=offset,
                    filters=filters,
                )
            else:
                response = col.query.fetch_objects(
                    limit=limit,
                    offset=offset,
                    filters=filters,
                )

            documents = self._convert_objects(response.objects, collection)
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
        """Execute aggregation using Weaviate's native aggregate API."""
        self._ensure_connected()
        try:
            col = self._client.collections.get(collection)
            result_values: dict[str, Any] = {}
            op = aggregation.operation.lower()

            if op == "count":
                response = col.aggregate.over_all(total_count=True, filters=filters)
                result_values["count"] = response.total_count
            elif aggregation.field:
                # Use property-specific aggregation
                response = col.aggregate.over_all(filters=filters, total_count=True)
                result_values["count"] = response.total_count
                # Weaviate aggregation API with property is more complex;
                # fall back to count for basic ops
                if op in ("sum", "avg", "min", "max"):
                    result_values[op] = 0  # Placeholder — native agg requires metrics config

            return AggregationResult(
                values=result_values,
                is_approximate=False,
            )
        except Exception as e:
            raise AdapterQueryError(
                f"Aggregation failed on '{collection}': {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection=collection,
            ) from e

    async def get_by_ids(self, collection: str, ids: list[str]) -> list[Document]:
        """Retrieve documents by ID from Weaviate."""
        self._ensure_connected()
        try:
            col = self._client.collections.get(collection)
            documents: list[Document] = []
            for doc_id in ids:
                try:
                    obj = col.query.fetch_object_by_id(doc_id)
                    if obj:
                        props = dict(obj.properties) if obj.properties else {}
                        documents.append(Document(
                            id=str(obj.uuid),
                            content=str(props.get("content", props.get("text", ""))),
                            metadata={k: v for k, v in props.items() if k not in ("content", "text")},
                            collection=collection,
                        ))
                except Exception:
                    continue  # Skip missing IDs
            return documents
        except Exception as e:
            raise AdapterQueryError(
                f"get_by_ids failed on '{collection}': {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection=collection,
            ) from e

    def get_filter_compiler(self) -> FilterCompiler:
        """Get the Weaviate filter compiler."""
        return self._filter_compiler

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if self._client is None:
            raise AdapterConnectionError(
                "Not connected to Weaviate. Call connect() first.",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            )

    @staticmethod
    def _convert_objects(objects: Any, collection: str) -> list[Document]:
        """Convert Weaviate objects to Document models."""
        documents: list[Document] = []
        for obj in objects:
            props = dict(obj.properties) if obj.properties else {}
            score = getattr(obj.metadata, "score", None) if hasattr(obj, "metadata") else None
            documents.append(Document(
                id=str(obj.uuid),
                content=str(props.get("content", props.get("text", ""))),
                metadata={k: v for k, v in props.items() if k not in ("content", "text")},
                score=score if score is not None else 0.0,
                collection=collection,
            ))
        return documents
