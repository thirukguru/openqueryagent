"""Pinecone adapter — connects to Pinecone via the official SDK.

Implements the VectorStoreAdapter protocol for Pinecone.
Requires ``pinecone`` (install with ``pip install openqueryagent[pinecone]``).
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
from openqueryagent.adapters.pinecone_filters import PineconeFilterCompiler
from openqueryagent.core.exceptions import (
    AdapterConnectionError,
    AdapterQueryError,
    SchemaError,
)
from openqueryagent.core.types import (
    AggregationQuery,
    AggregationResult,
    CollectionSchema,
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


class PineconeConnectionConfig(ConnectionConfig):
    """Connection configuration for Pinecone."""

    api_key: str = ""
    index_name: str = ""
    namespace: str = ""
    environment: str | None = None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class PineconeAdapter:
    """Pinecone vector store adapter.

    Uses the ``pinecone`` SDK. Supports namespace-scoped queries,
    metadata-inferred schemas, and metadata filtering.
    """

    def __init__(self, adapter_id: str = "pinecone") -> None:
        self._adapter_id = adapter_id
        self._client: Any = None
        self._index: Any = None
        self._index_name: str = ""
        self._namespace: str = ""
        self._filter_compiler = PineconeFilterCompiler()

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_name(self) -> str:
        return "pinecone"

    @property
    def supports_native_aggregation(self) -> bool:
        return False  # Pinecone has no native aggregation

    async def connect(self, config: ConnectionConfig) -> None:
        """Connect to Pinecone."""
        try:
            from pinecone import Pinecone
        except ImportError as e:
            raise AdapterConnectionError(
                "pinecone is not installed. Install with: pip install openqueryagent[pinecone]",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

        if not isinstance(config, PineconeConnectionConfig):
            config = PineconeConnectionConfig(**config.model_dump())

        if not config.api_key:
            raise AdapterConnectionError(
                "Pinecone API key is required",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            )

        try:
            self._client = Pinecone(api_key=config.api_key)
            self._index = self._client.Index(config.index_name)
            self._index_name = config.index_name
            self._namespace = config.namespace
            logger.info("pinecone_connected", index=config.index_name)
        except Exception as e:
            raise AdapterConnectionError(
                f"Failed to connect to Pinecone: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

    async def disconnect(self) -> None:
        """Disconnect from Pinecone."""
        self._client = None
        self._index = None
        logger.info("pinecone_disconnected")

    async def health_check(self) -> HealthStatus:
        """Check Pinecone health."""
        self._ensure_connected()
        start = time.monotonic()
        try:
            stats = self._index.describe_index_stats()
            latency = (time.monotonic() - start) * 1000
            return HealthStatus(
                healthy=True,
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                latency_ms=latency,
                details={"total_vector_count": stats.get("total_vector_count", 0)},
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
        """List Pinecone indexes (mapped to collections)."""
        self._ensure_connected()
        try:
            indexes = self._client.list_indexes()
            return [idx.name for idx in indexes]
        except Exception as e:
            raise AdapterQueryError(
                f"Failed to list indexes: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection="",
            ) from e

    async def get_schema(self, collection: str) -> CollectionSchema:
        """Infer schema from Pinecone index stats and metadata sampling."""
        self._ensure_connected()
        try:
            stats = self._index.describe_index_stats()
        except Exception as e:
            raise SchemaError(
                f"Failed to get schema for '{collection}': {e}",
                collection=collection,
                adapter_id=self._adapter_id,
            ) from e

        # Infer vector config from stats
        dimension = stats.get("dimension", 0)
        vector_cfg = VectorConfig(dimensions=dimension) if dimension else None

        # Infer properties from namespace metadata if available
        properties: list[PropertySchema] = []
        # Pinecone doesn't have native schema; we infer from first query
        # For now, return minimal schema
        return CollectionSchema(
            name=collection,
            adapter_id=self._adapter_id,
            total_objects=stats.get("total_vector_count", 0),
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
        """Execute search on Pinecone index."""
        self._ensure_connected()

        if not query_vector:
            # Pinecone requires a vector for queries
            return SearchResult(documents=[], total_count=0)

        try:
            query_kwargs: dict[str, Any] = {
                "vector": query_vector,
                "top_k": limit,
                "include_metadata": True,
            }

            if self._namespace:
                query_kwargs["namespace"] = self._namespace

            if filters:
                query_kwargs["filter"] = filters

            response = self._index.query(**query_kwargs)

            documents: list[Document] = []
            for match in response.get("matches", []):
                metadata = match.get("metadata", {})
                documents.append(Document(
                    id=match["id"],
                    content=metadata.get("content", metadata.get("text", "")),
                    metadata={k: v for k, v in metadata.items() if k not in ("content", "text")},
                    score=match.get("score", 0.0),
                    collection=collection,
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
        """Client-side aggregation (Pinecone has no native aggregation)."""
        self._ensure_connected()
        try:
            result_values: dict[str, Any] = {}
            op = aggregation.operation.lower()

            if op == "count":
                stats = self._index.describe_index_stats()
                result_values["count"] = stats.get("total_vector_count", 0)
            else:
                result_values[op] = 0  # Requires full fetch for other ops

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
        """Retrieve documents by ID from Pinecone."""
        self._ensure_connected()
        try:
            fetch_kwargs: dict[str, Any] = {"ids": ids}
            if self._namespace:
                fetch_kwargs["namespace"] = self._namespace

            response = self._index.fetch(**fetch_kwargs)
            documents: list[Document] = []
            for doc_id, vector_data in response.get("vectors", {}).items():
                metadata = vector_data.get("metadata", {})
                documents.append(Document(
                    id=doc_id,
                    content=metadata.get("content", metadata.get("text", "")),
                    metadata={k: v for k, v in metadata.items() if k not in ("content", "text")},
                    collection=collection,
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
        """Get the Pinecone filter compiler."""
        return self._filter_compiler

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if self._index is None:
            raise AdapterConnectionError(
                "Not connected to Pinecone. Call connect() first.",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            )
