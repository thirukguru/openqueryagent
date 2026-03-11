"""Chroma adapter — connects to ChromaDB.

Implements the VectorStoreAdapter protocol for ChromaDB.
Requires ``chromadb`` (install with ``pip install openqueryagent[chroma]``).
Supports both ``HttpClient`` (remote) and ``PersistentClient`` (local).
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
from openqueryagent.adapters.chroma_filters import ChromaFilterCompiler
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
    Document,
    PropertySchema,
    SearchResult,
    SearchType,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Connection Config
# ---------------------------------------------------------------------------


class ChromaConnectionConfig(ConnectionConfig):
    """Connection configuration for ChromaDB."""

    host: str = "localhost"
    port: int = 8000
    persist_directory: str | None = None
    mode: str = "http"  # "http" or "persistent"
    tenant: str | None = None
    database: str | None = None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ChromaAdapter:
    """ChromaDB vector store adapter.

    Supports both remote (HttpClient) and local (PersistentClient) modes.
    """

    def __init__(self, adapter_id: str = "chroma") -> None:
        self._adapter_id = adapter_id
        self._client: Any = None
        self._filter_compiler = ChromaFilterCompiler()

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_name(self) -> str:
        return "chroma"

    @property
    def supports_native_aggregation(self) -> bool:
        return False  # Chroma has no native aggregation

    async def connect(self, config: ConnectionConfig) -> None:
        """Connect to ChromaDB."""
        try:
            import chromadb
        except ImportError as e:
            raise AdapterConnectionError(
                "chromadb is not installed. Install with: pip install openqueryagent[chroma]",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

        if not isinstance(config, ChromaConnectionConfig):
            config = ChromaConnectionConfig(**config.model_dump())

        try:
            if config.mode == "persistent" and config.persist_directory:
                self._client = chromadb.PersistentClient(path=config.persist_directory)
            else:
                self._client = chromadb.HttpClient(
                    host=config.host,
                    port=config.port,
                )
            logger.info("chroma_connected", mode=config.mode)
        except Exception as e:
            raise AdapterConnectionError(
                f"Failed to connect to ChromaDB: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

    async def disconnect(self) -> None:
        """Disconnect from ChromaDB."""
        self._client = None
        logger.info("chroma_disconnected")

    async def health_check(self) -> HealthStatus:
        """Check ChromaDB health."""
        self._ensure_connected()
        start = time.monotonic()
        try:
            heartbeat = self._client.heartbeat()
            latency = (time.monotonic() - start) * 1000
            return HealthStatus(
                healthy=True,
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                latency_ms=latency,
                details={"heartbeat": heartbeat},
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
        """List all ChromaDB collections."""
        self._ensure_connected()
        try:
            collections = self._client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            raise AdapterQueryError(
                f"Failed to list collections: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection="",
            ) from e

    async def get_schema(self, collection: str) -> CollectionSchema:
        """Introspect schema from ChromaDB collection metadata + sampling."""
        self._ensure_connected()
        try:
            col = self._client.get_collection(collection)
        except Exception as e:
            raise SchemaError(
                f"Failed to get schema for '{collection}': {e}",
                collection=collection,
                adapter_id=self._adapter_id,
            ) from e

        # Infer properties from metadata sampling
        properties: list[PropertySchema] = []
        try:
            sample = col.peek(limit=10)
            if sample and sample.get("metadatas"):
                seen_keys: set[str] = set()
                for meta in sample["metadatas"]:
                    if meta:
                        for key, value in meta.items():
                            if key not in seen_keys:
                                seen_keys.add(key)
                                dt = self._infer_type(value)
                                properties.append(PropertySchema(
                                    name=key,
                                    data_type=dt,
                                    filterable=True,
                                ))
        except Exception:
            pass  # Sampling failed; return empty properties

        return CollectionSchema(
            name=collection,
            adapter_id=self._adapter_id,
            total_objects=col.count(),
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
        """Execute search on ChromaDB collection."""
        self._ensure_connected()

        try:
            col = self._client.get_collection(collection)

            query_kwargs: dict[str, Any] = {
                "n_results": limit,
            }

            if filters:
                # Separate where_document from where
                if "$contains" in filters:
                    query_kwargs["where_document"] = filters
                else:
                    query_kwargs["where"] = filters

            if search_type == SearchType.VECTOR and query_vector:
                query_kwargs["query_embeddings"] = [query_vector]
                results = col.query(**query_kwargs)
            elif search_type == SearchType.KEYWORD and query_text:
                query_kwargs["query_texts"] = [query_text]
                results = col.query(**query_kwargs)
            elif query_vector:
                query_kwargs["query_embeddings"] = [query_vector]
                results = col.query(**query_kwargs)
            elif query_text:
                query_kwargs["query_texts"] = [query_text]
                results = col.query(**query_kwargs)
            else:
                # No query — get all
                get_results = col.get(limit=limit)
                return self._convert_get_results(get_results, collection)

            return self._convert_query_results(results, collection)
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
        """Client-side aggregation via get() (Chroma has no native aggregation)."""
        self._ensure_connected()
        try:
            col = self._client.get_collection(collection)
            result_values: dict[str, Any] = {}
            op = aggregation.operation.lower()

            if op == "count":
                if filters:
                    # Get with filter and count
                    data = col.get(where=filters)
                    result_values["count"] = len(data.get("ids", []))
                else:
                    result_values["count"] = col.count()
            elif aggregation.field:
                # Fetch metadata and compute
                get_kwargs: dict[str, Any] = {}
                if filters:
                    get_kwargs["where"] = filters
                data = col.get(**get_kwargs)
                values: list[Any] = []
                for meta in data.get("metadatas", []):
                    if meta and aggregation.field in meta:
                        values.append(meta[aggregation.field])

                if values:
                    if op == "sum":
                        result_values["sum"] = sum(float(v) for v in values)
                    elif op == "avg":
                        result_values["avg"] = sum(float(v) for v in values) / len(values)
                    elif op == "min":
                        result_values["min"] = min(values)
                    elif op == "max":
                        result_values["max"] = max(values)

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
        """Retrieve documents by ID from ChromaDB."""
        self._ensure_connected()
        try:
            col = self._client.get_collection(collection)
            results = col.get(ids=ids)
            return self._convert_get_results(results, collection).documents
        except Exception as e:
            raise AdapterQueryError(
                f"get_by_ids failed on '{collection}': {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection=collection,
            ) from e

    def get_filter_compiler(self) -> FilterCompiler:
        """Get the Chroma filter compiler."""
        return self._filter_compiler

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if self._client is None:
            raise AdapterConnectionError(
                "Not connected to ChromaDB. Call connect() first.",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            )

    @staticmethod
    def _convert_query_results(results: dict[str, Any], collection: str) -> SearchResult:
        """Convert Chroma query results to SearchResult."""
        documents: list[Document] = []
        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids_list):
            content = docs_list[i] if i < len(docs_list) else ""
            metadata = metas_list[i] if i < len(metas_list) else {}
            # Chroma returns distance; convert to similarity score
            distance = distances[i] if i < len(distances) else 0.0
            score = 1.0 / (1.0 + distance) if distance >= 0 else 0.0

            documents.append(Document(
                id=str(doc_id),
                content=content or "",
                metadata=metadata or {},
                score=score,
                collection=collection,
            ))

        return SearchResult(documents=documents, total_count=len(documents))

    @staticmethod
    def _convert_get_results(results: dict[str, Any], collection: str) -> SearchResult:
        """Convert Chroma get results to SearchResult."""
        documents: list[Document] = []
        ids_list = results.get("ids", [])
        docs_list = results.get("documents", [])
        metas_list = results.get("metadatas", [])

        for i, doc_id in enumerate(ids_list):
            content = docs_list[i] if i < len(docs_list) and docs_list[i] else ""
            metadata = metas_list[i] if i < len(metas_list) else {}

            documents.append(Document(
                id=str(doc_id),
                content=content,
                metadata=metadata or {},
                collection=collection,
            ))

        return SearchResult(documents=documents, total_count=len(documents))

    @staticmethod
    def _infer_type(value: Any) -> DataType:
        """Infer DataType from a sample value."""
        if isinstance(value, bool):
            return DataType.BOOL
        if isinstance(value, int):
            return DataType.INT
        if isinstance(value, float):
            return DataType.FLOAT
        return DataType.TEXT
