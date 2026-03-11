"""Milvus adapter — connects to Milvus/Zilliz via pymilvus.

Implements the VectorStoreAdapter protocol for Milvus vector database.
Requires ``pymilvus`` (install with ``pip install openqueryagent[milvus]``).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from openqueryagent.adapters.base import (
    ConnectionConfig,
    FilterCompiler,
    HealthStatus,
)
from openqueryagent.adapters.milvus_filters import MilvusFilterCompiler
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


class MilvusConnectionConfig(ConnectionConfig):
    """Connection configuration for Milvus/Zilliz."""

    uri: str = "http://localhost:19530"
    token: str | None = None
    db_name: str = "default"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

# Milvus data type mapping
_TYPE_MAP: dict[int, DataType] = {}  # Populated at runtime from pymilvus

_DTYPE_NAME_MAP: dict[str, DataType] = {
    "VARCHAR": DataType.TEXT,
    "INT8": DataType.INT,
    "INT16": DataType.INT,
    "INT32": DataType.INT,
    "INT64": DataType.INT,
    "FLOAT": DataType.FLOAT,
    "DOUBLE": DataType.FLOAT,
    "BOOL": DataType.BOOL,
    "JSON": DataType.TEXT,
    "ARRAY": DataType.FLOAT_ARRAY,
    "FLOAT_VECTOR": DataType.FLOAT_ARRAY,
    "BINARY_VECTOR": DataType.FLOAT_ARRAY,
    "SPARSE_FLOAT_VECTOR": DataType.FLOAT_ARRAY,
}

_METRIC_MAP: dict[str, DistanceMetric] = {
    "COSINE": DistanceMetric.COSINE,
    "L2": DistanceMetric.EUCLIDEAN,
    "IP": DistanceMetric.DOT_PRODUCT,
}


class MilvusAdapter:
    """Milvus vector store adapter.

    Uses ``pymilvus.MilvusClient`` for simplified async-compatible operations.
    """

    def __init__(self, adapter_id: str = "milvus") -> None:
        self._adapter_id = adapter_id
        self._client: Any = None
        self._filter_compiler = MilvusFilterCompiler()

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_name(self) -> str:
        return "milvus"

    @property
    def supports_native_aggregation(self) -> bool:
        return False  # Milvus doesn't have native aggregation

    async def connect(self, config: ConnectionConfig) -> None:
        """Connect to Milvus server."""
        try:
            from pymilvus import MilvusClient
        except ImportError as e:
            raise AdapterConnectionError(
                "pymilvus is not installed. Install with: pip install openqueryagent[milvus]",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

        if not isinstance(config, MilvusConnectionConfig):
            config = MilvusConnectionConfig(**config.model_dump())

        try:
            self._client = MilvusClient(
                uri=config.uri,
                token=config.token or "",
                db_name=config.db_name,
                timeout=config.timeout_seconds,
            )
            logger.info("milvus_connected", uri=config.uri.split("@")[-1] if "@" in config.uri else config.uri)
        except Exception as e:
            raise AdapterConnectionError(
                f"Failed to connect to Milvus: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

    async def disconnect(self) -> None:
        """Disconnect from Milvus server."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("milvus_disconnected")

    async def health_check(self) -> HealthStatus:
        """Check Milvus server health."""
        self._ensure_connected()
        start = time.monotonic()
        try:
            await asyncio.to_thread(self._client.list_collections)
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
        """List all Milvus collections."""
        self._ensure_connected()
        try:
            collections: list[str] = await asyncio.to_thread(self._client.list_collections)
            return collections
        except Exception as e:
            raise AdapterQueryError(
                f"Failed to list collections: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection="",
            ) from e

    async def get_schema(self, collection: str) -> CollectionSchema:
        """Introspect collection schema from Milvus."""
        self._ensure_connected()
        try:
            desc = await asyncio.to_thread(self._client.describe_collection, collection)
        except Exception as e:
            raise SchemaError(
                f"Failed to get schema for '{collection}': {e}",
                collection=collection,
                adapter_id=self._adapter_id,
            ) from e

        properties: list[PropertySchema] = []
        vector_config: VectorConfig | None = None

        # desc is a dict with collection info
        fields = desc.get("fields", [])
        for field in fields:
            field_name = field.get("name", "")
            type_name = field.get("type", "").upper() if isinstance(field.get("type"), str) else ""

            # Detect vector fields
            if type_name in ("FLOAT_VECTOR", "BINARY_VECTOR"):
                dim = field.get("params", {}).get("dim", 0)
                if dim:
                    index_params = field.get("index_params", {})
                    metric = index_params.get("metric_type", "COSINE")
                    vector_config = VectorConfig(
                        dimensions=int(dim),
                        distance_metric=_METRIC_MAP.get(metric, DistanceMetric.COSINE),
                    )
                continue

            data_type = _DTYPE_NAME_MAP.get(type_name, DataType.TEXT)
            is_primary = field.get("is_primary", False)

            properties.append(PropertySchema(
                name=field_name,
                data_type=data_type,
                filterable=not is_primary,
            ))

        return CollectionSchema(
            name=collection,
            adapter_id=self._adapter_id,
            vector_config=vector_config,
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
        """Execute search on Milvus collection."""
        self._ensure_connected()
        try:
            filter_expr = filters if isinstance(filters, str) else None
            anns_field = (search_params or {}).get("anns_field", "embedding")
            output_fields = (search_params or {}).get("output_fields", ["*"])

            # Clamp limit/offset
            limit = max(1, min(limit, 1000))
            offset = max(0, min(offset, 100_000))

            if search_type == SearchType.VECTOR and query_vector:
                results = await asyncio.to_thread(
                    self._client.search,
                    collection_name=collection,
                    data=[query_vector],
                    anns_field=anns_field,
                    filter=filter_expr or "",
                    limit=limit,
                    offset=offset,
                    output_fields=output_fields,
                )
            elif search_type == SearchType.KEYWORD and filter_expr:
                # Keyword search via query with filter
                results_raw = await asyncio.to_thread(
                    self._client.query,
                    collection_name=collection,
                    filter=filter_expr,
                    limit=limit,
                    offset=offset,
                    output_fields=output_fields,
                )
                return self._query_results_to_search_result(results_raw)
            else:
                # Hybrid or fallback
                if query_vector:
                    results = await asyncio.to_thread(
                        self._client.search,
                        collection_name=collection,
                        data=[query_vector],
                        anns_field=anns_field,
                        filter=filter_expr or "",
                        limit=limit,
                        offset=offset,
                        output_fields=output_fields,
                    )
                else:
                    results_raw = await asyncio.to_thread(
                        self._client.query,
                        collection_name=collection,
                        filter=filter_expr or "id >= 0",
                        limit=limit,
                        offset=offset,
                        output_fields=output_fields,
                    )
                    return self._query_results_to_search_result(results_raw)

            return self._search_results_to_search_result(results)
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
        """Aggregate via client-side query (Milvus has no native aggregation)."""
        self._ensure_connected()
        try:
            filter_expr = filters if isinstance(filters, str) else "id >= 0"
            output_fields = [aggregation.field] if aggregation.field else ["*"]

            results = await asyncio.to_thread(
                self._client.query,
                collection_name=collection,
                filter=filter_expr,
                output_fields=output_fields,
                limit=10000,
            )

            all_values: list[Any] = []
            for row in results:
                if aggregation.field and aggregation.field in row:
                    all_values.append(row[aggregation.field])

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

            return AggregationResult(values=result_values, is_approximate=True)
        except Exception as e:
            raise AdapterQueryError(
                f"Aggregation failed on '{collection}': {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection=collection,
            ) from e

    async def get_by_ids(self, collection: str, ids: list[str]) -> list[Document]:
        """Retrieve documents by ID from Milvus."""
        self._ensure_connected()
        try:
            results = await asyncio.to_thread(
                self._client.get,
                collection_name=collection,
                ids=ids,
                output_fields=["*"],
            )
            return [
                Document(
                    id=str(row.get("id", "")),
                    content=row.get("content", row.get("text", "")),
                    metadata={k: v for k, v in row.items() if k not in ("id", "content", "text")},
                )
                for row in results
            ]
        except Exception as e:
            raise AdapterQueryError(
                f"get_by_ids failed on '{collection}': {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection=collection,
            ) from e

    def get_filter_compiler(self) -> FilterCompiler:
        """Get the Milvus filter compiler."""
        return self._filter_compiler

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if self._client is None:
            raise AdapterConnectionError(
                "Not connected to Milvus. Call connect() first.",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            )

    @staticmethod
    def _search_results_to_search_result(results: Any) -> SearchResult:
        """Convert pymilvus search results to SearchResult."""
        documents: list[Document] = []
        # results is list of lists (one per query vector)
        hits = results[0] if results else []
        for hit in hits:
            entity = hit.get("entity", hit) if isinstance(hit, dict) else {}
            doc_id = str(hit.get("id", entity.get("id", "")))
            score = hit.get("distance", 0.0) if isinstance(hit, dict) else 0.0
            content = entity.get("content", entity.get("text", ""))
            metadata = {
                k: v for k, v in entity.items()
                if k not in ("id", "content", "text", "embedding")
            }
            documents.append(Document(id=doc_id, content=content, metadata=metadata, score=score))
        return SearchResult(documents=documents, total_count=len(documents))

    @staticmethod
    def _query_results_to_search_result(results: Any) -> SearchResult:
        """Convert pymilvus query results to SearchResult."""
        documents: list[Document] = []
        for row in results:
            documents.append(Document(
                id=str(row.get("id", "")),
                content=row.get("content", row.get("text", "")),
                metadata={
                    k: v for k, v in row.items()
                    if k not in ("id", "content", "text", "embedding")
                },
            ))
        return SearchResult(documents=documents, total_count=len(documents))
