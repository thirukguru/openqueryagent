"""Elasticsearch adapter — connects via elasticsearch[async].

Implements the VectorStoreAdapter protocol for Elasticsearch.
Requires ``elasticsearch[async]`` (install with ``pip install openqueryagent[elasticsearch]``).
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
from openqueryagent.adapters.elasticsearch_filters import ElasticsearchFilterCompiler
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
    VectorConfig,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Connection Config
# ---------------------------------------------------------------------------


class ElasticsearchConnectionConfig(ConnectionConfig):
    """Connection configuration for Elasticsearch."""

    hosts: list[str] | None = None
    url: str = "http://localhost:9200"
    api_key: str | None = None
    username: str | None = None
    password: str | None = None
    verify_certs: bool = True
    ca_certs: str | None = None


# ---------------------------------------------------------------------------
# Type Mapping
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, DataType] = {
    "text": DataType.TEXT,
    "keyword": DataType.TEXT,
    "long": DataType.INT,
    "integer": DataType.INT,
    "short": DataType.INT,
    "byte": DataType.INT,
    "double": DataType.FLOAT,
    "float": DataType.FLOAT,
    "half_float": DataType.FLOAT,
    "scaled_float": DataType.FLOAT,
    "boolean": DataType.BOOL,
    "date": DataType.DATE,
    "geo_point": DataType.GEO,
    "geo_shape": DataType.GEO,
    "object": DataType.OBJECT,
}


class ElasticsearchAdapter:
    """Elasticsearch vector store adapter.

    Uses ``elasticsearch.AsyncElasticsearch`` for async operations.
    Supports kNN vector search, BM25 keyword search, hybrid (bool query
    combining kNN + match), and native aggregation framework.
    """

    def __init__(self, adapter_id: str = "elasticsearch") -> None:
        self._adapter_id = adapter_id
        self._client: Any = None
        self._filter_compiler = ElasticsearchFilterCompiler()

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_name(self) -> str:
        return "elasticsearch"

    @property
    def supports_native_aggregation(self) -> bool:
        return True  # Elasticsearch has native aggregation framework

    async def connect(self, config: ConnectionConfig) -> None:
        """Connect to Elasticsearch."""
        try:
            from elasticsearch import AsyncElasticsearch
        except ImportError as e:
            raise AdapterConnectionError(
                "elasticsearch is not installed. Install with: "
                "pip install openqueryagent[elasticsearch]",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

        if not isinstance(config, ElasticsearchConnectionConfig):
            config = ElasticsearchConnectionConfig(**config.model_dump())

        try:
            connect_kwargs: dict[str, Any] = {
                "request_timeout": int(config.timeout_seconds),
                "verify_certs": config.verify_certs,
            }

            hosts = config.hosts or [config.url]
            connect_kwargs["hosts"] = hosts

            if config.api_key:
                connect_kwargs["api_key"] = config.api_key
            elif config.username and config.password:
                connect_kwargs["basic_auth"] = (config.username, config.password)

            if config.ca_certs:
                connect_kwargs["ca_certs"] = config.ca_certs

            self._client = AsyncElasticsearch(**connect_kwargs)
            logger.info("elasticsearch_connected", hosts=hosts)
        except Exception as e:
            raise AdapterConnectionError(
                f"Failed to connect to Elasticsearch: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

    async def disconnect(self) -> None:
        """Disconnect from Elasticsearch."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("elasticsearch_disconnected")

    async def health_check(self) -> HealthStatus:
        """Check Elasticsearch cluster health."""
        self._ensure_connected()
        start = time.monotonic()
        try:
            health = await self._client.cluster.health()
            latency = (time.monotonic() - start) * 1000
            return HealthStatus(
                healthy=health["status"] in ("green", "yellow"),
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                latency_ms=latency,
                details={"status": health["status"], "cluster_name": health.get("cluster_name", "")},
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
        """List all Elasticsearch indices (mapped to collections)."""
        self._ensure_connected()
        try:
            indices = await self._client.cat.indices(format="json")
            return [idx["index"] for idx in indices if not idx["index"].startswith(".")]
        except Exception as e:
            raise AdapterQueryError(
                f"Failed to list indices: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection="",
            ) from e

    async def get_schema(self, collection: str) -> CollectionSchema:
        """Introspect index mapping to build schema."""
        self._ensure_connected()
        try:
            mapping = await self._client.indices.get_mapping(index=collection)
        except Exception as e:
            raise SchemaError(
                f"Failed to get schema for '{collection}': {e}",
                collection=collection,
                adapter_id=self._adapter_id,
            ) from e

        properties: list[PropertySchema] = []
        vector_cfg = None
        index_mapping = mapping.get(collection, {}).get("mappings", {}).get("properties", {})

        for field_name, field_info in index_mapping.items():
            field_type = field_info.get("type", "object")

            if field_type == "dense_vector":
                # This is a vector field
                dims = field_info.get("dims", 0)
                similarity = field_info.get("similarity", "cosine")
                from openqueryagent.core.types import DistanceMetric
                dist_map: dict[str, DistanceMetric] = {
                    "cosine": DistanceMetric.COSINE,
                    "l2_norm": DistanceMetric.EUCLIDEAN,
                    "dot_product": DistanceMetric.DOT_PRODUCT,
                }
                vector_cfg = VectorConfig(
                    dimensions=dims,
                    distance_metric=dist_map.get(similarity, DistanceMetric.COSINE),
                )
                continue

            dt = _TYPE_MAP.get(field_type, DataType.TEXT)
            properties.append(PropertySchema(
                name=field_name,
                data_type=dt,
                filterable=field_type != "text",
                searchable=field_type in ("text", "keyword"),
            ))

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
        """Execute search on Elasticsearch index."""
        self._ensure_connected()

        try:
            body: dict[str, Any] = {"size": limit, "from": offset}

            filter_clause = (filters if isinstance(filters, dict) else {}) if filters else None

            if search_type == SearchType.VECTOR and query_vector:
                body["knn"] = {
                    "field": self._vector_field(search_params),
                    "query_vector": query_vector,
                    "k": limit,
                    "num_candidates": limit * 10,
                }
                if filter_clause:
                    body["knn"]["filter"] = filter_clause

            elif search_type == SearchType.KEYWORD and query_text:
                query: dict[str, Any] = {"multi_match": {"query": query_text, "fields": ["*"]}}
                if filter_clause:
                    body["query"] = {"bool": {"must": [query], "filter": [filter_clause]}}
                else:
                    body["query"] = query

            elif search_type == SearchType.HYBRID and query_vector and query_text:
                body["knn"] = {
                    "field": self._vector_field(search_params),
                    "query_vector": query_vector,
                    "k": limit,
                    "num_candidates": limit * 10,
                }
                query = {"multi_match": {"query": query_text, "fields": ["*"]}}
                if filter_clause:
                    body["query"] = {"bool": {"must": [query], "filter": [filter_clause]}}
                    body["knn"]["filter"] = filter_clause
                else:
                    body["query"] = query

            elif query_vector:
                body["knn"] = {
                    "field": self._vector_field(search_params),
                    "query_vector": query_vector,
                    "k": limit,
                    "num_candidates": limit * 10,
                }
            elif query_text:
                query = {"multi_match": {"query": query_text, "fields": ["*"]}}
                body["query"] = query
            else:
                body["query"] = {"match_all": {}}

            response = await self._client.search(index=collection, body=body)

            documents: list[Document] = []
            for hit in response["hits"]["hits"]:
                source = hit.get("_source", {})
                documents.append(Document(
                    id=hit["_id"],
                    content=str(source.get("content", source.get("text", ""))),
                    metadata={k: v for k, v in source.items() if k not in ("content", "text")},
                    score=hit.get("_score", 0.0),
                    collection=collection,
                ))

            total = response["hits"]["total"]
            total_count = total["value"] if isinstance(total, dict) else total

            return SearchResult(
                documents=documents,
                total_count=total_count,
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
        """Execute aggregation using Elasticsearch's native agg framework."""
        self._ensure_connected()
        try:
            body: dict[str, Any] = {"size": 0}  # No docs, only aggregation

            if filters:
                body["query"] = {"bool": {"filter": [filters]}}

            op = aggregation.operation.lower()
            field = aggregation.field or "_id"

            if op == "count":
                body["aggs"] = {"result": {"value_count": {"field": field}}}
            elif op == "avg":
                body["aggs"] = {"result": {"avg": {"field": field}}}
            elif op == "sum":
                body["aggs"] = {"result": {"sum": {"field": field}}}
            elif op == "min":
                body["aggs"] = {"result": {"min": {"field": field}}}
            elif op == "max":
                body["aggs"] = {"result": {"max": {"field": field}}}
            elif op == "group_by" and aggregation.group_by:
                body["aggs"] = {"result": {"terms": {"field": aggregation.group_by, "size": 100}}}

            response = await self._client.search(index=collection, body=body)

            result_values: dict[str, Any] = {}
            agg_result = response.get("aggregations", {}).get("result", {})

            if op == "group_by":
                buckets = agg_result.get("buckets", [])
                result_values["buckets"] = [
                    {"key": b["key"], "doc_count": b["doc_count"]} for b in buckets
                ]
            elif "value" in agg_result:
                result_values[op] = agg_result["value"]

            return AggregationResult(
                values=result_values,
                is_approximate=False,
                total_count=response["hits"]["total"]["value"]
                if isinstance(response["hits"]["total"], dict)
                else response["hits"]["total"],
            )
        except Exception as e:
            raise AdapterQueryError(
                f"Aggregation failed on '{collection}': {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection=collection,
            ) from e

    async def get_by_ids(self, collection: str, ids: list[str]) -> list[Document]:
        """Retrieve documents by ID from Elasticsearch."""
        self._ensure_connected()
        try:
            response = await self._client.mget(index=collection, body={"ids": ids})
            documents: list[Document] = []
            for doc in response.get("docs", []):
                if doc.get("found"):
                    source = doc.get("_source", {})
                    documents.append(Document(
                        id=doc["_id"],
                        content=str(source.get("content", source.get("text", ""))),
                        metadata={k: v for k, v in source.items() if k not in ("content", "text")},
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
        """Get the Elasticsearch filter compiler."""
        return self._filter_compiler

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if self._client is None:
            raise AdapterConnectionError(
                "Not connected to Elasticsearch. Call connect() first.",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            )

    @staticmethod
    def _vector_field(search_params: dict[str, Any] | None) -> str:
        """Get the vector field name from search params or default."""
        if search_params and "vector_field" in search_params:
            return str(search_params["vector_field"])
        return "embedding"
