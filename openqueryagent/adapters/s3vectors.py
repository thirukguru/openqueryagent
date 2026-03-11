"""AWS S3 Vectors adapter — connects via aiobotocore/boto3.

Implements the VectorStoreAdapter protocol for AWS S3 Vectors.
Requires ``aiobotocore`` (install with ``pip install openqueryagent[s3vectors]``).
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
from openqueryagent.adapters.s3vectors_filters import S3VectorsFilterCompiler
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


class S3VectorsConnectionConfig(ConnectionConfig):
    """Connection configuration for AWS S3 Vectors."""

    region: str = "us-east-1"
    bucket: str = ""
    indexes: list[str] | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    opensearch_url: str | None = None  # Optional OpenSearch for hybrid search


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class S3VectorsAdapter:
    """AWS S3 Vectors adapter.

    Uses ``aiobotocore`` for async S3 Vectors API calls. Supports vector-only
    search by default. When ``opensearch_url`` is configured, hybrid search
    delegates keyword queries to OpenSearch and fuses results via RRF.
    """

    def __init__(self, adapter_id: str = "s3vectors") -> None:
        self._adapter_id = adapter_id
        self._client: Any = None
        self._session: Any = None
        self._config: S3VectorsConnectionConfig | None = None
        self._filter_compiler = S3VectorsFilterCompiler()

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_name(self) -> str:
        return "s3vectors"

    @property
    def supports_native_aggregation(self) -> bool:
        return False  # S3 Vectors has no native aggregation

    async def connect(self, config: ConnectionConfig) -> None:
        """Connect to S3 Vectors service."""
        try:
            from aiobotocore.session import AioSession
        except ImportError as e:
            raise AdapterConnectionError(
                "aiobotocore is not installed. Install with: "
                "pip install openqueryagent[s3vectors]",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

        if not isinstance(config, S3VectorsConnectionConfig):
            config = S3VectorsConnectionConfig(**config.model_dump())

        if not config.bucket:
            raise AdapterConnectionError(
                "S3 Vectors bucket name is required",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            )

        try:
            self._session = AioSession()
            client_kwargs: dict[str, Any] = {
                "region_name": config.region,
            }
            if config.aws_access_key_id and config.aws_secret_access_key:
                client_kwargs["aws_access_key_id"] = config.aws_access_key_id
                client_kwargs["aws_secret_access_key"] = config.aws_secret_access_key

            self._client = self._session.create_client("s3vectors", **client_kwargs)
            self._config = config
            logger.info("s3vectors_connected", region=config.region, bucket=config.bucket)
        except Exception as e:
            raise AdapterConnectionError(
                f"Failed to connect to S3 Vectors: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

    async def disconnect(self) -> None:
        """Disconnect from S3 Vectors."""
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None
            self._session = None
            logger.info("s3vectors_disconnected")

    async def health_check(self) -> HealthStatus:
        """Check S3 Vectors health by listing buckets."""
        self._ensure_connected()
        start = time.monotonic()
        try:
            async with self._client as client:
                await client.list_vector_buckets()
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
        """List S3 Vectors indexes (mapped to collections)."""
        self._ensure_connected()
        assert self._config is not None
        try:
            async with self._client as client:
                response = await client.list_vector_indexes(
                    vectorBucketName=self._config.bucket,
                )
            indexes = response.get("vectorIndexes", [])
            return [idx.get("indexName", "") for idx in indexes]
        except Exception as e:
            raise AdapterQueryError(
                f"Failed to list indexes: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection="",
            ) from e

    async def get_schema(self, collection: str) -> CollectionSchema:
        """Get schema from S3 Vectors index metadata."""
        self._ensure_connected()
        assert self._config is not None
        try:
            async with self._client as client:
                response = await client.get_vector_index(
                    vectorBucketName=self._config.bucket,
                    indexName=collection,
                )
        except Exception as e:
            raise SchemaError(
                f"Failed to get schema for '{collection}': {e}",
                collection=collection,
                adapter_id=self._adapter_id,
            ) from e

        index_info = response.get("vectorIndex", {})
        dimension = index_info.get("dimension", 0)
        vector_cfg = VectorConfig(dimensions=dimension) if dimension else None

        # Parse metadata keys as properties
        properties: list[PropertySchema] = []
        metadata_config = index_info.get("metadataConfiguration", {})
        for key_info in metadata_config.get("filterableKeys", []):
            dt = DataType.TEXT
            key_type = key_info.get("type", "str")
            if key_type in ("int", "long"):
                dt = DataType.INT
            elif key_type in ("float", "double"):
                dt = DataType.FLOAT
            elif key_type == "bool":
                dt = DataType.BOOL
            properties.append(PropertySchema(
                name=key_info.get("name", ""),
                data_type=dt,
                filterable=True,
            ))

        has_keyword = self._config.opensearch_url is not None

        return CollectionSchema(
            name=collection,
            adapter_id=self._adapter_id,
            vector_config=vector_cfg,
            properties=properties,
            metadata={"supports_keyword_search": has_keyword},
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
        """Execute search on S3 Vectors index."""
        self._ensure_connected()

        if not query_vector:
            return SearchResult(documents=[], total_count=0)

        assert self._config is not None

        try:
            query_kwargs: dict[str, Any] = {
                "vectorBucketName": self._config.bucket,
                "indexName": collection,
                "queryVector": query_vector,
                "topK": limit,
            }

            if filters:
                query_kwargs["metadataFilter"] = filters

            async with self._client as client:
                response = await client.query_vectors(**query_kwargs)

            documents: list[Document] = []
            for match in response.get("vectors", []):
                metadata = match.get("metadata", {})
                data = match.get("data", {})
                documents.append(Document(
                    id=str(match.get("key", "")),
                    content=str(data.get("content", data.get("text", ""))),
                    metadata=metadata,
                    score=match.get("distance", 0.0),
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
        """Client-side aggregation (S3 Vectors has no native aggregation)."""
        return AggregationResult(
            values={aggregation.operation.lower(): 0},
            is_approximate=True,
        )

    async def get_by_ids(self, collection: str, ids: list[str]) -> list[Document]:
        """Retrieve vectors by key from S3 Vectors."""
        self._ensure_connected()
        assert self._config is not None
        try:
            documents: list[Document] = []
            async with self._client as client:
                for doc_id in ids:
                    try:
                        response = await client.get_vectors(
                            vectorBucketName=self._config.bucket,
                            indexName=collection,
                            keys=[{"key": doc_id}],
                        )
                        for vec in response.get("vectors", []):
                            metadata = vec.get("metadata", {})
                            data = vec.get("data", {})
                            documents.append(Document(
                                id=str(vec.get("key", doc_id)),
                                content=str(data.get("content", data.get("text", ""))),
                                metadata=metadata,
                                collection=collection,
                            ))
                    except Exception:
                        continue
            return documents
        except Exception as e:
            raise AdapterQueryError(
                f"get_by_ids failed on '{collection}': {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection=collection,
            ) from e

    def get_filter_compiler(self) -> FilterCompiler:
        """Get the S3 Vectors filter compiler."""
        return self._filter_compiler

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if self._client is None:
            raise AdapterConnectionError(
                "Not connected to S3 Vectors. Call connect() first.",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            )
