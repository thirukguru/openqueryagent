"""Base protocol and types for vector store adapters.

All adapter implementations must conform to the VectorStoreAdapter protocol.
Each adapter also provides a FilterCompiler that translates the universal
filter DSL into the backend's native filter format.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from openqueryagent.core.types import (
    AggregationQuery,
    AggregationResult,
    CollectionSchema,
    Document,
    FilterExpression,
    SearchResult,
    SearchType,
)

# ---------------------------------------------------------------------------
# Supporting Types
# ---------------------------------------------------------------------------


class ConnectionConfig(BaseModel):
    """Base configuration for adapter connections.

    Subclass this for adapter-specific connection parameters.
    """

    timeout_seconds: float = 30.0
    extra: dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    """Health check response from an adapter."""

    healthy: bool
    adapter_id: str
    adapter_name: str
    message: str = ""
    latency_ms: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class FilterValidationError(BaseModel):
    """A single validation error for a filter expression."""

    field: str
    operator: str
    message: str


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class FilterCompiler(Protocol):
    """Protocol for compiling universal filters to native backend formats."""

    def compile(
        self,
        expression: FilterExpression,
        schema: CollectionSchema,
    ) -> Any:
        """Compile a FilterExpression into the backend's native filter format.

        Args:
            expression: The universal filter expression to compile.
            schema: The collection schema for validation context.

        Returns:
            The native filter object for this backend.

        Raises:
            FilterCompilationError: If the filter cannot be compiled
                (e.g., unsupported extended operator).
        """
        ...

    def validate(
        self,
        expression: FilterExpression,
        schema: CollectionSchema,
    ) -> list[FilterValidationError]:
        """Validate a filter expression against a collection schema.

        Checks for type mismatches, unknown fields, non-filterable fields,
        and unsupported operators.

        Args:
            expression: The filter expression to validate.
            schema: The collection schema to validate against.

        Returns:
            A list of validation errors (empty if valid).
        """
        ...


@runtime_checkable
class VectorStoreAdapter(Protocol):
    """Universal interface for vector database backends.

    All 8 supported backends (Qdrant, Milvus, pgvector, Weaviate,
    Pinecone, Chroma, Elasticsearch, S3 Vectors) implement this protocol.
    """

    @property
    def adapter_id(self) -> str:
        """Unique identifier for this adapter instance."""
        ...

    @property
    def adapter_name(self) -> str:
        """Human-readable name of this adapter (e.g., 'qdrant')."""
        ...

    @property
    def supports_native_aggregation(self) -> bool:
        """Whether this adapter supports server-side aggregation.

        If False, aggregation is performed client-side via scroll/get,
        which is expensive for large datasets.
        """
        ...

    async def connect(self, config: ConnectionConfig) -> None:
        """Establish connection to the vector database.

        Args:
            config: Connection configuration.

        Raises:
            AdapterConnectionError: If the connection fails.
        """
        ...

    async def disconnect(self) -> None:
        """Close the connection to the vector database."""
        ...

    async def health_check(self) -> HealthStatus:
        """Check if the adapter's backend is healthy and reachable.

        Returns:
            HealthStatus with connectivity details.
        """
        ...

    async def get_collections(self) -> list[str]:
        """List all available collections in this backend.

        Returns:
            List of collection names.
        """
        ...

    async def get_schema(self, collection: str) -> CollectionSchema:
        """Introspect the schema of a collection.

        Args:
            collection: Name of the collection.

        Returns:
            CollectionSchema with properties, vector config, etc.

        Raises:
            SchemaError: If the collection doesn't exist or schema
                introspection fails.
        """
        ...

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
        """Execute a search query against a collection.

        Args:
            collection: Target collection name.
            query_vector: Pre-computed query embedding vector.
            query_text: Raw text for server-side embedding or keyword search.
            filters: Native filter object (post-compilation by FilterCompiler).
            limit: Maximum number of results.
            offset: Pagination offset.
            search_type: Type of search (vector, keyword, or hybrid).
            search_params: Additional backend-specific search parameters.

        Returns:
            SearchResult with matching documents.

        Raises:
            AdapterQueryError: If the search fails.
        """
        ...

    async def aggregate(
        self,
        collection: str,
        aggregation: AggregationQuery,
        filters: Any | None = None,
    ) -> AggregationResult:
        """Execute an aggregation query against a collection.

        For backends without native aggregation, this uses client-side
        computation with configurable scroll limits.

        Args:
            collection: Target collection name.
            aggregation: The aggregation operation to perform.
            filters: Native filter object (post-compilation).

        Returns:
            AggregationResult with computed values.

        Raises:
            AdapterQueryError: If the aggregation fails.
        """
        ...

    async def get_by_ids(
        self,
        collection: str,
        ids: list[str],
    ) -> list[Document]:
        """Retrieve documents by their IDs.

        Args:
            collection: Target collection name.
            ids: List of document IDs to retrieve.

        Returns:
            List of documents (may be fewer than requested if some
            IDs don't exist).
        """
        ...

    def get_filter_compiler(self) -> FilterCompiler:
        """Get the filter compiler for this adapter.

        Returns:
            A FilterCompiler instance that translates universal
            FilterExpression objects to this backend's native format.
        """
        ...
