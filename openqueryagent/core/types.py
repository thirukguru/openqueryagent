"""Core type definitions for OpenQueryAgent.

All shared types used across the pipeline: documents, search results,
schemas, filters, aggregations, and response types.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SearchType(StrEnum):
    """Type of search to perform."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class QueryIntent(StrEnum):
    """Intent classification of a user query."""

    SEARCH = "search"
    AGGREGATE = "aggregate"
    HYBRID = "hybrid"
    CONVERSATIONAL = "conversational"


class DataType(StrEnum):
    """Property data types in collection schemas."""

    TEXT = "text"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    DATE = "date"
    GEO = "geo"
    TEXT_ARRAY = "text_array"
    INT_ARRAY = "int_array"
    FLOAT_ARRAY = "float_array"
    OBJECT = "object"


class DistanceMetric(StrEnum):
    """Vector similarity distance metrics."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


class FilterOperator(StrEnum):
    """Filter operators for the universal filter DSL.

    Core operators are portable across all 8 backends.
    Extended operators are supported by a subset of backends.
    """

    # Core operators (portable)
    EQ = "$eq"
    NE = "$ne"
    GT = "$gt"
    GTE = "$gte"
    LT = "$lt"
    LTE = "$lte"
    IN = "$in"
    NIN = "$nin"
    CONTAINS = "$contains"
    BETWEEN = "$between"
    EXISTS = "$exists"
    GEO_RADIUS = "$geo_radius"
    AND = "$and"
    OR = "$or"
    NOT = "$not"

    # Extended operators (subset of backends)
    STARTS_WITH = "$starts_with"
    ENDS_WITH = "$ends_with"
    REGEX = "$regex"
    NOT_CONTAINS = "$not_contains"


class ExecutionStatus(StrEnum):
    """Status of a sub-query execution."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    PARTIAL = "partial"


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class TokenUsage(BaseModel):
    """Token usage statistics from an LLM call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Document(BaseModel):
    """A document retrieved from a vector store."""

    id: str
    collection: str = ""
    content: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)
    vector: list[float] | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Result from a vector store search operation."""

    documents: list[Document] = Field(default_factory=list)
    total_count: int | None = None
    latency_ms: float = 0.0
    search_type_used: SearchType = SearchType.HYBRID


class AggregationQuery(BaseModel):
    """A structured aggregation query."""

    operation: str  # count, avg, sum, min, max, group_by
    field: str | None = None
    group_by: str | None = None


class AggregationResult(BaseModel):
    """Result from an aggregation operation."""

    values: dict[str, Any] = Field(default_factory=dict)
    is_approximate: bool = False
    scanned_count: int | None = None
    total_count: int | None = None
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Schema Models
# ---------------------------------------------------------------------------


class VectorConfig(BaseModel):
    """Vector index configuration for a collection."""

    dimensions: int
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    model_name: str | None = None


class PropertySchema(BaseModel):
    """Schema of a single property in a collection."""

    name: str
    data_type: DataType
    description: str | None = None
    filterable: bool = True
    searchable: bool = False
    vectorized: bool = False
    sample_values: list[Any] | None = None


class CollectionSchema(BaseModel):
    """Schema of a vector database collection."""

    name: str
    description: str | None = None
    adapter_id: str
    properties: list[PropertySchema] = Field(default_factory=list)
    vector_config: VectorConfig | None = None
    total_objects: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SchemaMap(BaseModel):
    """Unified schema map across all connected adapters.

    Note: adapter_mapping stores adapter_id strings (not adapter references)
    to keep the model serializable. The Agent resolves these to actual
    adapter instances at runtime.
    """

    collections: dict[str, CollectionSchema] = Field(default_factory=dict)
    adapter_mapping: dict[str, str] = Field(default_factory=dict)  # collection -> adapter_id
    last_refreshed: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Filter Models
# ---------------------------------------------------------------------------


class FilterExpression(BaseModel):
    """A node in the filter expression tree."""

    operator: FilterOperator
    field: str | None = None  # None for $and/$or/$not
    value: Any | None = None
    children: list[FilterExpression] | None = None


# ---------------------------------------------------------------------------
# Query Plan Models
# ---------------------------------------------------------------------------


class SubQuery(BaseModel):
    """A single sub-query in a query plan."""

    id: str
    collection: str
    query_text: str = ""
    search_type: SearchType = SearchType.HYBRID
    filters: FilterExpression | None = None
    aggregation: AggregationQuery | None = None
    limit: int = 10
    depends_on: list[str] | None = None
    priority: int = 0


class QueryPlan(BaseModel):
    """A decomposed query plan with one or more sub-queries."""

    original_query: str
    intent: QueryIntent
    sub_queries: list[SubQuery] = Field(default_factory=list)
    reasoning: str = ""
    requires_synthesis: bool = True


# ---------------------------------------------------------------------------
# Execution Models
# ---------------------------------------------------------------------------


class ExecutionResult(BaseModel):
    """Result of executing a single sub-query."""

    sub_query_id: str
    status: ExecutionStatus
    documents: list[Document] = Field(default_factory=list)
    aggregation_result: AggregationResult | None = None
    latency_ms: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Synthesis Models
# ---------------------------------------------------------------------------


class RankedDocument(BaseModel):
    """A document with reranking scores."""

    document: Document
    score: float = 0.0
    original_rank: int = 0
    new_rank: int = 0


class Citation(BaseModel):
    """A source citation in a synthesized answer."""

    document_id: str
    collection: str
    text_snippet: str = ""
    relevance_score: float = 0.0


class SynthesisResult(BaseModel):
    """Result from answer synthesis."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    model_used: str = ""
    tokens_used: TokenUsage | None = None


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class AskResponse(BaseModel):
    """Complete response from the ask() endpoint."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    query_plan: QueryPlan | None = None
    confidence: float = 0.0
    total_latency_ms: float = 0.0
    tokens_used: TokenUsage | None = None


class AskResponseChunk(BaseModel):
    """A single chunk in a streaming ask response."""

    text: str = ""
    stage: str = ""  # planning, searching, reranking, synthesizing
    is_final: bool = False
    citations: list[Citation] | None = None
    query_plan: QueryPlan | None = None


class SearchResponse(BaseModel):
    """Complete response from the search() endpoint."""

    documents: list[RankedDocument] = Field(default_factory=list)
    query_plan: QueryPlan | None = None
    total_latency_ms: float = 0.0


class AggregationResponse(BaseModel):
    """Complete response from the aggregate() endpoint."""

    result: AggregationResult | None = None
    query_plan: QueryPlan | None = None
    total_latency_ms: float = 0.0
