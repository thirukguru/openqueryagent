"""Protocol conformance tests — skeletons for adapter implementations.

These tests verify that concrete adapter, LLM, and embedding implementations
conform to the defined protocols. They are designed as base test classes
that adapter implementations extend in Sprint 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from openqueryagent.adapters.base import (
    ConnectionConfig,
    FilterCompiler,
    FilterValidationError,
    HealthStatus,
    VectorStoreAdapter,
)
from openqueryagent.core.types import (
    AggregationQuery,
    AggregationResult,
    ChatMessage,
    CollectionSchema,
    Document,
    FilterExpression,
    FilterOperator,
    SearchResult,
    SearchType,
    TokenUsage,
)
from openqueryagent.embeddings.base import EmbeddingProvider
from openqueryagent.llm.base import LLMChunk, LLMProvider, LLMResponse, ResponseFormat

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# ---------------------------------------------------------------------------
# Stub Implementations (for protocol conformance testing)
# ---------------------------------------------------------------------------


class StubFilterCompiler:
    """Minimal FilterCompiler implementation for conformance testing."""

    def compile(
        self,
        expression: FilterExpression,
        schema: CollectionSchema,
    ) -> Any:
        return {"stub": True}

    def validate(
        self,
        expression: FilterExpression,
        schema: CollectionSchema,
    ) -> list[FilterValidationError]:
        return []


class StubAdapter:
    """Minimal VectorStoreAdapter implementation for conformance testing."""

    @property
    def adapter_id(self) -> str:
        return "stub-1"

    @property
    def adapter_name(self) -> str:
        return "stub"

    @property
    def supports_native_aggregation(self) -> bool:
        return False

    async def connect(self, config: ConnectionConfig) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def health_check(self) -> HealthStatus:
        return HealthStatus(
            healthy=True, adapter_id="stub-1", adapter_name="stub",
        )

    async def get_collections(self) -> list[str]:
        return ["test_collection"]

    async def get_schema(self, collection: str) -> CollectionSchema:
        return CollectionSchema(name=collection, adapter_id="stub-1")

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
        return SearchResult()

    async def aggregate(
        self,
        collection: str,
        aggregation: AggregationQuery,
        filters: Any | None = None,
    ) -> AggregationResult:
        return AggregationResult(values={})

    async def get_by_ids(
        self,
        collection: str,
        ids: list[str],
    ) -> list[Document]:
        return []

    def get_filter_compiler(self) -> FilterCompiler:
        return StubFilterCompiler()


class StubLLMProvider:
    """Minimal LLMProvider implementation for conformance testing."""

    @property
    def model_name(self) -> str:
        return "stub-model"

    async def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content="stub response",
            model="stub-model",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
        )

    async def complete_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[LLMChunk]:
        yield LLMChunk(content="stub", model="stub-model", finish_reason="stop")


class StubEmbeddingProvider:
    """Minimal EmbeddingProvider implementation for conformance testing."""

    @property
    def dimensions(self) -> int:
        return 3

    @property
    def model_name(self) -> str:
        return "stub-embedding"

    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


# ---------------------------------------------------------------------------
# Protocol Conformance Tests
# ---------------------------------------------------------------------------


class TestVectorStoreAdapterProtocol:
    """Verify StubAdapter conforms to VectorStoreAdapter protocol."""

    def test_isinstance_check(self) -> None:
        adapter = StubAdapter()
        assert isinstance(adapter, VectorStoreAdapter)

    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        adapter = StubAdapter()
        await adapter.connect(ConnectionConfig())
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        adapter = StubAdapter()
        status = await adapter.health_check()
        assert status.healthy is True
        assert status.adapter_id == "stub-1"

    @pytest.mark.asyncio
    async def test_get_collections(self) -> None:
        adapter = StubAdapter()
        collections = await adapter.get_collections()
        assert isinstance(collections, list)
        assert len(collections) > 0

    @pytest.mark.asyncio
    async def test_get_schema(self) -> None:
        adapter = StubAdapter()
        schema = await adapter.get_schema("test_collection")
        assert schema.name == "test_collection"
        assert schema.adapter_id == "stub-1"

    @pytest.mark.asyncio
    async def test_search(self) -> None:
        adapter = StubAdapter()
        result = await adapter.search("test_collection", query_text="test query")
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_aggregate(self) -> None:
        adapter = StubAdapter()
        result = await adapter.aggregate(
            "test_collection",
            AggregationQuery(operation="count"),
        )
        assert isinstance(result, AggregationResult)

    @pytest.mark.asyncio
    async def test_get_by_ids(self) -> None:
        adapter = StubAdapter()
        docs = await adapter.get_by_ids("test_collection", ["1", "2"])
        assert isinstance(docs, list)

    def test_supports_native_aggregation(self) -> None:
        adapter = StubAdapter()
        assert adapter.supports_native_aggregation is False


class TestFilterCompilerProtocol:
    """Verify StubFilterCompiler conforms to FilterCompiler protocol."""

    def test_isinstance_check(self) -> None:
        compiler = StubFilterCompiler()
        assert isinstance(compiler, FilterCompiler)

    def test_compile(self) -> None:
        compiler = StubFilterCompiler()
        expr = FilterExpression(operator=FilterOperator.EQ, field="x", value=1)
        schema = CollectionSchema(name="test", adapter_id="stub")
        result = compiler.compile(expr, schema)
        assert result is not None

    def test_validate(self) -> None:
        compiler = StubFilterCompiler()
        expr = FilterExpression(operator=FilterOperator.EQ, field="x", value=1)
        schema = CollectionSchema(name="test", adapter_id="stub")
        errors = compiler.validate(expr, schema)
        assert isinstance(errors, list)


class TestLLMProviderProtocol:
    """Verify StubLLMProvider conforms to LLMProvider protocol."""

    def test_isinstance_check(self) -> None:
        provider = StubLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_model_name(self) -> None:
        provider = StubLLMProvider()
        assert provider.model_name == "stub-model"

    @pytest.mark.asyncio
    async def test_complete(self) -> None:
        provider = StubLLMProvider()
        response = await provider.complete(
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert isinstance(response, LLMResponse)
        assert response.content == "stub response"
        assert response.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_complete_stream(self) -> None:
        provider = StubLLMProvider()
        chunks: list[LLMChunk] = []
        async for chunk in provider.complete_stream(
            messages=[ChatMessage(role="user", content="Hello")],
        ):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert chunks[0].content == "stub"


class TestEmbeddingProviderProtocol:
    """Verify StubEmbeddingProvider conforms to EmbeddingProvider protocol."""

    def test_isinstance_check(self) -> None:
        provider = StubEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_properties(self) -> None:
        provider = StubEmbeddingProvider()
        assert provider.dimensions == 3
        assert provider.model_name == "stub-embedding"

    @pytest.mark.asyncio
    async def test_embed_query(self) -> None:
        provider = StubEmbeddingProvider()
        vector = await provider.embed_query("test text")
        assert len(vector) == 3
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.asyncio
    async def test_embed_documents(self) -> None:
        provider = StubEmbeddingProvider()
        vectors = await provider.embed_documents(["text1", "text2", "text3"])
        assert len(vectors) == 3
        assert all(len(v) == 3 for v in vectors)
