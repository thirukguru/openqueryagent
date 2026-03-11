"""Tests for the Schema Inspector — caching, TTL, multi-adapter aggregation."""

from __future__ import annotations

from typing import Any

import pytest

from openqueryagent.adapters.base import (
    ConnectionConfig,
    HealthStatus,
)
from openqueryagent.core.schema import SchemaInspector
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

# ---------------------------------------------------------------------------
# Mock Adapter
# ---------------------------------------------------------------------------


class MockAdapter:
    """A mock adapter for testing schema inspector."""

    def __init__(
        self,
        adapter_id: str,
        collections: dict[str, CollectionSchema],
    ) -> None:
        self._adapter_id = adapter_id
        self._collections = collections
        self.schema_call_count = 0
        self.collection_call_count = 0

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_name(self) -> str:
        return self._adapter_id

    @property
    def supports_native_aggregation(self) -> bool:
        return False

    async def connect(self, config: ConnectionConfig) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def health_check(self) -> HealthStatus:
        return HealthStatus(healthy=True, adapter_id=self._adapter_id, adapter_name=self._adapter_id)

    async def get_collections(self) -> list[str]:
        self.collection_call_count += 1
        return list(self._collections.keys())

    async def get_schema(self, collection: str) -> CollectionSchema:
        self.schema_call_count += 1
        if collection not in self._collections:
            msg = f"Collection {collection} not found"
            raise ValueError(msg)
        return self._collections[collection]

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

    async def get_by_ids(self, collection: str, ids: list[str]) -> list[Document]:
        return []

    def get_filter_compiler(self) -> Any:
        return None


class FailingAdapter(MockAdapter):
    """An adapter that fails on get_collections."""

    async def get_collections(self) -> list[str]:
        msg = "Connection refused"
        raise ConnectionError(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_adapter(adapter_id: str, collections: list[str]) -> MockAdapter:
    schemas = {}
    for name in collections:
        schemas[name] = CollectionSchema(
            name=name,
            adapter_id=adapter_id,
            properties=[
                PropertySchema(name="name", data_type=DataType.TEXT),
                PropertySchema(name="price", data_type=DataType.FLOAT),
            ],
        )
    return MockAdapter(adapter_id=adapter_id, collections=schemas)


class TestSchemaInspector:
    @pytest.mark.asyncio
    async def test_single_adapter(self) -> None:
        adapter = _make_adapter("qdrant-1", ["products", "reviews"])
        inspector = SchemaInspector([adapter])

        schema_map = await inspector.get_schema_map()
        assert "products" in schema_map.collections
        assert "reviews" in schema_map.collections
        assert schema_map.adapter_mapping["products"] == "qdrant-1"

    @pytest.mark.asyncio
    async def test_multiple_adapters(self) -> None:
        adapter1 = _make_adapter("qdrant-1", ["products"])
        adapter2 = _make_adapter("pgvector-1", ["inventory"])

        inspector = SchemaInspector([adapter1, adapter2])
        schema_map = await inspector.get_schema_map()

        assert len(schema_map.collections) == 2
        assert schema_map.adapter_mapping["products"] == "qdrant-1"
        assert schema_map.adapter_mapping["inventory"] == "pgvector-1"

    @pytest.mark.asyncio
    async def test_caching(self) -> None:
        adapter = _make_adapter("qdrant-1", ["products"])
        inspector = SchemaInspector([adapter], cache_ttl_seconds=300)

        # First call — fetches from adapter
        await inspector.get_schema_map()
        assert adapter.collection_call_count == 1

        # Second call — should use cache
        await inspector.get_schema_map()
        assert adapter.collection_call_count == 1  # Still 1

    @pytest.mark.asyncio
    async def test_cache_invalidation(self) -> None:
        adapter = _make_adapter("qdrant-1", ["products"])
        inspector = SchemaInspector([adapter])

        await inspector.get_schema_map()
        assert adapter.collection_call_count == 1

        inspector.invalidate()
        assert inspector.is_stale

        await inspector.get_schema_map()
        assert adapter.collection_call_count == 2

    @pytest.mark.asyncio
    async def test_get_schema_by_name(self) -> None:
        adapter = _make_adapter("qdrant-1", ["products"])
        inspector = SchemaInspector([adapter])

        schema = await inspector.get_schema("products")
        assert schema is not None
        assert schema.name == "products"

    @pytest.mark.asyncio
    async def test_get_schema_unknown(self) -> None:
        adapter = _make_adapter("qdrant-1", ["products"])
        inspector = SchemaInspector([adapter])

        schema = await inspector.get_schema("nonexistent")
        assert schema is None

    @pytest.mark.asyncio
    async def test_get_adapter_for_collection(self) -> None:
        adapter = _make_adapter("qdrant-1", ["products"])
        inspector = SchemaInspector([adapter])

        adapter_id = await inspector.get_adapter_for_collection("products")
        assert adapter_id == "qdrant-1"

        adapter_id = await inspector.get_adapter_for_collection("unknown")
        assert adapter_id is None

    @pytest.mark.asyncio
    async def test_failing_adapter_graceful(self) -> None:
        """A failing adapter should not prevent other adapters from loading."""
        good_adapter = _make_adapter("qdrant-1", ["products"])
        failing_adapter = FailingAdapter(adapter_id="bad-1", collections={})

        inspector = SchemaInspector([good_adapter, failing_adapter])
        schema_map = await inspector.get_schema_map()

        # Good adapter's collections should still be present
        assert "products" in schema_map.collections
        assert len(schema_map.collections) == 1

    @pytest.mark.asyncio
    async def test_is_stale_initially(self) -> None:
        adapter = _make_adapter("qdrant-1", ["products"])
        inspector = SchemaInspector([adapter])
        assert inspector.is_stale

    @pytest.mark.asyncio
    async def test_not_stale_after_refresh(self) -> None:
        adapter = _make_adapter("qdrant-1", ["products"])
        inspector = SchemaInspector([adapter], cache_ttl_seconds=300)
        await inspector.refresh()
        assert not inspector.is_stale
