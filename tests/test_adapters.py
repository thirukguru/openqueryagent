"""Tests for adapter implementations — Qdrant, Milvus, pgvector.

Uses mocked clients to test adapter logic without requiring real databases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openqueryagent.adapters.base import ConnectionConfig
from openqueryagent.core.exceptions import AdapterConnectionError
from openqueryagent.core.types import SearchType

# ===========================================================================
# Qdrant Adapter Tests
# ===========================================================================


class TestQdrantAdapter:
    def test_properties(self) -> None:
        from openqueryagent.adapters.qdrant import QdrantAdapter

        adapter = QdrantAdapter(adapter_id="test-qdrant")
        assert adapter.adapter_id == "test-qdrant"
        assert adapter.adapter_name == "qdrant"
        assert adapter.supports_native_aggregation is False

    def test_not_connected_raises(self) -> None:
        from openqueryagent.adapters.qdrant import QdrantAdapter

        adapter = QdrantAdapter()
        with pytest.raises(AdapterConnectionError, match="Not connected"):
            adapter._ensure_connected()

    def test_filter_compiler(self) -> None:
        from openqueryagent.adapters.qdrant import QdrantAdapter
        from openqueryagent.adapters.qdrant_filters import QdrantFilterCompiler

        adapter = QdrantAdapter()
        compiler = adapter.get_filter_compiler()
        assert isinstance(compiler, QdrantFilterCompiler)

    @pytest.mark.asyncio
    async def test_connect_missing_dependency(self) -> None:
        from openqueryagent.adapters.qdrant import QdrantAdapter

        adapter = QdrantAdapter()
        with patch.dict("sys.modules", {"qdrant_client": None}), pytest.raises(
            AdapterConnectionError, match="qdrant-client"
        ):
            await adapter.connect(ConnectionConfig())

    @pytest.mark.asyncio
    async def test_disconnect_no_op_when_not_connected(self) -> None:
        from openqueryagent.adapters.qdrant import QdrantAdapter

        adapter = QdrantAdapter()
        await adapter.disconnect()  # Should not raise

    @pytest.mark.asyncio
    async def test_health_check_connected(self) -> None:
        from openqueryagent.adapters.qdrant import QdrantAdapter

        adapter = QdrantAdapter()
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        adapter._client = mock_client

        status = await adapter.health_check()
        assert status.healthy is True
        assert status.adapter_id == "qdrant"

    @pytest.mark.asyncio
    async def test_get_collections(self) -> None:
        from openqueryagent.adapters.qdrant import QdrantAdapter

        adapter = QdrantAdapter()
        mock_client = AsyncMock()

        # Mock collection objects
        mock_col1 = MagicMock()
        mock_col1.name = "products"
        mock_col2 = MagicMock()
        mock_col2.name = "reviews"
        mock_response = MagicMock()
        mock_response.collections = [mock_col1, mock_col2]
        mock_client.get_collections.return_value = mock_response
        adapter._client = mock_client

        collections = await adapter.get_collections()
        assert collections == ["products", "reviews"]

    @pytest.mark.asyncio
    async def test_search_vector(self) -> None:
        from openqueryagent.adapters.qdrant import QdrantAdapter

        adapter = QdrantAdapter()
        mock_client = AsyncMock()

        mock_point = MagicMock()
        mock_point.id = "point-1"
        mock_point.payload = {"content": "test doc", "brand": "Nike"}
        mock_point.score = 0.95
        mock_client.search.return_value = [mock_point]
        adapter._client = mock_client

        result = await adapter.search(
            collection="products",
            query_vector=[0.1, 0.2, 0.3],
            search_type=SearchType.VECTOR,
        )

        assert len(result.documents) == 1
        assert result.documents[0].id == "point-1"
        assert result.documents[0].content == "test doc"
        assert result.documents[0].score == 0.95

    @pytest.mark.asyncio
    async def test_get_by_ids(self) -> None:
        from openqueryagent.adapters.qdrant import QdrantAdapter

        adapter = QdrantAdapter()
        mock_client = AsyncMock()

        mock_point = MagicMock()
        mock_point.id = "id-1"
        mock_point.payload = {"content": "doc 1", "price": 99}
        mock_client.retrieve.return_value = [mock_point]
        adapter._client = mock_client

        docs = await adapter.get_by_ids("products", ["id-1"])
        assert len(docs) == 1
        assert docs[0].id == "id-1"
        assert docs[0].content == "doc 1"


# ===========================================================================
# Milvus Adapter Tests
# ===========================================================================


class TestMilvusAdapter:
    def test_properties(self) -> None:
        from openqueryagent.adapters.milvus import MilvusAdapter

        adapter = MilvusAdapter(adapter_id="test-milvus")
        assert adapter.adapter_id == "test-milvus"
        assert adapter.adapter_name == "milvus"
        assert adapter.supports_native_aggregation is False

    def test_not_connected_raises(self) -> None:
        from openqueryagent.adapters.milvus import MilvusAdapter

        adapter = MilvusAdapter()
        with pytest.raises(AdapterConnectionError, match="Not connected"):
            adapter._ensure_connected()

    def test_filter_compiler(self) -> None:
        from openqueryagent.adapters.milvus import MilvusAdapter
        from openqueryagent.adapters.milvus_filters import MilvusFilterCompiler

        adapter = MilvusAdapter()
        compiler = adapter.get_filter_compiler()
        assert isinstance(compiler, MilvusFilterCompiler)

    @pytest.mark.asyncio
    async def test_get_collections(self) -> None:
        from openqueryagent.adapters.milvus import MilvusAdapter

        adapter = MilvusAdapter()
        mock_client = MagicMock()
        mock_client.list_collections.return_value = ["products", "users"]
        adapter._client = mock_client

        collections = await adapter.get_collections()
        assert collections == ["products", "users"]

    @pytest.mark.asyncio
    async def test_health_check_connected(self) -> None:
        from openqueryagent.adapters.milvus import MilvusAdapter

        adapter = MilvusAdapter()
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        adapter._client = mock_client

        status = await adapter.health_check()
        assert status.healthy is True

    @pytest.mark.asyncio
    async def test_search_query_fallback(self) -> None:
        from openqueryagent.adapters.milvus import MilvusAdapter

        adapter = MilvusAdapter()
        mock_client = MagicMock()
        mock_client.query.return_value = [
            {"id": "1", "content": "doc 1", "price": 99},
        ]
        adapter._client = mock_client

        result = await adapter.search(
            collection="products",
            filters='price < 100',
            search_type=SearchType.KEYWORD,
        )
        assert len(result.documents) == 1
        assert result.documents[0].id == "1"

    @pytest.mark.asyncio
    async def test_get_by_ids(self) -> None:
        from openqueryagent.adapters.milvus import MilvusAdapter

        adapter = MilvusAdapter()
        mock_client = MagicMock()
        mock_client.get.return_value = [
            {"id": "1", "content": "doc 1"},
        ]
        adapter._client = mock_client

        docs = await adapter.get_by_ids("products", ["1"])
        assert len(docs) == 1
        assert docs[0].content == "doc 1"


# ===========================================================================
# pgvector Adapter Tests
# ===========================================================================


class TestPgvectorAdapter:
    def test_properties(self) -> None:
        from openqueryagent.adapters.pgvector import PgvectorAdapter

        adapter = PgvectorAdapter(adapter_id="test-pg")
        assert adapter.adapter_id == "test-pg"
        assert adapter.adapter_name == "pgvector"
        assert adapter.supports_native_aggregation is True

    def test_not_connected_raises(self) -> None:
        from openqueryagent.adapters.pgvector import PgvectorAdapter

        adapter = PgvectorAdapter()
        with pytest.raises(AdapterConnectionError, match="Not connected"):
            adapter._ensure_connected()

    def test_filter_compiler(self) -> None:
        from openqueryagent.adapters.pgvector import PgvectorAdapter
        from openqueryagent.adapters.pgvector_filters import PgvectorFilterCompiler

        adapter = PgvectorAdapter()
        compiler = adapter.get_filter_compiler()
        assert isinstance(compiler, PgvectorFilterCompiler)

    @pytest.mark.asyncio
    async def test_connect_missing_dependency(self) -> None:
        from openqueryagent.adapters.pgvector import PgvectorAdapter

        adapter = PgvectorAdapter()
        with patch.dict("sys.modules", {"asyncpg": None}), pytest.raises(
            AdapterConnectionError, match="asyncpg"
        ):
            await adapter.connect(ConnectionConfig())

    @pytest.mark.asyncio
    async def test_health_check_connected(self) -> None:
        from openqueryagent.adapters.pgvector import PgvectorAdapter

        adapter = PgvectorAdapter()
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1

        # asyncpg pool uses async context manager
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        adapter._pool = mock_pool

        status = await adapter.health_check()
        assert status.healthy is True

    @pytest.mark.asyncio
    async def test_disconnect_no_op_when_not_connected(self) -> None:
        from openqueryagent.adapters.pgvector import PgvectorAdapter

        adapter = PgvectorAdapter()
        await adapter.disconnect()  # Should not raise
