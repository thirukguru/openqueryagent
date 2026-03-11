"""Tests for PineconeAdapter and PineconeFilterCompiler."""

from __future__ import annotations

import pytest

from openqueryagent.adapters.pinecone import PineconeAdapter
from openqueryagent.adapters.pinecone_filters import PineconeFilterCompiler
from openqueryagent.core.exceptions import AdapterConnectionError, FilterCompilationError
from openqueryagent.core.types import (
    CollectionSchema,
    FilterExpression,
    FilterOperator,
    PropertySchema,
)


class TestPineconeAdapter:
    def test_properties(self) -> None:
        adapter = PineconeAdapter(adapter_id="test-pinecone")
        assert adapter.adapter_id == "test-pinecone"
        assert adapter.adapter_name == "pinecone"
        assert adapter.supports_native_aggregation is False

    def test_not_connected_raises(self) -> None:
        adapter = PineconeAdapter()
        with pytest.raises(AdapterConnectionError, match="Not connected"):
            adapter._ensure_connected()

    def test_filter_compiler(self) -> None:
        adapter = PineconeAdapter()
        compiler = adapter.get_filter_compiler()
        assert compiler is not None

    @pytest.mark.asyncio
    async def test_search_no_vector(self) -> None:
        adapter = PineconeAdapter()
        adapter._index = True  # Mock connected state
        result = await adapter.search("test", query_vector=None)
        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        adapter = PineconeAdapter()
        adapter._client = "mock"
        adapter._index = "mock"
        await adapter.disconnect()
        assert adapter._client is None
        assert adapter._index is None


class TestPineconeFilterCompiler:
    def _schema(self) -> CollectionSchema:
        return CollectionSchema(
            name="test",
            adapter_id="pinecone",
            properties=[
                PropertySchema(name="price", data_type="float"),
                PropertySchema(name="category", data_type="text"),
            ],
        )

    def test_eq_filter(self) -> None:
        compiler = PineconeFilterCompiler()
        expr = FilterExpression(field="price", operator=FilterOperator.EQ, value=100)
        result = compiler.compile(expr, self._schema())
        assert result == {"price": {"$eq": 100}}

    def test_lt_filter(self) -> None:
        compiler = PineconeFilterCompiler()
        expr = FilterExpression(field="price", operator=FilterOperator.LT, value=50)
        result = compiler.compile(expr, self._schema())
        assert result == {"price": {"$lt": 50}}

    def test_in_filter(self) -> None:
        compiler = PineconeFilterCompiler()
        expr = FilterExpression(field="category", operator=FilterOperator.IN, value=["a", "b"])
        result = compiler.compile(expr, self._schema())
        assert result == {"category": {"$in": ["a", "b"]}}

    def test_and_filter(self) -> None:
        compiler = PineconeFilterCompiler()
        expr = FilterExpression(
            operator=FilterOperator.AND,
            children=[
                FilterExpression(field="price", operator=FilterOperator.GT, value=10),
                FilterExpression(field="price", operator=FilterOperator.LT, value=100),
            ],
        )
        result = compiler.compile(expr, self._schema())
        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_between_filter(self) -> None:
        compiler = PineconeFilterCompiler()
        expr = FilterExpression(field="price", operator=FilterOperator.BETWEEN, value=[10, 100])
        result = compiler.compile(expr, self._schema())
        assert "$and" in result

    def test_unsupported_operator(self) -> None:
        compiler = PineconeFilterCompiler()
        expr = FilterExpression(field="category", operator=FilterOperator.CONTAINS, value="test")
        with pytest.raises(FilterCompilationError):
            compiler.compile(expr, self._schema())

    def test_validate_unknown_field(self) -> None:
        compiler = PineconeFilterCompiler()
        expr = FilterExpression(field="nonexistent", operator=FilterOperator.EQ, value=1)
        errors = compiler.validate(expr, self._schema())
        assert len(errors) == 1

    def test_or_filter(self) -> None:
        compiler = PineconeFilterCompiler()
        expr = FilterExpression(
            operator=FilterOperator.OR,
            children=[
                FilterExpression(field="price", operator=FilterOperator.EQ, value=10),
                FilterExpression(field="price", operator=FilterOperator.EQ, value=20),
            ],
        )
        result = compiler.compile(expr, self._schema())
        assert "$or" in result
