"""Tests for S3VectorsAdapter and S3VectorsFilterCompiler."""

from __future__ import annotations

import pytest

from openqueryagent.adapters.s3vectors import S3VectorsAdapter
from openqueryagent.adapters.s3vectors_filters import S3VectorsFilterCompiler
from openqueryagent.core.exceptions import AdapterConnectionError, FilterCompilationError
from openqueryagent.core.types import (
    AggregationQuery,
    CollectionSchema,
    FilterExpression,
    FilterOperator,
    PropertySchema,
)


class TestS3VectorsAdapter:
    def test_properties(self) -> None:
        adapter = S3VectorsAdapter(adapter_id="test-s3v")
        assert adapter.adapter_id == "test-s3v"
        assert adapter.adapter_name == "s3vectors"
        assert adapter.supports_native_aggregation is False

    def test_not_connected_raises(self) -> None:
        adapter = S3VectorsAdapter()
        with pytest.raises(AdapterConnectionError, match="Not connected"):
            adapter._ensure_connected()

    def test_filter_compiler(self) -> None:
        adapter = S3VectorsAdapter()
        compiler = adapter.get_filter_compiler()
        assert compiler is not None

    @pytest.mark.asyncio
    async def test_search_no_vector(self) -> None:
        adapter = S3VectorsAdapter()
        adapter._client = True  # Mock connected
        adapter._config = None  # Would fail if called
        # search without vector returns empty
        result = await adapter.search("test", query_vector=None)
        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_aggregate_stub(self) -> None:
        adapter = S3VectorsAdapter()
        result = await adapter.aggregate(
            "test",
            AggregationQuery(operation="count"),
        )
        assert result.is_approximate is True


class TestS3VectorsFilterCompiler:
    def _schema(self) -> CollectionSchema:
        return CollectionSchema(
            name="test",
            adapter_id="s3vectors",
            properties=[
                PropertySchema(name="price", data_type="float"),
                PropertySchema(name="category", data_type="text"),
            ],
        )

    def test_eq_filter(self) -> None:
        compiler = S3VectorsFilterCompiler()
        expr = FilterExpression(field="category", operator=FilterOperator.EQ, value="electronics")
        result = compiler.compile(expr, self._schema())
        assert result == {"category": {"eq": "electronics"}}

    def test_gt_filter(self) -> None:
        compiler = S3VectorsFilterCompiler()
        expr = FilterExpression(field="price", operator=FilterOperator.GT, value=100)
        result = compiler.compile(expr, self._schema())
        assert result == {"price": {"gt": 100}}

    def test_in_filter(self) -> None:
        compiler = S3VectorsFilterCompiler()
        expr = FilterExpression(field="category", operator=FilterOperator.IN, value=["a", "b"])
        result = compiler.compile(expr, self._schema())
        assert result == {"category": {"in": ["a", "b"]}}

    def test_ne_filter(self) -> None:
        compiler = S3VectorsFilterCompiler()
        expr = FilterExpression(field="price", operator=FilterOperator.NE, value=50)
        result = compiler.compile(expr, self._schema())
        assert result == {"not": {"price": {"eq": 50}}}

    def test_and_filter(self) -> None:
        compiler = S3VectorsFilterCompiler()
        expr = FilterExpression(
            operator=FilterOperator.AND,
            children=[
                FilterExpression(field="price", operator=FilterOperator.GT, value=10),
                FilterExpression(field="price", operator=FilterOperator.LT, value=100),
            ],
        )
        result = compiler.compile(expr, self._schema())
        assert "and" in result
        assert len(result["and"]) == 2

    def test_unsupported_operator(self) -> None:
        compiler = S3VectorsFilterCompiler()
        expr = FilterExpression(field="category", operator=FilterOperator.CONTAINS, value="test")
        with pytest.raises(FilterCompilationError):
            compiler.compile(expr, self._schema())

    def test_validate_unknown_field(self) -> None:
        compiler = S3VectorsFilterCompiler()
        expr = FilterExpression(field="nonexistent", operator=FilterOperator.EQ, value=1)
        errors = compiler.validate(expr, self._schema())
        assert len(errors) == 1
        assert "Unknown field" in errors[0].message

    def test_nin_filter(self) -> None:
        compiler = S3VectorsFilterCompiler()
        expr = FilterExpression(field="category", operator=FilterOperator.NIN, value=["a", "b"])
        result = compiler.compile(expr, self._schema())
        assert result == {"not": {"category": {"in": ["a", "b"]}}}
