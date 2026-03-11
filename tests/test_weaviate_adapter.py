"""Tests for WeaviateAdapter and WeaviateFilterCompiler."""

from __future__ import annotations

import pytest

from openqueryagent.adapters.weaviate import WeaviateAdapter
from openqueryagent.adapters.weaviate_filters import WeaviateFilterCompiler
from openqueryagent.core.exceptions import AdapterConnectionError, FilterCompilationError
from openqueryagent.core.types import (
    CollectionSchema,
    FilterExpression,
    FilterOperator,
    PropertySchema,
)


class TestWeaviateAdapter:
    def test_properties(self) -> None:
        adapter = WeaviateAdapter(adapter_id="test-weaviate")
        assert adapter.adapter_id == "test-weaviate"
        assert adapter.adapter_name == "weaviate"
        assert adapter.supports_native_aggregation is True

    def test_not_connected_raises(self) -> None:
        adapter = WeaviateAdapter()
        with pytest.raises(AdapterConnectionError, match="Not connected"):
            adapter._ensure_connected()

    def test_filter_compiler(self) -> None:
        adapter = WeaviateAdapter()
        compiler = adapter.get_filter_compiler()
        assert compiler is not None


class TestWeaviateFilterCompiler:
    def _schema(self) -> CollectionSchema:
        return CollectionSchema(
            name="test",
            adapter_id="weaviate",
            properties=[
                PropertySchema(name="price", data_type="float"),
                PropertySchema(name="title", data_type="text"),
            ],
        )

    def test_eq_filter(self) -> None:
        compiler = WeaviateFilterCompiler()
        expr = FilterExpression(field="price", operator=FilterOperator.EQ, value=100)
        result = compiler.compile(expr, self._schema())
        assert result["path"] == ["price"]
        assert "Equal" in result["operator"]

    def test_and_filter(self) -> None:
        compiler = WeaviateFilterCompiler()
        expr = FilterExpression(
            operator=FilterOperator.AND,
            children=[
                FilterExpression(field="price", operator=FilterOperator.GT, value=10),
                FilterExpression(field="price", operator=FilterOperator.LT, value=100),
            ],
        )
        result = compiler.compile(expr, self._schema())
        assert result["operator"] == "And"
        assert len(result["operands"]) == 2

    def test_in_filter(self) -> None:
        compiler = WeaviateFilterCompiler()
        expr = FilterExpression(field="title", operator=FilterOperator.IN, value=["a", "b"])
        result = compiler.compile(expr, self._schema())
        assert result["operator"] == "ContainsAny"

    def test_unsupported_operator(self) -> None:
        compiler = WeaviateFilterCompiler()
        expr = FilterExpression(field="title", operator=FilterOperator.REGEX, value=".*")
        with pytest.raises(FilterCompilationError):
            compiler.compile(expr, self._schema())

    def test_validate_unknown_field(self) -> None:
        compiler = WeaviateFilterCompiler()
        expr = FilterExpression(field="nonexistent", operator=FilterOperator.EQ, value=1)
        errors = compiler.validate(expr, self._schema())
        assert len(errors) == 1
        assert "Unknown field" in errors[0].message

    def test_validate_unsupported_op(self) -> None:
        compiler = WeaviateFilterCompiler()
        expr = FilterExpression(field="title", operator=FilterOperator.REGEX, value=".*")
        errors = compiler.validate(expr, self._schema())
        assert len(errors) >= 1

    def test_exists_filter(self) -> None:
        compiler = WeaviateFilterCompiler()
        expr = FilterExpression(field="title", operator=FilterOperator.EXISTS, value=True)
        result = compiler.compile(expr, self._schema())
        assert result["operator"] == "IsNull"

    def test_or_filter(self) -> None:
        compiler = WeaviateFilterCompiler()
        expr = FilterExpression(
            operator=FilterOperator.OR,
            children=[
                FilterExpression(field="price", operator=FilterOperator.EQ, value=10),
                FilterExpression(field="price", operator=FilterOperator.EQ, value=20),
            ],
        )
        result = compiler.compile(expr, self._schema())
        assert result["operator"] == "Or"
