"""Tests for ChromaAdapter and ChromaFilterCompiler."""

from __future__ import annotations

import pytest

from openqueryagent.adapters.chroma import ChromaAdapter
from openqueryagent.adapters.chroma_filters import ChromaFilterCompiler
from openqueryagent.core.exceptions import AdapterConnectionError, FilterCompilationError
from openqueryagent.core.types import (
    CollectionSchema,
    FilterExpression,
    FilterOperator,
    PropertySchema,
)


class TestChromaAdapter:
    def test_properties(self) -> None:
        adapter = ChromaAdapter(adapter_id="test-chroma")
        assert adapter.adapter_id == "test-chroma"
        assert adapter.adapter_name == "chroma"
        assert adapter.supports_native_aggregation is False

    def test_not_connected_raises(self) -> None:
        adapter = ChromaAdapter()
        with pytest.raises(AdapterConnectionError, match="Not connected"):
            adapter._ensure_connected()

    def test_filter_compiler(self) -> None:
        adapter = ChromaAdapter()
        compiler = adapter.get_filter_compiler()
        assert compiler is not None

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        adapter = ChromaAdapter()
        adapter._client = "mock"
        await adapter.disconnect()
        assert adapter._client is None

    def test_convert_query_results(self) -> None:
        results = {
            "ids": [["doc-1", "doc-2"]],
            "documents": [["Hello", "World"]],
            "metadatas": [[{"key": "val"}, None]],
            "distances": [[0.1, 0.5]],
        }
        sr = ChromaAdapter._convert_query_results(results, "test")
        assert len(sr.documents) == 2
        assert sr.documents[0].id == "doc-1"
        assert sr.documents[0].content == "Hello"
        assert sr.documents[0].score > sr.documents[1].score

    def test_convert_get_results(self) -> None:
        results = {
            "ids": ["doc-1", "doc-2"],
            "documents": ["Hello", "World"],
            "metadatas": [{"key": "val"}, None],
        }
        sr = ChromaAdapter._convert_get_results(results, "test")
        assert len(sr.documents) == 2
        assert sr.documents[0].content == "Hello"

    def test_infer_type(self) -> None:
        assert ChromaAdapter._infer_type(True).value == "bool"
        assert ChromaAdapter._infer_type(42).value == "int"
        assert ChromaAdapter._infer_type(3.14).value == "float"
        assert ChromaAdapter._infer_type("text").value == "text"


class TestChromaFilterCompiler:
    def _schema(self) -> CollectionSchema:
        return CollectionSchema(
            name="test",
            adapter_id="chroma",
            properties=[
                PropertySchema(name="price", data_type="float"),
                PropertySchema(name="title", data_type="text"),
            ],
        )

    def test_eq_filter(self) -> None:
        compiler = ChromaFilterCompiler()
        expr = FilterExpression(field="price", operator=FilterOperator.EQ, value=100)
        result = compiler.compile(expr, self._schema())
        assert result == {"price": {"$eq": 100}}

    def test_lt_filter(self) -> None:
        compiler = ChromaFilterCompiler()
        expr = FilterExpression(field="price", operator=FilterOperator.LT, value=50)
        result = compiler.compile(expr, self._schema())
        assert result == {"price": {"$lt": 50}}

    def test_in_filter(self) -> None:
        compiler = ChromaFilterCompiler()
        expr = FilterExpression(field="title", operator=FilterOperator.IN, value=["a", "b"])
        result = compiler.compile(expr, self._schema())
        assert result == {"title": {"$in": ["a", "b"]}}

    def test_and_filter(self) -> None:
        compiler = ChromaFilterCompiler()
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

    def test_or_filter(self) -> None:
        compiler = ChromaFilterCompiler()
        expr = FilterExpression(
            operator=FilterOperator.OR,
            children=[
                FilterExpression(field="price", operator=FilterOperator.EQ, value=10),
                FilterExpression(field="price", operator=FilterOperator.EQ, value=20),
            ],
        )
        result = compiler.compile(expr, self._schema())
        assert "$or" in result

    def test_not_filter(self) -> None:
        compiler = ChromaFilterCompiler()
        expr = FilterExpression(
            operator=FilterOperator.NOT,
            children=[
                FilterExpression(field="price", operator=FilterOperator.EQ, value=10),
            ],
        )
        result = compiler.compile(expr, self._schema())
        assert "$not" in result

    def test_contains_filter(self) -> None:
        compiler = ChromaFilterCompiler()
        expr = FilterExpression(field="title", operator=FilterOperator.CONTAINS, value="test")
        result = compiler.compile(expr, self._schema())
        assert result == {"$contains": "test"}

    def test_between_filter(self) -> None:
        compiler = ChromaFilterCompiler()
        expr = FilterExpression(field="price", operator=FilterOperator.BETWEEN, value=[10, 100])
        result = compiler.compile(expr, self._schema())
        assert "$and" in result

    def test_unsupported_operator(self) -> None:
        compiler = ChromaFilterCompiler()
        expr = FilterExpression(field="title", operator=FilterOperator.REGEX, value=".*")
        with pytest.raises(FilterCompilationError):
            compiler.compile(expr, self._schema())

    def test_validate_unknown_field(self) -> None:
        compiler = ChromaFilterCompiler()
        expr = FilterExpression(field="nonexistent", operator=FilterOperator.EQ, value=1)
        errors = compiler.validate(expr, self._schema())
        assert len(errors) == 1
        assert "Unknown field" in errors[0].message
