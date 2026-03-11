"""Tests for ElasticsearchAdapter and ElasticsearchFilterCompiler."""

from __future__ import annotations

import pytest

from openqueryagent.adapters.elasticsearch import ElasticsearchAdapter
from openqueryagent.adapters.elasticsearch_filters import ElasticsearchFilterCompiler
from openqueryagent.core.exceptions import AdapterConnectionError
from openqueryagent.core.types import (
    CollectionSchema,
    FilterExpression,
    FilterOperator,
    PropertySchema,
)


class TestElasticsearchAdapter:
    def test_properties(self) -> None:
        adapter = ElasticsearchAdapter(adapter_id="test-es")
        assert adapter.adapter_id == "test-es"
        assert adapter.adapter_name == "elasticsearch"
        assert adapter.supports_native_aggregation is True

    def test_not_connected_raises(self) -> None:
        adapter = ElasticsearchAdapter()
        with pytest.raises(AdapterConnectionError, match="Not connected"):
            adapter._ensure_connected()

    def test_filter_compiler(self) -> None:
        adapter = ElasticsearchAdapter()
        compiler = adapter.get_filter_compiler()
        assert compiler is not None

    def test_vector_field_default(self) -> None:
        assert ElasticsearchAdapter._vector_field(None) == "embedding"

    def test_vector_field_custom(self) -> None:
        assert ElasticsearchAdapter._vector_field({"vector_field": "vec"}) == "vec"


class TestElasticsearchFilterCompiler:
    def _schema(self) -> CollectionSchema:
        return CollectionSchema(
            name="test",
            adapter_id="elasticsearch",
            properties=[
                PropertySchema(name="price", data_type="float"),
                PropertySchema(name="title", data_type="text"),
                PropertySchema(name="category", data_type="text"),
            ],
        )

    def test_eq_filter(self) -> None:
        compiler = ElasticsearchFilterCompiler()
        expr = FilterExpression(field="category", operator=FilterOperator.EQ, value="electronics")
        result = compiler.compile(expr, self._schema())
        assert result == {"term": {"category": "electronics"}}

    def test_ne_filter(self) -> None:
        compiler = ElasticsearchFilterCompiler()
        expr = FilterExpression(field="category", operator=FilterOperator.NE, value="food")
        result = compiler.compile(expr, self._schema())
        assert result == {"bool": {"must_not": [{"term": {"category": "food"}}]}}

    def test_range_gt(self) -> None:
        compiler = ElasticsearchFilterCompiler()
        expr = FilterExpression(field="price", operator=FilterOperator.GT, value=100)
        result = compiler.compile(expr, self._schema())
        assert result == {"range": {"price": {"gt": 100}}}

    def test_in_filter(self) -> None:
        compiler = ElasticsearchFilterCompiler()
        expr = FilterExpression(field="category", operator=FilterOperator.IN, value=["a", "b"])
        result = compiler.compile(expr, self._schema())
        assert result == {"terms": {"category": ["a", "b"]}}

    def test_contains_filter(self) -> None:
        compiler = ElasticsearchFilterCompiler()
        expr = FilterExpression(field="title", operator=FilterOperator.CONTAINS, value="test")
        result = compiler.compile(expr, self._schema())
        assert result == {"match": {"title": "test"}}

    def test_exists_filter(self) -> None:
        compiler = ElasticsearchFilterCompiler()
        expr = FilterExpression(field="title", operator=FilterOperator.EXISTS, value=True)
        result = compiler.compile(expr, self._schema())
        assert result == {"exists": {"field": "title"}}

    def test_between_filter(self) -> None:
        compiler = ElasticsearchFilterCompiler()
        expr = FilterExpression(field="price", operator=FilterOperator.BETWEEN, value=[10, 100])
        result = compiler.compile(expr, self._schema())
        assert result == {"range": {"price": {"gte": 10, "lte": 100}}}

    def test_and_filter(self) -> None:
        compiler = ElasticsearchFilterCompiler()
        expr = FilterExpression(
            operator=FilterOperator.AND,
            children=[
                FilterExpression(field="price", operator=FilterOperator.GT, value=10),
                FilterExpression(field="price", operator=FilterOperator.LT, value=100),
            ],
        )
        result = compiler.compile(expr, self._schema())
        assert "bool" in result
        assert "must" in result["bool"]

    def test_or_filter(self) -> None:
        compiler = ElasticsearchFilterCompiler()
        expr = FilterExpression(
            operator=FilterOperator.OR,
            children=[
                FilterExpression(field="price", operator=FilterOperator.EQ, value=10),
                FilterExpression(field="price", operator=FilterOperator.EQ, value=20),
            ],
        )
        result = compiler.compile(expr, self._schema())
        assert result["bool"]["should"] is not None

    def test_prefix_filter(self) -> None:
        compiler = ElasticsearchFilterCompiler()
        expr = FilterExpression(field="title", operator=FilterOperator.STARTS_WITH, value="hello")
        result = compiler.compile(expr, self._schema())
        assert result == {"prefix": {"title": "hello"}}

    def test_regex_filter(self) -> None:
        compiler = ElasticsearchFilterCompiler()
        expr = FilterExpression(field="title", operator=FilterOperator.REGEX, value="hel.*")
        result = compiler.compile(expr, self._schema())
        assert result == {"regexp": {"title": "hel.*"}}

    def test_validate_unknown_field(self) -> None:
        compiler = ElasticsearchFilterCompiler()
        expr = FilterExpression(field="nonexistent", operator=FilterOperator.EQ, value=1)
        errors = compiler.validate(expr, self._schema())
        assert len(errors) == 1
        assert "Unknown field" in errors[0].message
