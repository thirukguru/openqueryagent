"""Elasticsearch filter compiler.

Compiles universal FilterExpression objects into Elasticsearch bool query
DSL format with ``must``, ``should``, ``must_not``, and ``filter`` clauses.
"""

from __future__ import annotations

from typing import Any

from openqueryagent.adapters.base import FilterValidationError
from openqueryagent.core.exceptions import FilterCompilationError
from openqueryagent.core.types import CollectionSchema, FilterExpression, FilterOperator


class ElasticsearchFilterCompiler:
    """Compile FilterExpression → Elasticsearch bool query dicts."""

    ADAPTER_ID = "elasticsearch"

    def compile(
        self,
        expression: FilterExpression,
        schema: CollectionSchema,
    ) -> Any:
        """Compile a FilterExpression into Elasticsearch query dict."""
        return self._compile_node(expression)

    def validate(
        self,
        expression: FilterExpression,
        schema: CollectionSchema,
    ) -> list[FilterValidationError]:
        """Validate filter expression against schema."""
        errors: list[FilterValidationError] = []
        self._validate_node(expression, schema, errors)
        return errors

    def _compile_node(self, expr: FilterExpression) -> dict[str, Any]:
        """Recursively compile a filter node."""
        if expr.operator == FilterOperator.AND:
            children = [self._compile_node(c) for c in (expr.children or [])]
            return {"bool": {"must": children}}

        if expr.operator == FilterOperator.OR:
            children = [self._compile_node(c) for c in (expr.children or [])]
            return {"bool": {"should": children, "minimum_should_match": 1}}

        if expr.operator == FilterOperator.NOT:
            children = [self._compile_node(c) for c in (expr.children or [])]
            return {"bool": {"must_not": children}}

        return self._compile_leaf(expr)

    def _compile_leaf(self, expr: FilterExpression) -> dict[str, Any]:
        """Compile a leaf-level filter condition."""
        field = expr.field or ""
        value = expr.value

        if expr.operator == FilterOperator.EQ:
            return {"term": {field: value}}

        if expr.operator == FilterOperator.NE:
            return {"bool": {"must_not": [{"term": {field: value}}]}}

        if expr.operator == FilterOperator.GT:
            return {"range": {field: {"gt": value}}}

        if expr.operator == FilterOperator.GTE:
            return {"range": {field: {"gte": value}}}

        if expr.operator == FilterOperator.LT:
            return {"range": {field: {"lt": value}}}

        if expr.operator == FilterOperator.LTE:
            return {"range": {field: {"lte": value}}}

        if expr.operator == FilterOperator.IN:
            return {"terms": {field: value}}

        if expr.operator == FilterOperator.NIN:
            return {"bool": {"must_not": [{"terms": {field: value}}]}}

        if expr.operator == FilterOperator.CONTAINS:
            return {"match": {field: value}}

        if expr.operator == FilterOperator.NOT_CONTAINS:
            return {"bool": {"must_not": [{"match": {field: value}}]}}

        if expr.operator == FilterOperator.STARTS_WITH:
            return {"prefix": {field: value}}

        if expr.operator == FilterOperator.REGEX:
            return {"regexp": {field: value}}

        if expr.operator == FilterOperator.EXISTS:
            return {"exists": {"field": field}}

        if expr.operator == FilterOperator.BETWEEN:
            if not isinstance(value, list) or len(value) != 2:
                raise FilterCompilationError(
                    f"$between requires a two-element list, got {value}",
                    adapter_id=self.ADAPTER_ID,
                )
            return {"range": {field: {"gte": value[0], "lte": value[1]}}}

        if expr.operator == FilterOperator.GEO_RADIUS:
            if not isinstance(value, dict):
                raise FilterCompilationError(
                    "$geo_radius requires a dict with lat, lon, radius",
                    adapter_id=self.ADAPTER_ID,
                )
            return {
                "geo_distance": {
                    "distance": value.get("radius", "10km"),
                    field: {"lat": value.get("lat", 0), "lon": value.get("lon", 0)},
                },
            }

        if expr.operator == FilterOperator.ENDS_WITH:
            return {"wildcard": {field: f"*{value}"}}

        msg = f"Unsupported operator: {expr.operator}"
        raise FilterCompilationError(msg, adapter_id=self.ADAPTER_ID)

    def _validate_node(
        self,
        expr: FilterExpression,
        schema: CollectionSchema,
        errors: list[FilterValidationError],
    ) -> None:
        """Validate a filter node recursively."""
        if expr.operator in (FilterOperator.AND, FilterOperator.OR, FilterOperator.NOT):
            for child in expr.children or []:
                self._validate_node(child, schema, errors)
            return

        if expr.field:
            prop_names = {p.name for p in schema.properties}
            if expr.field not in prop_names:
                errors.append(FilterValidationError(
                    field=expr.field,
                    operator=str(expr.operator),
                    message=f"Unknown field '{expr.field}'",
                ))
