"""Qdrant filter compiler.

Compiles universal FilterExpression objects into ``qdrant_client.models`` filter objects.

Since qdrant-client is an optional dependency, this module uses dictionary
representations that match the Qdrant filter JSON format. The actual
qdrant_client.models objects can be created from these dicts when the
client is available.
"""

from __future__ import annotations

from typing import Any

from openqueryagent.adapters.base import FilterCompiler as FilterCompilerProtocol
from openqueryagent.adapters.base import FilterValidationError
from openqueryagent.core.exceptions import FilterCompilationError
from openqueryagent.core.types import CollectionSchema, FilterExpression, FilterOperator


class QdrantFilterCompiler:
    """Compile FilterExpression → Qdrant filter dicts.

    Output format mirrors ``qdrant_client.models.Filter`` as nested dicts,
    which qdrant-client accepts interchangeably with model objects.
    """

    ADAPTER_ID = "qdrant"

    # Operators not natively supported by Qdrant
    _UNSUPPORTED: frozenset[FilterOperator] = frozenset({
        FilterOperator.STARTS_WITH,
        FilterOperator.ENDS_WITH,
        FilterOperator.REGEX,
        FilterOperator.NOT_CONTAINS,
        FilterOperator.BETWEEN,  # decomposed to GTE + LTE
    })

    def compile(
        self,
        expression: FilterExpression,
        schema: CollectionSchema,
    ) -> dict[str, Any]:
        """Compile a FilterExpression into Qdrant filter dict.

        Raises:
            FilterCompilationError: If an unsupported operator is encountered.
        """
        return self._compile_node(expression)

    def validate(
        self,
        expression: FilterExpression,
        schema: CollectionSchema,
    ) -> list[FilterValidationError]:
        errors: list[FilterValidationError] = []
        self._validate_node(expression, errors)
        return errors

    def _validate_node(
        self,
        expr: FilterExpression,
        errors: list[FilterValidationError],
    ) -> None:
        if expr.operator in (FilterOperator.AND, FilterOperator.OR, FilterOperator.NOT):
            if expr.children:
                for child in expr.children:
                    self._validate_node(child, errors)
            return

        if expr.operator in (
            FilterOperator.STARTS_WITH,
            FilterOperator.ENDS_WITH,
            FilterOperator.REGEX,
            FilterOperator.NOT_CONTAINS,
        ):
            errors.append(FilterValidationError(
                field=expr.field or "",
                operator=expr.operator.value,
                message=f"Operator {expr.operator.value} is not supported by Qdrant",
            ))

    def _compile_node(self, expr: FilterExpression) -> dict[str, Any]:
        """Recursively compile a filter node."""
        if expr.operator == FilterOperator.AND:
            must = [self._compile_node(c) for c in (expr.children or [])]
            return {"must": must}

        if expr.operator == FilterOperator.OR:
            should = [self._compile_node(c) for c in (expr.children or [])]
            return {"should": should}

        if expr.operator == FilterOperator.NOT:
            children = expr.children or []
            must_not = [self._compile_node(c) for c in children]
            return {"must_not": must_not}

        # $between → decompose to GTE + LTE
        if expr.operator == FilterOperator.BETWEEN:
            if not isinstance(expr.value, list) or len(expr.value) != 2:
                raise FilterCompilationError(
                    f"$between requires a two-element list, got {expr.value}",
                    operator="$between",
                    field=expr.field or "",
                    adapter_id=self.ADAPTER_ID,
                )
            low, high = expr.value
            return {
                "must": [
                    {"key": expr.field, "range": {"gte": low}},
                    {"key": expr.field, "range": {"lte": high}},
                ],
            }

        return self._compile_leaf(expr)

    def _compile_leaf(self, expr: FilterExpression) -> dict[str, Any]:
        """Compile a leaf-level filter condition."""
        field = expr.field or ""
        value = expr.value

        if expr.operator in (
            FilterOperator.STARTS_WITH,
            FilterOperator.ENDS_WITH,
            FilterOperator.REGEX,
            FilterOperator.NOT_CONTAINS,
        ):
            raise FilterCompilationError(
                f"Operator {expr.operator.value} is not supported by Qdrant",
                operator=expr.operator.value,
                field=field,
                adapter_id=self.ADAPTER_ID,
            )

        if expr.operator == FilterOperator.EQ:
            return {"key": field, "match": {"value": value}}

        if expr.operator == FilterOperator.NE:
            return {"must_not": [{"key": field, "match": {"value": value}}]}

        if expr.operator == FilterOperator.GT:
            return {"key": field, "range": {"gt": value}}

        if expr.operator == FilterOperator.GTE:
            return {"key": field, "range": {"gte": value}}

        if expr.operator == FilterOperator.LT:
            return {"key": field, "range": {"lt": value}}

        if expr.operator == FilterOperator.LTE:
            return {"key": field, "range": {"lte": value}}

        if expr.operator == FilterOperator.IN:
            return {"key": field, "match": {"any": value}}

        if expr.operator == FilterOperator.NIN:
            return {"must_not": [{"key": field, "match": {"any": value}}]}

        if expr.operator == FilterOperator.CONTAINS:
            return {"key": field, "match": {"text": value}}

        if expr.operator == FilterOperator.EXISTS:
            return {"is_empty": {"key": field, "is_empty": False}}

        if expr.operator == FilterOperator.GEO_RADIUS and isinstance(value, dict):
            return {
                "key": field,
                "geo_radius": {
                    "center": {"lat": value["lat"], "lon": value["lon"]},
                    "radius": value["radius_km"] * 1000,  # km → meters
                },
            }

        raise FilterCompilationError(
            f"Unknown operator {expr.operator.value}",
            operator=expr.operator.value,
            field=field,
            adapter_id=self.ADAPTER_ID,
        )


# Protocol conformance assertion
_: type[FilterCompilerProtocol] = QdrantFilterCompiler
