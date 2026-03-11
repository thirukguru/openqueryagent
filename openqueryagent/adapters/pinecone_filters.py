"""Pinecone filter compiler.

Compiles universal FilterExpression objects into Pinecone metadata filter
dictionary format: ``{"price": {"$lt": 100}}``.
"""

from __future__ import annotations

from typing import Any

from openqueryagent.adapters.base import FilterValidationError
from openqueryagent.core.exceptions import FilterCompilationError
from openqueryagent.core.types import CollectionSchema, FilterExpression, FilterOperator

# Map universal operators to Pinecone operators
_OP_MAP: dict[FilterOperator, str] = {
    FilterOperator.EQ: "$eq",
    FilterOperator.NE: "$ne",
    FilterOperator.GT: "$gt",
    FilterOperator.GTE: "$gte",
    FilterOperator.LT: "$lt",
    FilterOperator.LTE: "$lte",
    FilterOperator.IN: "$in",
    FilterOperator.NIN: "$nin",
    FilterOperator.EXISTS: "$exists",
}

# Operators not natively supported by Pinecone
_UNSUPPORTED: frozenset[FilterOperator] = frozenset({
    FilterOperator.CONTAINS,
    FilterOperator.NOT_CONTAINS,
    FilterOperator.STARTS_WITH,
    FilterOperator.ENDS_WITH,
    FilterOperator.REGEX,
    FilterOperator.GEO_RADIUS,
})


class PineconeFilterCompiler:
    """Compile FilterExpression → Pinecone metadata filter dicts."""

    ADAPTER_ID = "pinecone"

    def compile(
        self,
        expression: FilterExpression,
        schema: CollectionSchema,
    ) -> Any:
        """Compile a FilterExpression into Pinecone filter dict."""
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
            return {"$and": children}

        if expr.operator == FilterOperator.OR:
            children = [self._compile_node(c) for c in (expr.children or [])]
            return {"$or": children}

        if expr.operator == FilterOperator.NOT:
            children = [self._compile_node(c) for c in (expr.children or [])]
            if children:
                return children[0]
            return {}

        if expr.operator in _UNSUPPORTED:
            msg = f"Operator {expr.operator} is not supported by Pinecone"
            raise FilterCompilationError(msg, adapter_id=self.ADAPTER_ID)

        if expr.operator == FilterOperator.BETWEEN:
            if not isinstance(expr.value, list) or len(expr.value) != 2:
                raise FilterCompilationError(
                    f"$between requires a two-element list, got {expr.value}",
                    adapter_id=self.ADAPTER_ID,
                )
            field = expr.field or ""
            return {
                "$and": [
                    {field: {"$gte": expr.value[0]}},
                    {field: {"$lte": expr.value[1]}},
                ],
            }

        pinecone_op = _OP_MAP.get(expr.operator, "$eq")
        field = expr.field or ""
        return {field: {pinecone_op: expr.value}}

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

        if expr.operator in _UNSUPPORTED:
            errors.append(FilterValidationError(
                field=expr.field or "",
                operator=str(expr.operator),
                message=f"Operator {expr.operator} not supported by Pinecone",
            ))
