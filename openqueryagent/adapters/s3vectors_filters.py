"""AWS S3 Vectors filter compiler.

Compiles universal FilterExpression objects into S3 Vectors metadata filter
format. S3 Vectors uses a simple key-value filter structure.
"""

from __future__ import annotations

from typing import Any

from openqueryagent.adapters.base import FilterValidationError
from openqueryagent.core.exceptions import FilterCompilationError
from openqueryagent.core.types import CollectionSchema, FilterExpression, FilterOperator

# Operators not natively supported by S3 Vectors
_UNSUPPORTED: frozenset[FilterOperator] = frozenset({
    FilterOperator.CONTAINS,
    FilterOperator.NOT_CONTAINS,
    FilterOperator.STARTS_WITH,
    FilterOperator.ENDS_WITH,
    FilterOperator.REGEX,
    FilterOperator.GEO_RADIUS,
    FilterOperator.EXISTS,
    FilterOperator.BETWEEN,
})


class S3VectorsFilterCompiler:
    """Compile FilterExpression → S3 Vectors metadata filter dicts."""

    ADAPTER_ID = "s3vectors"

    def compile(
        self,
        expression: FilterExpression,
        schema: CollectionSchema,
    ) -> Any:
        """Compile a FilterExpression into S3 Vectors filter dict."""
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
            return {"and": children}

        if expr.operator == FilterOperator.OR:
            children = [self._compile_node(c) for c in (expr.children or [])]
            return {"or": children}

        if expr.operator == FilterOperator.NOT:
            children = [self._compile_node(c) for c in (expr.children or [])]
            if children:
                return {"not": children[0]}
            return {}

        if expr.operator in _UNSUPPORTED:
            msg = f"Operator {expr.operator} is not supported by S3 Vectors"
            raise FilterCompilationError(msg, adapter_id=self.ADAPTER_ID)

        # Map to S3 Vectors metadata filter format
        field = expr.field or ""

        if expr.operator == FilterOperator.EQ:
            return {field: {"eq": expr.value}}

        if expr.operator == FilterOperator.NE:
            return {"not": {field: {"eq": expr.value}}}

        if expr.operator == FilterOperator.GT:
            return {field: {"gt": expr.value}}

        if expr.operator == FilterOperator.GTE:
            return {field: {"gte": expr.value}}

        if expr.operator == FilterOperator.LT:
            return {field: {"lt": expr.value}}

        if expr.operator == FilterOperator.LTE:
            return {field: {"lte": expr.value}}

        if expr.operator == FilterOperator.IN:
            return {field: {"in": expr.value}}

        if expr.operator == FilterOperator.NIN:
            return {"not": {field: {"in": expr.value}}}

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

        if expr.operator in _UNSUPPORTED:
            errors.append(FilterValidationError(
                field=expr.field or "",
                operator=str(expr.operator),
                message=f"Operator {expr.operator} not supported by S3 Vectors",
            ))
