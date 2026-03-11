"""Weaviate filter compiler.

Compiles universal FilterExpression objects into Weaviate v4 filter
dictionary format compatible with ``weaviate.classes.query.Filter``.
"""

from __future__ import annotations

from typing import Any

from openqueryagent.adapters.base import FilterValidationError
from openqueryagent.core.exceptions import FilterCompilationError
from openqueryagent.core.types import CollectionSchema, FilterExpression, FilterOperator

# Operators not natively supported by Weaviate
_UNSUPPORTED: frozenset[FilterOperator] = frozenset({
    FilterOperator.REGEX,
    FilterOperator.NOT_CONTAINS,
    FilterOperator.STARTS_WITH,
    FilterOperator.ENDS_WITH,
    FilterOperator.GEO_RADIUS,
})


class WeaviateFilterCompiler:
    """Compile FilterExpression → Weaviate filter dicts."""

    ADAPTER_ID = "weaviate"

    def compile(
        self,
        expression: FilterExpression,
        schema: CollectionSchema,
    ) -> Any:
        """Compile a FilterExpression into Weaviate filter dict."""
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
            return {"operator": "And", "operands": children}

        if expr.operator == FilterOperator.OR:
            children = [self._compile_node(c) for c in (expr.children or [])]
            return {"operator": "Or", "operands": children}

        if expr.operator == FilterOperator.NOT:
            children = [self._compile_node(c) for c in (expr.children or [])]
            return {"operator": "Not", "operands": children[:1]}

        if expr.operator in _UNSUPPORTED:
            msg = f"Operator {expr.operator} is not supported by Weaviate"
            raise FilterCompilationError(msg, adapter_id=self.ADAPTER_ID)

        if expr.operator == FilterOperator.IN:
            return {
                "path": [expr.field],
                "operator": "ContainsAny",
                "valueTextArray": expr.value,
            }

        if expr.operator == FilterOperator.NIN:
            return {
                "operator": "Not",
                "operands": [{
                    "path": [expr.field],
                    "operator": "ContainsAny",
                    "valueTextArray": expr.value,
                }],
            }

        if expr.operator == FilterOperator.BETWEEN:
            if not isinstance(expr.value, list) or len(expr.value) != 2:
                raise FilterCompilationError(
                    f"$between requires a two-element list, got {expr.value}",
                    adapter_id=self.ADAPTER_ID,
                )
            return {
                "operator": "And",
                "operands": [
                    {"path": [expr.field], "operator": "GreaterOrEqual", "valueNumber": expr.value[0]},
                    {"path": [expr.field], "operator": "LessOrEqual", "valueNumber": expr.value[1]},
                ],
            }

        if expr.operator == FilterOperator.EXISTS:
            return {
                "path": [expr.field],
                "operator": "IsNull",
                "valueBoolean": False,
            }

        # Map core operators
        op_map: dict[FilterOperator, str] = {
            FilterOperator.EQ: "Equal",
            FilterOperator.NE: "NotEqual",
            FilterOperator.GT: "GreaterThan",
            FilterOperator.GTE: "GreaterThanEqual",
            FilterOperator.LT: "LessThan",
            FilterOperator.LTE: "LessThanEqual",
            FilterOperator.CONTAINS: "Like",
        }
        weaviate_op = op_map.get(expr.operator, "Equal")
        value_key = self._value_key(expr.value)
        return {
            "path": [expr.field],
            "operator": weaviate_op,
            value_key: expr.value,
        }

    @staticmethod
    def _value_key(value: Any) -> str:
        """Determine the Weaviate value key based on type."""
        if isinstance(value, bool):
            return "valueBoolean"
        if isinstance(value, int):
            return "valueInt"
        if isinstance(value, float):
            return "valueNumber"
        if isinstance(value, list):
            return "valueTextArray"
        return "valueText"

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
                message=f"Operator {expr.operator} not supported by Weaviate",
            ))
