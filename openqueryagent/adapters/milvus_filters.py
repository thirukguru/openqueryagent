"""Milvus filter compiler.

Compiles universal FilterExpression objects into Milvus boolean expression
strings as used by ``pymilvus``.

Milvus uses a SQL-like expression syntax::

    price < 100 and brand in ["Nike", "Adidas"]
"""

from __future__ import annotations

from typing import Any

from openqueryagent.adapters.base import FilterCompiler as FilterCompilerProtocol
from openqueryagent.adapters.base import FilterValidationError
from openqueryagent.core.exceptions import FilterCompilationError
from openqueryagent.core.types import CollectionSchema, FilterExpression, FilterOperator


class MilvusFilterCompiler:
    """Compile FilterExpression → Milvus expression strings."""

    ADAPTER_ID = "milvus"

    _UNSUPPORTED: frozenset[FilterOperator] = frozenset({
        FilterOperator.REGEX,
        FilterOperator.NOT_CONTAINS,
        FilterOperator.ENDS_WITH,
        FilterOperator.GEO_RADIUS,
    })

    def compile(
        self,
        expression: FilterExpression,
        schema: CollectionSchema,
    ) -> str:
        """Compile to Milvus expression string.

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

        if expr.operator in self._UNSUPPORTED:
            errors.append(FilterValidationError(
                field=expr.field or "",
                operator=expr.operator.value,
                message=f"Operator {expr.operator.value} is not supported by Milvus",
            ))

    def _compile_node(self, expr: FilterExpression) -> str:
        """Recursively compile a filter node to an expression string."""
        if expr.operator == FilterOperator.AND:
            parts = [self._compile_node(c) for c in (expr.children or [])]
            return "(" + " and ".join(parts) + ")"

        if expr.operator == FilterOperator.OR:
            parts = [self._compile_node(c) for c in (expr.children or [])]
            return "(" + " or ".join(parts) + ")"

        if expr.operator == FilterOperator.NOT:
            children = expr.children or []
            if len(children) == 1:
                inner = self._compile_node(children[0])
                return f"not ({inner})"
            # Multiple children under NOT → NOT (A AND B AND ...)
            parts = [self._compile_node(c) for c in children]
            return "not (" + " and ".join(parts) + ")"

        return self._compile_leaf(expr)

    def _compile_leaf(self, expr: FilterExpression) -> str:
        """Compile a leaf-level filter condition."""
        field = expr.field or ""
        value = expr.value

        if expr.operator in self._UNSUPPORTED:
            raise FilterCompilationError(
                f"Operator {expr.operator.value} is not supported by Milvus",
                operator=expr.operator.value,
                field=field,
                adapter_id=self.ADAPTER_ID,
            )

        if expr.operator == FilterOperator.EQ:
            return f"{field} == {self._format_value(value)}"

        if expr.operator == FilterOperator.NE:
            return f"{field} != {self._format_value(value)}"

        if expr.operator == FilterOperator.GT:
            return f"{field} > {self._format_value(value)}"

        if expr.operator == FilterOperator.GTE:
            return f"{field} >= {self._format_value(value)}"

        if expr.operator == FilterOperator.LT:
            return f"{field} < {self._format_value(value)}"

        if expr.operator == FilterOperator.LTE:
            return f"{field} <= {self._format_value(value)}"

        if expr.operator == FilterOperator.IN:
            formatted = self._format_list(value)
            return f"{field} in {formatted}"

        if expr.operator == FilterOperator.NIN:
            formatted = self._format_list(value)
            return f"{field} not in {formatted}"

        if expr.operator == FilterOperator.CONTAINS:
            # Milvus uses array_contains for array fields; for text we use LIKE
            escaped = str(value).replace('"', '\\"')
            return f'{field} like "%{escaped}%"'

        if expr.operator == FilterOperator.STARTS_WITH:
            escaped = str(value).replace('"', '\\"')
            return f'{field} like "{escaped}%"'

        if expr.operator == FilterOperator.BETWEEN:
            if not isinstance(value, list) or len(value) != 2:
                raise FilterCompilationError(
                    f"$between requires a two-element list, got {value}",
                    operator="$between",
                    field=field,
                    adapter_id=self.ADAPTER_ID,
                )
            low, high = value
            return f"({field} >= {self._format_value(low)} and {field} <= {self._format_value(high)})"

        if expr.operator == FilterOperator.EXISTS:
            return f"{field} != null"

        raise FilterCompilationError(
            f"Unknown operator {expr.operator.value}",
            operator=expr.operator.value,
            field=field,
            adapter_id=self.ADAPTER_ID,
        )

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a single value for Milvus expression syntax."""
        if isinstance(value, str):
            # Escape internal double quotes
            escaped = value.replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _format_list(values: Any) -> str:
        """Format a list of values for Milvus IN expression."""
        if not isinstance(values, list):
            return str(values)
        formatted = []
        for v in values:
            if isinstance(v, str):
                escaped = v.replace('"', '\\"')
                formatted.append(f'"{escaped}"')
            else:
                formatted.append(str(v))
        return "[" + ", ".join(formatted) + "]"


# Protocol conformance assertion
_: type[FilterCompilerProtocol] = MilvusFilterCompiler
