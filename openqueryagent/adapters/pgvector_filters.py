"""pgvector filter compiler.

Compiles universal FilterExpression objects into parameterized SQL fragments
for use with ``asyncpg``.

Output format: ``(sql_fragment: str, params: list[Any])``

Parameters use positional placeholders ($1, $2, ...) as required by asyncpg.
"""

from __future__ import annotations

from typing import Any

from openqueryagent.adapters.base import FilterCompiler as FilterCompilerProtocol
from openqueryagent.adapters.base import FilterValidationError
from openqueryagent.core.exceptions import FilterCompilationError
from openqueryagent.core.types import CollectionSchema, FilterExpression, FilterOperator


class PgvectorFilterResult:
    """Result of compiling a filter to SQL.

    Attributes:
        sql: The SQL WHERE clause fragment (e.g., ``"price < $1 AND brand = $2"``).
        params: The parameter values in order.
    """

    __slots__ = ("params", "sql")

    def __init__(self, sql: str, params: list[Any]) -> None:
        self.sql = sql
        self.params = params


class PgvectorFilterCompiler:
    """Compile FilterExpression → parameterized SQL fragments.

    All values are parameterized to prevent SQL injection.
    """

    ADAPTER_ID = "pgvector"

    _UNSUPPORTED: frozenset[FilterOperator] = frozenset({
        FilterOperator.GEO_RADIUS,
    })

    def compile(
        self,
        expression: FilterExpression,
        schema: CollectionSchema,
    ) -> PgvectorFilterResult:
        """Compile to parameterized SQL (fragment, params).

        Raises:
            FilterCompilationError: If an unsupported operator is encountered.
        """
        params: list[Any] = []
        sql = self._compile_node(expression, params)
        return PgvectorFilterResult(sql=sql, params=params)

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
                message=f"Operator {expr.operator.value} is not supported by pgvector adapter",
            ))

    def _compile_node(self, expr: FilterExpression, params: list[Any]) -> str:
        """Recursively compile a filter node."""
        if expr.operator == FilterOperator.AND:
            parts = [self._compile_node(c, params) for c in (expr.children or [])]
            return "(" + " AND ".join(parts) + ")"

        if expr.operator == FilterOperator.OR:
            parts = [self._compile_node(c, params) for c in (expr.children or [])]
            return "(" + " OR ".join(parts) + ")"

        if expr.operator == FilterOperator.NOT:
            children = expr.children or []
            if len(children) == 1:
                inner = self._compile_node(children[0], params)
                return f"NOT ({inner})"
            parts = [self._compile_node(c, params) for c in children]
            return "NOT (" + " AND ".join(parts) + ")"

        return self._compile_leaf(expr, params)

    def _compile_leaf(self, expr: FilterExpression, params: list[Any]) -> str:
        """Compile a leaf-level filter condition with parameterized values."""
        field = self._quote_field(expr.field or "")
        value = expr.value

        if expr.operator in self._UNSUPPORTED:
            raise FilterCompilationError(
                f"Operator {expr.operator.value} is not supported by pgvector adapter",
                operator=expr.operator.value,
                field=expr.field or "",
                adapter_id=self.ADAPTER_ID,
            )

        if expr.operator == FilterOperator.EQ:
            params.append(value)
            return f"{field} = ${len(params)}"

        if expr.operator == FilterOperator.NE:
            params.append(value)
            return f"{field} != ${len(params)}"

        if expr.operator == FilterOperator.GT:
            params.append(value)
            return f"{field} > ${len(params)}"

        if expr.operator == FilterOperator.GTE:
            params.append(value)
            return f"{field} >= ${len(params)}"

        if expr.operator == FilterOperator.LT:
            params.append(value)
            return f"{field} < ${len(params)}"

        if expr.operator == FilterOperator.LTE:
            params.append(value)
            return f"{field} <= ${len(params)}"

        if expr.operator == FilterOperator.IN:
            params.append(value)
            return f"{field} = ANY(${len(params)})"

        if expr.operator == FilterOperator.NIN:
            params.append(value)
            return f"{field} != ALL(${len(params)})"

        if expr.operator == FilterOperator.CONTAINS:
            params.append(f"%{value}%")
            return f"{field} ILIKE ${len(params)}"

        if expr.operator == FilterOperator.NOT_CONTAINS:
            params.append(f"%{value}%")
            return f"{field} NOT ILIKE ${len(params)}"

        if expr.operator == FilterOperator.STARTS_WITH:
            params.append(f"{value}%")
            return f"{field} ILIKE ${len(params)}"

        if expr.operator == FilterOperator.ENDS_WITH:
            params.append(f"%{value}")
            return f"{field} ILIKE ${len(params)}"

        if expr.operator == FilterOperator.REGEX:
            params.append(value)
            return f"{field} ~ ${len(params)}"

        if expr.operator == FilterOperator.BETWEEN:
            if not isinstance(value, list) or len(value) != 2:
                raise FilterCompilationError(
                    f"$between requires a two-element list, got {value}",
                    operator="$between",
                    field=expr.field or "",
                    adapter_id=self.ADAPTER_ID,
                )
            low, high = value
            params.append(low)
            params.append(high)
            return f"{field} BETWEEN ${len(params) - 1} AND ${len(params)}"

        if expr.operator == FilterOperator.EXISTS:
            return f"{field} IS NOT NULL"

        raise FilterCompilationError(
            f"Unknown operator {expr.operator.value}",
            operator=expr.operator.value,
            field=expr.field or "",
            adapter_id=self.ADAPTER_ID,
        )

    @staticmethod
    def _quote_field(field: str) -> str:
        """Quote a field name to prevent SQL injection.

        Uses double-quoting for identifiers, which is the SQL standard.
        Strips any existing quotes to prevent double-quoting.
        """
        cleaned = field.replace('"', "")
        return f'"{cleaned}"'


# Protocol conformance assertion
_: type[FilterCompilerProtocol] = PgvectorFilterCompiler
