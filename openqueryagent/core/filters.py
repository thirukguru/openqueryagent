"""Universal Filter DSL for OpenQueryAgent.

Provides a fluent API for constructing database-agnostic filter expressions
that compile to each backend's native filter format.

Usage::

    from openqueryagent.core.filters import F

    # Simple comparison
    f = F.price < 100

    # Compound
    f = (F.price < 100) & (F.category == "electronics")

    # Set membership
    f = F.brand.in_(["Nike", "Adidas", "Puma"])

    # Range
    f = F.price.between(50, 200)

    # Text contains
    f = F.description.contains("wireless")

    # Geo radius
    f = F.location.geo_radius(37.7749, -122.4194, 10)

    # Boolean negation
    f = ~(F.price > 100)

    # Nested
    f = (F.price < 100) | ((F.rating >= 4.5) & (F.reviews > 50))
"""

from __future__ import annotations

from typing import Any

from openqueryagent.core.types import (
    CollectionSchema,
    DataType,
    FilterExpression,
    FilterOperator,
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class FilterValidationResult:
    """Result of validating a filter against a schema."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, message: str) -> None:
        self.errors.append(message)


def validate_filter(
    expression: FilterExpression,
    schema: CollectionSchema,
) -> FilterValidationResult:
    """Validate a filter expression against a collection schema.

    Checks for:
    - Unknown fields (field not in schema)
    - Non-filterable fields
    - Type mismatches (comparing string field with int, etc.)

    Args:
        expression: The filter expression to validate.
        schema: The collection schema to validate against.

    Returns:
        FilterValidationResult with any errors found.
    """
    result = FilterValidationResult()
    _validate_recursive(expression, schema, result)
    return result


_NUMERIC_TYPES = {DataType.INT, DataType.FLOAT}
_NUMERIC_OPERATORS = {
    FilterOperator.GT,
    FilterOperator.GTE,
    FilterOperator.LT,
    FilterOperator.LTE,
    FilterOperator.BETWEEN,
}


def _validate_recursive(
    expr: FilterExpression,
    schema: CollectionSchema,
    result: FilterValidationResult,
) -> None:
    """Recursively validate a filter expression tree."""
    # Boolean combinators — validate children
    if expr.operator in (FilterOperator.AND, FilterOperator.OR, FilterOperator.NOT):
        if expr.children:
            for child in expr.children:
                _validate_recursive(child, schema, result)
        return

    # Field-level operators
    if expr.field is None:
        result.add_error(f"Operator {expr.operator.value} requires a field name")
        return

    # Build property lookup
    prop_map = {p.name: p for p in schema.properties}

    if expr.field not in prop_map:
        result.add_error(f"Unknown field '{expr.field}' (not in schema for '{schema.name}')")
        return

    prop = prop_map[expr.field]

    if not prop.filterable:
        result.add_error(f"Field '{expr.field}' is not filterable")
        return

    # Type-specific validation
    if expr.operator in _NUMERIC_OPERATORS and prop.data_type not in _NUMERIC_TYPES:
        result.add_error(
            f"Operator {expr.operator.value} requires a numeric field, "
            f"but '{expr.field}' is {prop.data_type.value}"
        )

    if expr.operator == FilterOperator.GEO_RADIUS and prop.data_type != DataType.GEO:
        result.add_error(
            f"Operator $geo_radius requires a geo field, "
            f"but '{expr.field}' is {prop.data_type.value}"
        )

    if expr.operator in (FilterOperator.CONTAINS, FilterOperator.NOT_CONTAINS,
                         FilterOperator.STARTS_WITH, FilterOperator.ENDS_WITH,
                         FilterOperator.REGEX) and prop.data_type != DataType.TEXT:
        result.add_error(
            f"Operator {expr.operator.value} requires a text field, "
            f"but '{expr.field}' is {prop.data_type.value}"
        )


# ---------------------------------------------------------------------------
# F Proxy — Fluent Filter Construction
# ---------------------------------------------------------------------------


class _FieldProxy:
    """Proxy for a field name, enabling operator overloading for filter construction."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    # -- Comparison operators --

    def __eq__(self, other: object) -> FilterExpression:  # type: ignore[override]
        return FilterExpression(operator=FilterOperator.EQ, field=self._name, value=other)

    def __ne__(self, other: object) -> FilterExpression:  # type: ignore[override]
        return FilterExpression(operator=FilterOperator.NE, field=self._name, value=other)

    def __lt__(self, other: Any) -> FilterExpression:
        return FilterExpression(operator=FilterOperator.LT, field=self._name, value=other)

    def __le__(self, other: Any) -> FilterExpression:
        return FilterExpression(operator=FilterOperator.LTE, field=self._name, value=other)

    def __gt__(self, other: Any) -> FilterExpression:
        return FilterExpression(operator=FilterOperator.GT, field=self._name, value=other)

    def __ge__(self, other: Any) -> FilterExpression:
        return FilterExpression(operator=FilterOperator.GTE, field=self._name, value=other)

    # -- Set operators --

    def in_(self, values: list[Any]) -> FilterExpression:
        """Match if the field value is in the given list."""
        return FilterExpression(operator=FilterOperator.IN, field=self._name, value=values)

    def nin_(self, values: list[Any]) -> FilterExpression:
        """Match if the field value is NOT in the given list."""
        return FilterExpression(operator=FilterOperator.NIN, field=self._name, value=values)

    # -- Range --

    def between(self, low: Any, high: Any) -> FilterExpression:
        """Match if low <= field <= high."""
        return FilterExpression(
            operator=FilterOperator.BETWEEN, field=self._name, value=[low, high]
        )

    # -- Text --

    def contains(self, text: str) -> FilterExpression:
        """Match if the field contains the given text."""
        return FilterExpression(operator=FilterOperator.CONTAINS, field=self._name, value=text)

    def not_contains(self, text: str) -> FilterExpression:
        """Match if the field does NOT contain the given text."""
        return FilterExpression(
            operator=FilterOperator.NOT_CONTAINS, field=self._name, value=text
        )

    def starts_with(self, prefix: str) -> FilterExpression:
        """Match if the field starts with the given prefix."""
        return FilterExpression(
            operator=FilterOperator.STARTS_WITH, field=self._name, value=prefix
        )

    def ends_with(self, suffix: str) -> FilterExpression:
        """Match if the field ends with the given suffix."""
        return FilterExpression(
            operator=FilterOperator.ENDS_WITH, field=self._name, value=suffix
        )

    def regex(self, pattern: str) -> FilterExpression:
        """Match if the field matches the given regex pattern."""
        return FilterExpression(operator=FilterOperator.REGEX, field=self._name, value=pattern)

    # -- Geo --

    def geo_radius(
        self, lat: float, lon: float, radius_km: float
    ) -> FilterExpression:
        """Match if the geo field is within a radius of the given point."""
        return FilterExpression(
            operator=FilterOperator.GEO_RADIUS,
            field=self._name,
            value={"lat": lat, "lon": lon, "radius_km": radius_km},
        )

    # -- Existence --

    def exists(self) -> FilterExpression:
        """Match if the field exists (is not null)."""
        return FilterExpression(
            operator=FilterOperator.EXISTS, field=self._name, value=True
        )


class _FProxy:
    """Proxy class that creates _FieldProxy instances on attribute access.

    Usage::

        F.price < 100        # _FProxy.__getattr__("price") returns _FieldProxy("price")
                              # _FieldProxy.__lt__(100) returns FilterExpression(...)
    """

    def __getattr__(self, name: str) -> _FieldProxy:
        return _FieldProxy(name)


# Module-level singleton
F = _FProxy()
"""Fluent filter builder. Access field names as attributes: ``F.price < 100``."""


# ---------------------------------------------------------------------------
# FilterExpression operator overloads (monkey-patched onto the Pydantic model)
# ---------------------------------------------------------------------------


def _filter_and(self: FilterExpression, other: FilterExpression) -> FilterExpression:
    """Combine two filters with AND."""
    # Flatten nested ANDs: (A & B) & C → AND(A, B, C)
    left_children = (
        self.children if self.operator == FilterOperator.AND and self.children else [self]
    )
    right_children = (
        other.children if other.operator == FilterOperator.AND and other.children else [other]
    )
    return FilterExpression(
        operator=FilterOperator.AND,
        children=[*left_children, *right_children],
    )


def _filter_or(self: FilterExpression, other: FilterExpression) -> FilterExpression:
    """Combine two filters with OR."""
    left_children = (
        self.children if self.operator == FilterOperator.OR and self.children else [self]
    )
    right_children = (
        other.children if other.operator == FilterOperator.OR and other.children else [other]
    )
    return FilterExpression(
        operator=FilterOperator.OR,
        children=[*left_children, *right_children],
    )


def _filter_invert(self: FilterExpression) -> FilterExpression:
    """Negate a filter with NOT."""
    return FilterExpression(operator=FilterOperator.NOT, children=[self])


# Attach operators to FilterExpression via setattr to satisfy mypy strict
setattr(FilterExpression, "__and__", _filter_and)  # noqa: B010
setattr(FilterExpression, "__or__", _filter_or)  # noqa: B010
setattr(FilterExpression, "__invert__", _filter_invert)  # noqa: B010
