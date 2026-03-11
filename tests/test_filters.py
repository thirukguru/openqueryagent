"""Tests for the Filter DSL — F proxy, operators, combinators, and validation."""

from __future__ import annotations

from openqueryagent.core.filters import F, validate_filter
from openqueryagent.core.types import (
    CollectionSchema,
    DataType,
    FilterExpression,
    FilterOperator,
    PropertySchema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schema() -> CollectionSchema:
    """Create a test schema with various field types."""
    return CollectionSchema(
        name="products",
        adapter_id="test",
        properties=[
            PropertySchema(name="price", data_type=DataType.FLOAT, filterable=True),
            PropertySchema(name="name", data_type=DataType.TEXT, filterable=True, searchable=True),
            PropertySchema(name="brand", data_type=DataType.TEXT, filterable=True),
            PropertySchema(name="rating", data_type=DataType.FLOAT, filterable=True),
            PropertySchema(name="reviews", data_type=DataType.INT, filterable=True),
            PropertySchema(name="location", data_type=DataType.GEO, filterable=True),
            PropertySchema(name="embedding", data_type=DataType.FLOAT_ARRAY, filterable=False),
            PropertySchema(name="in_stock", data_type=DataType.BOOL, filterable=True),
        ],
    )


# ---------------------------------------------------------------------------
# F Proxy — Comparison Operators
# ---------------------------------------------------------------------------


class TestFProxyComparison:
    def test_eq(self) -> None:
        f = F.brand == "Nike"
        assert isinstance(f, FilterExpression)
        assert f.operator == FilterOperator.EQ
        assert f.field == "brand"
        assert f.value == "Nike"

    def test_ne(self) -> None:
        f = F.brand != "Adidas"
        assert f.operator == FilterOperator.NE
        assert f.field == "brand"

    def test_lt(self) -> None:
        f = F.price < 100
        assert f.operator == FilterOperator.LT
        assert f.value == 100

    def test_lte(self) -> None:
        f = F.price <= 100
        assert f.operator == FilterOperator.LTE

    def test_gt(self) -> None:
        f = F.price > 50
        assert f.operator == FilterOperator.GT

    def test_gte(self) -> None:
        f = F.rating >= 4.5
        assert f.operator == FilterOperator.GTE
        assert f.value == 4.5


class TestFProxySetOps:
    def test_in(self) -> None:
        f = F.brand.in_(["Nike", "Adidas", "Puma"])
        assert f.operator == FilterOperator.IN
        assert f.value == ["Nike", "Adidas", "Puma"]

    def test_nin(self) -> None:
        f = F.brand.nin_(["Reebok"])
        assert f.operator == FilterOperator.NIN


class TestFProxyRange:
    def test_between(self) -> None:
        f = F.price.between(50, 200)
        assert f.operator == FilterOperator.BETWEEN
        assert f.value == [50, 200]


class TestFProxyText:
    def test_contains(self) -> None:
        f = F.name.contains("wireless")
        assert f.operator == FilterOperator.CONTAINS
        assert f.value == "wireless"

    def test_not_contains(self) -> None:
        f = F.name.not_contains("wired")
        assert f.operator == FilterOperator.NOT_CONTAINS

    def test_starts_with(self) -> None:
        f = F.name.starts_with("Air")
        assert f.operator == FilterOperator.STARTS_WITH

    def test_ends_with(self) -> None:
        f = F.name.ends_with("Pro")
        assert f.operator == FilterOperator.ENDS_WITH

    def test_regex(self) -> None:
        f = F.name.regex(r"^Air\s+Max")
        assert f.operator == FilterOperator.REGEX


class TestFProxyGeo:
    def test_geo_radius(self) -> None:
        f = F.location.geo_radius(37.7749, -122.4194, 10)
        assert f.operator == FilterOperator.GEO_RADIUS
        assert f.value == {"lat": 37.7749, "lon": -122.4194, "radius_km": 10}


class TestFProxyExistence:
    def test_exists(self) -> None:
        f = F.brand.exists()
        assert f.operator == FilterOperator.EXISTS
        assert f.value is True


# ---------------------------------------------------------------------------
# Boolean Combinators
# ---------------------------------------------------------------------------


class TestBooleanCombinators:
    def test_and(self) -> None:
        f = (F.price < 100) & (F.brand == "Nike")
        assert f.operator == FilterOperator.AND
        assert f.children is not None
        assert len(f.children) == 2

    def test_or(self) -> None:
        f = (F.price < 50) | (F.price > 200)
        assert f.operator == FilterOperator.OR
        assert f.children is not None
        assert len(f.children) == 2

    def test_not(self) -> None:
        f = ~(F.price > 100)
        assert f.operator == FilterOperator.NOT
        assert f.children is not None
        assert len(f.children) == 1
        assert f.children[0].operator == FilterOperator.GT

    def test_and_flattening(self) -> None:
        """(A & B) & C should flatten to AND(A, B, C)."""
        f = (F.price < 100) & (F.brand == "Nike") & (F.rating >= 4.0)
        assert f.operator == FilterOperator.AND
        assert f.children is not None
        assert len(f.children) == 3

    def test_or_flattening(self) -> None:
        """(A | B) | C should flatten to OR(A, B, C)."""
        f = (F.price < 50) | (F.price > 200) | (F.brand == "Nike")
        assert f.operator == FilterOperator.OR
        assert f.children is not None
        assert len(f.children) == 3

    def test_mixed_and_or(self) -> None:
        """(A & B) | (C & D) should NOT flatten."""
        f = ((F.price < 100) & (F.brand == "Nike")) | (
            (F.rating >= 4.0) & (F.reviews > 50)
        )
        assert f.operator == FilterOperator.OR
        assert f.children is not None
        assert len(f.children) == 2
        assert f.children[0].operator == FilterOperator.AND
        assert f.children[1].operator == FilterOperator.AND

    def test_complex_nesting(self) -> None:
        """~(A & B) | C."""
        f = ~((F.price < 100) & (F.brand == "Nike")) | (F.rating >= 4.5)
        assert f.operator == FilterOperator.OR
        assert f.children is not None
        assert len(f.children) == 2
        assert f.children[0].operator == FilterOperator.NOT


# ---------------------------------------------------------------------------
# Serialization Round-Trip
# ---------------------------------------------------------------------------


class TestFilterSerialization:
    def test_simple_json_round_trip(self) -> None:
        f = F.price < 100
        data = f.model_dump()
        restored = FilterExpression.model_validate(data)
        assert restored.operator == FilterOperator.LT
        assert restored.field == "price"
        assert restored.value == 100

    def test_compound_json_round_trip(self) -> None:
        f = (F.price < 100) & (F.brand.in_(["Nike", "Adidas"]))
        json_str = f.model_dump_json()
        restored = FilterExpression.model_validate_json(json_str)
        assert restored.operator == FilterOperator.AND
        assert restored.children is not None
        assert len(restored.children) == 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestFilterValidation:
    def test_valid_filter(self) -> None:
        f = F.price < 100
        result = validate_filter(f, _make_schema())
        assert result.is_valid

    def test_unknown_field(self) -> None:
        f = F.unknown_field < 100
        result = validate_filter(f, _make_schema())
        assert not result.is_valid
        assert "Unknown field" in result.errors[0]

    def test_non_filterable_field(self) -> None:
        f = F.embedding == [0.1, 0.2, 0.3]
        result = validate_filter(f, _make_schema())
        assert not result.is_valid
        assert "not filterable" in result.errors[0]

    def test_numeric_op_on_text_field(self) -> None:
        f = F.brand > 100
        result = validate_filter(f, _make_schema())
        assert not result.is_valid
        assert "numeric" in result.errors[0]

    def test_geo_op_on_non_geo_field(self) -> None:
        f = F.price.geo_radius(0, 0, 10)
        result = validate_filter(f, _make_schema())
        assert not result.is_valid
        assert "geo" in result.errors[0]

    def test_text_op_on_numeric_field(self) -> None:
        f = F.price.contains("test")
        result = validate_filter(f, _make_schema())
        assert not result.is_valid
        assert "text" in result.errors[0]

    def test_compound_with_errors(self) -> None:
        f = (F.price < 100) & (F.unknown > 5)
        result = validate_filter(f, _make_schema())
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "Unknown field" in result.errors[0]

    def test_valid_compound(self) -> None:
        f = (F.price < 100) & (F.brand == "Nike") & (F.rating >= 4.0)
        result = validate_filter(f, _make_schema())
        assert result.is_valid
