"""Tests for all three filter compilers — Qdrant, Milvus, pgvector."""

from __future__ import annotations

import pytest

from openqueryagent.adapters.milvus_filters import MilvusFilterCompiler
from openqueryagent.adapters.pgvector_filters import PgvectorFilterCompiler
from openqueryagent.adapters.qdrant_filters import QdrantFilterCompiler
from openqueryagent.core.exceptions import FilterCompilationError
from openqueryagent.core.filters import F
from openqueryagent.core.types import CollectionSchema

DUMMY_SCHEMA = CollectionSchema(name="test", adapter_id="test")


# ===========================================================================
# Qdrant Compiler Tests
# ===========================================================================


class TestQdrantFilterCompiler:
    def setup_method(self) -> None:
        self.compiler = QdrantFilterCompiler()

    def test_eq(self) -> None:
        result = self.compiler.compile(F.brand == "Nike", DUMMY_SCHEMA)
        assert result == {"key": "brand", "match": {"value": "Nike"}}

    def test_ne(self) -> None:
        result = self.compiler.compile(F.brand != "Adidas", DUMMY_SCHEMA)
        assert result == {"must_not": [{"key": "brand", "match": {"value": "Adidas"}}]}

    def test_lt(self) -> None:
        result = self.compiler.compile(F.price < 100, DUMMY_SCHEMA)
        assert result == {"key": "price", "range": {"lt": 100}}

    def test_gte(self) -> None:
        result = self.compiler.compile(F.price >= 50, DUMMY_SCHEMA)
        assert result == {"key": "price", "range": {"gte": 50}}

    def test_in(self) -> None:
        result = self.compiler.compile(F.brand.in_(["Nike", "Adidas"]), DUMMY_SCHEMA)
        assert result == {"key": "brand", "match": {"any": ["Nike", "Adidas"]}}

    def test_nin(self) -> None:
        result = self.compiler.compile(F.brand.nin_(["Reebok"]), DUMMY_SCHEMA)
        assert result == {"must_not": [{"key": "brand", "match": {"any": ["Reebok"]}}]}

    def test_contains(self) -> None:
        result = self.compiler.compile(F.name.contains("wireless"), DUMMY_SCHEMA)
        assert result == {"key": "name", "match": {"text": "wireless"}}

    def test_between(self) -> None:
        result = self.compiler.compile(F.price.between(50, 200), DUMMY_SCHEMA)
        assert result == {
            "must": [
                {"key": "price", "range": {"gte": 50}},
                {"key": "price", "range": {"lte": 200}},
            ],
        }

    def test_and(self) -> None:
        f = (F.price < 100) & (F.brand == "Nike")
        result = self.compiler.compile(f, DUMMY_SCHEMA)
        assert "must" in result
        assert len(result["must"]) == 2

    def test_or(self) -> None:
        f = (F.price < 50) | (F.price > 200)
        result = self.compiler.compile(f, DUMMY_SCHEMA)
        assert "should" in result
        assert len(result["should"]) == 2

    def test_not(self) -> None:
        f = ~(F.price > 100)
        result = self.compiler.compile(f, DUMMY_SCHEMA)
        assert "must_not" in result

    def test_geo_radius(self) -> None:
        f = F.location.geo_radius(37.77, -122.42, 10)
        result = self.compiler.compile(f, DUMMY_SCHEMA)
        assert result["key"] == "location"
        assert result["geo_radius"]["center"]["lat"] == 37.77
        assert result["geo_radius"]["radius"] == 10000  # km → meters

    def test_unsupported_starts_with(self) -> None:
        with pytest.raises(FilterCompilationError) as exc_info:
            self.compiler.compile(F.name.starts_with("Air"), DUMMY_SCHEMA)
        assert "not supported by Qdrant" in str(exc_info.value)

    def test_unsupported_regex(self) -> None:
        with pytest.raises(FilterCompilationError):
            self.compiler.compile(F.name.regex("test"), DUMMY_SCHEMA)

    def test_validate_unsupported(self) -> None:
        errors = self.compiler.validate(F.name.starts_with("Air"), DUMMY_SCHEMA)
        assert len(errors) == 1
        assert errors[0].operator == "$starts_with"


# ===========================================================================
# Milvus Compiler Tests
# ===========================================================================


class TestMilvusFilterCompiler:
    def setup_method(self) -> None:
        self.compiler = MilvusFilterCompiler()

    def test_eq(self) -> None:
        result = self.compiler.compile(F.brand == "Nike", DUMMY_SCHEMA)
        assert result == 'brand == "Nike"'

    def test_ne(self) -> None:
        result = self.compiler.compile(F.brand != "Adidas", DUMMY_SCHEMA)
        assert result == 'brand != "Adidas"'

    def test_lt(self) -> None:
        result = self.compiler.compile(F.price < 100, DUMMY_SCHEMA)
        assert result == "price < 100"

    def test_gte(self) -> None:
        result = self.compiler.compile(F.price >= 50.5, DUMMY_SCHEMA)
        assert result == "price >= 50.5"

    def test_in(self) -> None:
        result = self.compiler.compile(F.brand.in_(["Nike", "Adidas"]), DUMMY_SCHEMA)
        assert result == 'brand in ["Nike", "Adidas"]'

    def test_nin(self) -> None:
        result = self.compiler.compile(F.brand.nin_(["Reebok"]), DUMMY_SCHEMA)
        assert result == 'brand not in ["Reebok"]'

    def test_contains(self) -> None:
        result = self.compiler.compile(F.name.contains("wireless"), DUMMY_SCHEMA)
        assert result == 'name like "%wireless%"'

    def test_starts_with(self) -> None:
        result = self.compiler.compile(F.name.starts_with("Air"), DUMMY_SCHEMA)
        assert result == 'name like "Air%"'

    def test_between(self) -> None:
        result = self.compiler.compile(F.price.between(50, 200), DUMMY_SCHEMA)
        assert result == "(price >= 50 and price <= 200)"

    def test_exists(self) -> None:
        result = self.compiler.compile(F.brand.exists(), DUMMY_SCHEMA)
        assert result == "brand != null"

    def test_and(self) -> None:
        f = (F.price < 100) & (F.brand == "Nike")
        result = self.compiler.compile(f, DUMMY_SCHEMA)
        assert result == '(price < 100 and brand == "Nike")'

    def test_or(self) -> None:
        f = (F.price < 50) | (F.price > 200)
        result = self.compiler.compile(f, DUMMY_SCHEMA)
        assert result == "(price < 50 or price > 200)"

    def test_not(self) -> None:
        f = ~(F.price > 100)
        result = self.compiler.compile(f, DUMMY_SCHEMA)
        assert result == "not (price > 100)"

    def test_complex(self) -> None:
        f = (F.price < 100) & (F.brand.in_(["Nike", "Adidas"])) & (F.rating >= 4.0)
        result = self.compiler.compile(f, DUMMY_SCHEMA)
        assert "and" in result
        assert "Nike" in result
        assert "rating >= 4.0" in result

    def test_unsupported_regex(self) -> None:
        with pytest.raises(FilterCompilationError) as exc_info:
            self.compiler.compile(F.name.regex("test"), DUMMY_SCHEMA)
        assert "not supported by Milvus" in str(exc_info.value)

    def test_bool_value(self) -> None:
        result = self.compiler.compile(F.in_stock == True, DUMMY_SCHEMA)  # noqa: E712
        assert result == "in_stock == true"

    def test_string_escaping(self) -> None:
        result = self.compiler.compile(F.name == 'O"Brien', DUMMY_SCHEMA)
        assert '\\"' in result


# ===========================================================================
# pgvector Compiler Tests
# ===========================================================================


class TestPgvectorFilterCompiler:
    def setup_method(self) -> None:
        self.compiler = PgvectorFilterCompiler()

    def test_eq(self) -> None:
        result = self.compiler.compile(F.brand == "Nike", DUMMY_SCHEMA)
        assert result.sql == '"brand" = $1'
        assert result.params == ["Nike"]

    def test_ne(self) -> None:
        result = self.compiler.compile(F.brand != "Adidas", DUMMY_SCHEMA)
        assert result.sql == '"brand" != $1'
        assert result.params == ["Adidas"]

    def test_lt(self) -> None:
        result = self.compiler.compile(F.price < 100, DUMMY_SCHEMA)
        assert result.sql == '"price" < $1'
        assert result.params == [100]

    def test_gte(self) -> None:
        result = self.compiler.compile(F.price >= 50, DUMMY_SCHEMA)
        assert result.sql == '"price" >= $1'
        assert result.params == [50]

    def test_in(self) -> None:
        result = self.compiler.compile(F.brand.in_(["Nike", "Adidas"]), DUMMY_SCHEMA)
        assert result.sql == '"brand" = ANY($1)'
        assert result.params == [["Nike", "Adidas"]]

    def test_nin(self) -> None:
        result = self.compiler.compile(F.brand.nin_(["Reebok"]), DUMMY_SCHEMA)
        assert result.sql == '"brand" != ALL($1)'
        assert result.params == [["Reebok"]]

    def test_contains(self) -> None:
        result = self.compiler.compile(F.name.contains("wireless"), DUMMY_SCHEMA)
        assert result.sql == '"name" ILIKE $1'
        assert result.params == ["%wireless%"]

    def test_not_contains(self) -> None:
        result = self.compiler.compile(F.name.not_contains("wired"), DUMMY_SCHEMA)
        assert result.sql == '"name" NOT ILIKE $1'
        assert result.params == ["%wired%"]

    def test_starts_with(self) -> None:
        result = self.compiler.compile(F.name.starts_with("Air"), DUMMY_SCHEMA)
        assert result.sql == '"name" ILIKE $1'
        assert result.params == ["Air%"]

    def test_ends_with(self) -> None:
        result = self.compiler.compile(F.name.ends_with("Pro"), DUMMY_SCHEMA)
        assert result.sql == '"name" ILIKE $1'
        assert result.params == ["%Pro"]

    def test_regex(self) -> None:
        result = self.compiler.compile(F.name.regex(r"^Air\s+Max"), DUMMY_SCHEMA)
        assert result.sql == '"name" ~ $1'
        assert result.params == [r"^Air\s+Max"]

    def test_between(self) -> None:
        result = self.compiler.compile(F.price.between(50, 200), DUMMY_SCHEMA)
        assert result.sql == '"price" BETWEEN $1 AND $2'
        assert result.params == [50, 200]

    def test_exists(self) -> None:
        result = self.compiler.compile(F.brand.exists(), DUMMY_SCHEMA)
        assert result.sql == '"brand" IS NOT NULL'
        assert result.params == []

    def test_and(self) -> None:
        f = (F.price < 100) & (F.brand == "Nike")
        result = self.compiler.compile(f, DUMMY_SCHEMA)
        assert result.sql == '("price" < $1 AND "brand" = $2)'
        assert result.params == [100, "Nike"]

    def test_or(self) -> None:
        f = (F.price < 50) | (F.price > 200)
        result = self.compiler.compile(f, DUMMY_SCHEMA)
        assert result.sql == '("price" < $1 OR "price" > $2)'
        assert result.params == [50, 200]

    def test_not(self) -> None:
        f = ~(F.price > 100)
        result = self.compiler.compile(f, DUMMY_SCHEMA)
        assert result.sql == 'NOT ("price" > $1)'
        assert result.params == [100]

    def test_param_ordering(self) -> None:
        """Parameters are numbered sequentially across the entire expression."""
        f = (F.price < 100) & (F.brand == "Nike") & (F.rating >= 4.0)
        result = self.compiler.compile(f, DUMMY_SCHEMA)
        assert "$1" in result.sql
        assert "$2" in result.sql
        assert "$3" in result.sql
        assert result.params == [100, "Nike", 4.0]

    def test_sql_injection_field_quoting(self) -> None:
        """Field names are double-quoted to prevent SQL injection."""
        f = F.price < 100
        result = self.compiler.compile(f, DUMMY_SCHEMA)
        assert '"price"' in result.sql

    def test_unsupported_geo_radius(self) -> None:
        with pytest.raises(FilterCompilationError) as exc_info:
            self.compiler.compile(F.location.geo_radius(0, 0, 10), DUMMY_SCHEMA)
        assert "not supported by pgvector" in str(exc_info.value)


# ===========================================================================
# Cross-Compiler Consistency Tests
# ===========================================================================


class TestCrossCompilerConsistency:
    """Verify that all compilers accept the same core operators."""

    def test_all_compile_simple_eq(self) -> None:
        f = F.brand == "Nike"
        qdrant = QdrantFilterCompiler().compile(f, DUMMY_SCHEMA)
        milvus = MilvusFilterCompiler().compile(f, DUMMY_SCHEMA)
        pgvector = PgvectorFilterCompiler().compile(f, DUMMY_SCHEMA)
        assert qdrant is not None
        assert milvus is not None
        assert pgvector is not None

    def test_all_compile_and_or(self) -> None:
        f = (F.price < 100) | (F.brand == "Nike")
        qdrant = QdrantFilterCompiler().compile(f, DUMMY_SCHEMA)
        milvus = MilvusFilterCompiler().compile(f, DUMMY_SCHEMA)
        pgvector = PgvectorFilterCompiler().compile(f, DUMMY_SCHEMA)
        assert qdrant is not None
        assert milvus is not None
        assert pgvector is not None

    def test_extended_op_compile_error_pattern(self) -> None:
        """Extended operators that aren't supported should raise FilterCompilationError."""
        f = F.name.regex("pattern")
        # Qdrant doesn't support regex
        with pytest.raises(FilterCompilationError):
            QdrantFilterCompiler().compile(f, DUMMY_SCHEMA)
        # Milvus doesn't support regex
        with pytest.raises(FilterCompilationError):
            MilvusFilterCompiler().compile(f, DUMMY_SCHEMA)
        # pgvector DOES support regex (via ~ operator), so it should NOT raise
        result = PgvectorFilterCompiler().compile(f, DUMMY_SCHEMA)
        assert result.sql == '"name" ~ $1'
