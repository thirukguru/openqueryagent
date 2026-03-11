"""Tests for QueryRouter."""

from __future__ import annotations

from unittest.mock import MagicMock

from openqueryagent.core.router import QueryRouter
from openqueryagent.core.types import (
    CollectionSchema,
    PropertySchema,
    QueryIntent,
    QueryPlan,
    SchemaMap,
    SearchType,
    SubQuery,
)


def _make_schema_map() -> SchemaMap:
    return SchemaMap(
        collections={
            "products": CollectionSchema(
                name="products",
                adapter_id="qdrant",
                properties=[
                    PropertySchema(name="title", data_type="text"),
                ],
            ),
        },
        adapter_mapping={"products": "qdrant"},
    )


def _make_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.get_filter_compiler.return_value = MagicMock()
    return adapter


class TestQueryRouter:
    def test_resolve_exact_match(self) -> None:
        router = QueryRouter(adapters={}, schema_map=_make_schema_map())
        assert router.resolve_collection("products") == "products"

    def test_resolve_case_insensitive(self) -> None:
        router = QueryRouter(adapters={}, schema_map=_make_schema_map())
        assert router.resolve_collection("Products") == "products"

    def test_resolve_substring(self) -> None:
        router = QueryRouter(adapters={}, schema_map=_make_schema_map())
        assert router.resolve_collection("prod") == "products"

    def test_resolve_not_found(self) -> None:
        router = QueryRouter(adapters={}, schema_map=_make_schema_map())
        assert router.resolve_collection("nonexistent") is None

    def test_get_adapter(self) -> None:
        adapter = _make_adapter()
        router = QueryRouter(
            adapters={"qdrant": adapter},
            schema_map=_make_schema_map(),
        )
        assert router.get_adapter_for_collection("products") == adapter

    def test_get_adapter_not_found(self) -> None:
        router = QueryRouter(adapters={}, schema_map=_make_schema_map())
        assert router.get_adapter_for_collection("products") is None

    def test_route_plan(self) -> None:
        adapter = _make_adapter()
        router = QueryRouter(
            adapters={"qdrant": adapter},
            schema_map=_make_schema_map(),
        )

        plan = QueryPlan(
            original_query="test",
            intent=QueryIntent.SEARCH,
            sub_queries=[
                SubQuery(
                    id="q1",
                    collection="products",
                    query_text="test",
                    search_type=SearchType.HYBRID,
                ),
            ],
        )

        routed = router.route(plan)
        assert len(routed) == 1
        assert routed[0]["collection"] == "products"
        assert routed[0]["adapter"] == adapter

    def test_route_skips_unknown_collection(self) -> None:
        adapter = _make_adapter()
        router = QueryRouter(
            adapters={"qdrant": adapter},
            schema_map=_make_schema_map(),
        )

        plan = QueryPlan(
            original_query="test",
            intent=QueryIntent.SEARCH,
            sub_queries=[
                SubQuery(id="q1", collection="nonexistent", query_text="test"),
            ],
        )

        routed = router.route(plan)
        assert len(routed) == 0
