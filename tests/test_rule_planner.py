"""Tests for RuleBasedPlanner."""

from __future__ import annotations

import pytest

from openqueryagent.core.rule_planner import RuleBasedPlanner
from openqueryagent.core.types import (
    CollectionSchema,
    PropertySchema,
    QueryIntent,
    SchemaMap,
    SearchType,
)


def _schema_map() -> SchemaMap:
    """Create a test schema map with two collections."""
    return SchemaMap(
        collections={
            "products": CollectionSchema(
                name="products",
                adapter_id="qdrant",
                properties=[
                    PropertySchema(name="price", data_type="float"),
                    PropertySchema(name="title", data_type="text"),
                    PropertySchema(name="category", data_type="text"),
                ],
            ),
            "articles": CollectionSchema(
                name="articles",
                adapter_id="qdrant",
                properties=[
                    PropertySchema(name="author", data_type="text"),
                    PropertySchema(name="content", data_type="text"),
                ],
            ),
        },
        adapter_mapping={"products": "qdrant", "articles": "qdrant"},
    )


class TestRuleBasedPlanner:
    @pytest.mark.asyncio
    async def test_simple_search(self) -> None:
        planner = RuleBasedPlanner()
        plan = await planner.plan("find cheap laptops", _schema_map())
        assert plan.intent == QueryIntent.SEARCH
        assert len(plan.sub_queries) >= 1

    @pytest.mark.asyncio
    async def test_aggregation_detected(self) -> None:
        planner = RuleBasedPlanner()
        plan = await planner.plan("how many products are there?", _schema_map())
        assert plan.intent == QueryIntent.AGGREGATE
        assert plan.sub_queries[0].aggregation is not None
        assert plan.sub_queries[0].aggregation.operation == "count"

    @pytest.mark.asyncio
    async def test_collection_matching(self) -> None:
        planner = RuleBasedPlanner()
        plan = await planner.plan("search articles about science", _schema_map())
        assert plan.sub_queries[0].collection == "articles"

    @pytest.mark.asyncio
    async def test_multi_collection(self) -> None:
        planner = RuleBasedPlanner()
        plan = await planner.plan("search products and articles", _schema_map())
        assert len(plan.sub_queries) == 2

    @pytest.mark.asyncio
    async def test_keyword_search_type(self) -> None:
        planner = RuleBasedPlanner()
        plan = await planner.plan("filter products where price equals 100", _schema_map())
        assert plan.sub_queries[0].search_type == SearchType.KEYWORD

    @pytest.mark.asyncio
    async def test_vector_search_type(self) -> None:
        planner = RuleBasedPlanner()
        plan = await planner.plan("find products similar to laptops", _schema_map())
        assert plan.sub_queries[0].search_type == SearchType.VECTOR

    @pytest.mark.asyncio
    async def test_fallback_collection(self) -> None:
        planner = RuleBasedPlanner(default_collection="default_coll")
        plan = await planner.plan("hello world", _schema_map())
        assert plan.sub_queries[0].collection == "default_coll"

    @pytest.mark.asyncio
    async def test_agg_field_detection(self) -> None:
        planner = RuleBasedPlanner()
        plan = await planner.plan("average price of products", _schema_map())
        assert plan.sub_queries[0].aggregation is not None
        assert plan.sub_queries[0].aggregation.field == "price"

    @pytest.mark.asyncio
    async def test_empty_schema(self) -> None:
        planner = RuleBasedPlanner()
        plan = await planner.plan("hello", SchemaMap())
        assert plan.sub_queries[0].collection == ""

    @pytest.mark.asyncio
    async def test_default_search_type(self) -> None:
        planner = RuleBasedPlanner(default_search_type=SearchType.VECTOR)
        plan = await planner.plan("some random query", _schema_map())
        # Default is VECTOR but query has no exact/semantic keywords → HYBRID
        assert plan.sub_queries[0].search_type == SearchType.HYBRID
