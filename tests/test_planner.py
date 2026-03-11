"""Tests for query planners — LLMQueryPlanner and SimpleQueryPlanner."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from openqueryagent.core.planner import LLMQueryPlanner, SimpleQueryPlanner
from openqueryagent.core.types import (
    CollectionSchema,
    PropertySchema,
    QueryIntent,
    SchemaMap,
    SearchType,
    TokenUsage,
)
from openqueryagent.llm.base import LLMResponse


def _make_schema_map() -> SchemaMap:
    return SchemaMap(
        collections={
            "products": CollectionSchema(
                name="products",
                adapter_id="qdrant",
                properties=[
                    PropertySchema(name="title", data_type="text"),
                    PropertySchema(name="price", data_type="float"),
                ],
            ),
            "reviews": CollectionSchema(
                name="reviews",
                adapter_id="qdrant",
                properties=[
                    PropertySchema(name="text", data_type="text"),
                    PropertySchema(name="rating", data_type="int"),
                ],
            ),
        },
        adapter_mapping={"products": "qdrant", "reviews": "qdrant"},
    )


class TestLLMQueryPlanner:
    @pytest.mark.asyncio
    async def test_plan_single_query(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = LLMResponse(
            content=json.dumps({
                "intent": "search",
                "reasoning": "Simple product search",
                "requires_synthesis": True,
                "sub_queries": [
                    {
                        "id": "q1",
                        "collection": "products",
                        "query_text": "best headphones",
                        "search_type": "hybrid",
                        "limit": 10,
                    }
                ],
            }),
            model="gpt-4o-mini",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )

        planner = LLMQueryPlanner(llm=mock_llm)
        plan = await planner.plan("best headphones", _make_schema_map())

        assert plan.intent == QueryIntent.SEARCH
        assert len(plan.sub_queries) == 1
        assert plan.sub_queries[0].collection == "products"
        assert plan.sub_queries[0].search_type == SearchType.HYBRID

    @pytest.mark.asyncio
    async def test_plan_aggregation(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = LLMResponse(
            content=json.dumps({
                "intent": "aggregate",
                "reasoning": "Count query",
                "requires_synthesis": False,
                "sub_queries": [
                    {
                        "id": "q1",
                        "collection": "products",
                        "query_text": "how many products",
                        "search_type": "keyword",
                        "limit": 10,
                        "aggregation": {"operation": "count"},
                    }
                ],
            }),
            model="gpt-4o-mini",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )

        planner = LLMQueryPlanner(llm=mock_llm)
        plan = await planner.plan("how many products?", _make_schema_map())

        assert plan.intent == QueryIntent.AGGREGATE
        assert plan.sub_queries[0].aggregation is not None
        assert plan.sub_queries[0].aggregation.operation == "count"

    @pytest.mark.asyncio
    async def test_plan_fallback_on_invalid_json(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = Exception("LLM failed")

        planner = LLMQueryPlanner(llm=mock_llm, max_retries=1)
        plan = await planner.plan("test query", _make_schema_map())

        # Should fall back to simple plan
        assert plan.intent == QueryIntent.SEARCH
        assert len(plan.sub_queries) == 1
        assert plan.reasoning.startswith("Fallback")


class TestSimpleQueryPlanner:
    @pytest.mark.asyncio
    async def test_plan_uses_first_collection(self) -> None:
        planner = SimpleQueryPlanner()
        plan = await planner.plan("test query", _make_schema_map())

        assert plan.intent == QueryIntent.SEARCH
        assert len(plan.sub_queries) == 1
        assert plan.sub_queries[0].collection in ("products", "reviews")

    @pytest.mark.asyncio
    async def test_plan_with_default_collection(self) -> None:
        planner = SimpleQueryPlanner(default_collection="reviews")
        plan = await planner.plan("test", _make_schema_map())

        assert plan.sub_queries[0].collection == "reviews"

    @pytest.mark.asyncio
    async def test_plan_empty_schema(self) -> None:
        planner = SimpleQueryPlanner()
        plan = await planner.plan("test", SchemaMap())

        assert plan.sub_queries[0].collection == ""
