"""Tests for QueryAgent — end-to-end pipeline tests with mocked components."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from openqueryagent.core.agent import QueryAgent
from openqueryagent.core.types import (
    CollectionSchema,
    Document,
    PropertySchema,
    SearchResult,
    TokenUsage,
)
from openqueryagent.llm.base import LLMResponse


def _make_mock_adapter() -> AsyncMock:
    adapter = AsyncMock()
    adapter.adapter_id = "mock-adapter"
    adapter.adapter_name = "mock"
    adapter.supports_native_aggregation = False
    adapter.get_collections.return_value = ["products"]
    adapter.get_schema.return_value = CollectionSchema(
        name="products",
        adapter_id="mock-adapter",
        properties=[
            PropertySchema(name="title", data_type="text"),
            PropertySchema(name="price", data_type="float"),
        ],
    )
    adapter.search.return_value = SearchResult(
        documents=[
            Document(id="d1", content="Widget A is great at $29", collection="products", score=0.95),
            Document(id="d2", content="Gadget B costs $45", collection="products", score=0.88),
        ],
        total_count=2,
    )
    adapter.get_filter_compiler.return_value = MagicMock()
    return adapter


def _make_mock_llm() -> AsyncMock:
    mock_llm = AsyncMock()
    # First call: planner
    plan_response = LLMResponse(
        content=json.dumps({
            "intent": "search",
            "reasoning": "Product search",
            "requires_synthesis": True,
            "sub_queries": [
                {
                    "id": "q1",
                    "collection": "products",
                    "query_text": "best products under $50",
                    "search_type": "hybrid",
                    "limit": 10,
                }
            ],
        }),
        model="gpt-4o-mini",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )
    # Second call: synthesizer
    synth_response = LLMResponse(
        content="Based on the results, Widget A [1] and Gadget B [2] are great choices under $50.",
        model="gpt-4o-mini",
        usage=TokenUsage(prompt_tokens=200, completion_tokens=60, total_tokens=260),
    )
    mock_llm.complete.side_effect = [plan_response, synth_response]
    return mock_llm


class TestQueryAgent:
    @pytest.mark.asyncio
    async def test_ask_pipeline(self) -> None:
        adapter = _make_mock_adapter()
        llm = _make_mock_llm()

        agent = QueryAgent(
            adapters={"mock-adapter": adapter},
            llm=llm,
        )
        await agent.initialize()

        response = await agent.ask("best products under $50")

        assert response.answer  # type: ignore[union-attr]
        assert response.query_plan is not None  # type: ignore[union-attr]
        assert response.total_latency_ms > 0  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_search_pipeline(self) -> None:
        adapter = _make_mock_adapter()
        llm = AsyncMock()
        llm.complete.return_value = LLMResponse(
            content=json.dumps({
                "intent": "search",
                "reasoning": "Simple search",
                "requires_synthesis": False,
                "sub_queries": [
                    {"id": "q1", "collection": "products", "query_text": "widgets", "search_type": "hybrid"}
                ],
            }),
            model="gpt-4o-mini",
            usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
        )

        agent = QueryAgent(adapters={"mock-adapter": adapter}, llm=llm)
        await agent.initialize()

        response = await agent.search("widgets")
        assert len(response.documents) > 0
        assert response.query_plan is not None

    @pytest.mark.asyncio
    async def test_no_llm_uses_simple_planner(self) -> None:
        adapter = _make_mock_adapter()

        agent = QueryAgent(adapters={"mock-adapter": adapter})
        await agent.initialize()

        # Without LLM, should use SimpleQueryPlanner and no synthesis
        response = await agent.ask("test query")
        assert response.answer  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_memory_tracks_conversation(self) -> None:
        adapter = _make_mock_adapter()

        agent = QueryAgent(adapters={"mock-adapter": adapter})
        await agent.initialize()

        await agent.ask("first question")
        assert agent.memory.message_count >= 2  # user + assistant

        await agent.ask("follow up")
        assert agent.memory.message_count >= 4

    @pytest.mark.asyncio
    async def test_schema_inspector_accessible(self) -> None:
        adapter = _make_mock_adapter()
        agent = QueryAgent(adapters={"mock-adapter": adapter})
        await agent.initialize()

        schema_map = await agent.schema_inspector.get_schema_map()
        assert "products" in schema_map.collections
