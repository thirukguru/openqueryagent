"""Tests for REST API endpoints using FastAPI TestClient with mocked QueryAgent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from openqueryagent.core.types import (
    AggregationResponse,
    AggregationResult,
    AskResponse,
    Citation,
    CollectionSchema,
    Document,
    PropertySchema,
    QueryIntent,
    QueryPlan,
    RankedDocument,
    SchemaMap,
    SearchResponse,
    TokenUsage,
)
from openqueryagent.server.api import create_app
from openqueryagent.server.config import ServerConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_agent() -> MagicMock:
    """Create a fully mocked QueryAgent for test injection."""
    agent = MagicMock()

    # Schema inspector
    schema_map = SchemaMap(
        collections={
            "products": CollectionSchema(
                name="products",
                adapter_id="mock",
                properties=[
                    PropertySchema(name="title", data_type="text"),
                    PropertySchema(name="price", data_type="float"),
                ],
            )
        },
        adapter_mapping={"products": "mock"},
    )
    agent.schema_inspector = MagicMock()
    agent.schema_inspector.get_schema_map = AsyncMock(return_value=schema_map)

    # ask()
    agent.ask = AsyncMock(
        return_value=AskResponse(
            answer="Widget A is great at $29.",
            citations=[
                Citation(document_id="d1", collection="products", text_snippet="Widget A"),
            ],
            query_plan=QueryPlan(
                original_query="test",
                intent=QueryIntent.SEARCH,
            ),
            confidence=0.9,
            total_latency_ms=123.4,
            tokens_used=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
    )

    # search()
    agent.search = AsyncMock(
        return_value=SearchResponse(
            documents=[
                RankedDocument(
                    document=Document(id="d1", content="Widget A", collection="products", score=0.95),
                    score=0.95,
                    original_rank=1,
                    new_rank=1,
                ),
            ],
            query_plan=QueryPlan(original_query="test", intent=QueryIntent.SEARCH),
            total_latency_ms=55.0,
        )
    )

    # aggregate()
    agent.aggregate = AsyncMock(
        return_value=AggregationResponse(
            result=AggregationResult(values={"count": 42, "avg_price": 29.99}),
            query_plan=QueryPlan(original_query="test", intent=QueryIntent.AGGREGATE),
            total_latency_ms=80.0,
        )
    )

    # adapters for health check
    mock_adapter = AsyncMock()
    mock_adapter.health_check = AsyncMock(return_value=None)
    agent._adapters = {"mock": mock_adapter}

    return agent


@pytest.fixture
def app() -> TestClient:
    """FastAPI TestClient with mocked agent."""
    config = ServerConfig()
    application = create_app(config)

    # Inject mocked agent directly onto app.state
    application.state.agent = _make_mock_agent()

    return TestClient(application)


# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------


class TestAskEndpoint:
    def test_ask_returns_answer(self, app: TestClient) -> None:
        resp = app.post("/v1/ask", json={"query": "What are the best products?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["answer"] == "Widget A is great at $29."

    def test_ask_includes_citations(self, app: TestClient) -> None:
        resp = app.post("/v1/ask", json={"query": "best products"})
        data = resp.json()
        assert len(data["citations"]) == 1
        assert data["citations"][0]["document_id"] == "d1"

    def test_ask_includes_query_plan(self, app: TestClient) -> None:
        resp = app.post("/v1/ask", json={"query": "test"})
        data = resp.json()
        assert data["query_plan"]["intent"] == "search"

    def test_ask_empty_query_returns_422(self, app: TestClient) -> None:
        resp = app.post("/v1/ask", json={"query": ""})
        assert resp.status_code == 422

    def test_ask_missing_query_returns_422(self, app: TestClient) -> None:
        resp = app.post("/v1/ask", json={})
        assert resp.status_code == 422


class TestSearchEndpoint:
    def test_search_returns_documents(self, app: TestClient) -> None:
        resp = app.post("/v1/search", json={"query": "widgets"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["documents"]) == 1
        assert data["documents"][0]["document"]["id"] == "d1"

    def test_search_with_limit(self, app: TestClient) -> None:
        resp = app.post("/v1/search", json={"query": "widgets", "limit": 5})
        assert resp.status_code == 200

    def test_search_invalid_limit_returns_422(self, app: TestClient) -> None:
        resp = app.post("/v1/search", json={"query": "widgets", "limit": 0})
        assert resp.status_code == 422

    def test_search_limit_too_high_returns_422(self, app: TestClient) -> None:
        resp = app.post("/v1/search", json={"query": "widgets", "limit": 5000})
        assert resp.status_code == 422


class TestAggregateEndpoint:
    def test_aggregate_returns_result(self, app: TestClient) -> None:
        resp = app.post("/v1/aggregate", json={"query": "how many products"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["values"]["count"] == 42

    def test_aggregate_empty_query_returns_422(self, app: TestClient) -> None:
        resp = app.post("/v1/aggregate", json={"query": ""})
        assert resp.status_code == 422


class TestCollectionsEndpoint:
    def test_list_collections(self, app: TestClient) -> None:
        resp = app.get("/v1/collections")
        assert resp.status_code == 200
        data = resp.json()
        assert "products" in data["collections"]

    def test_get_collection_schema(self, app: TestClient) -> None:
        resp = app.get("/v1/collections/products/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "products"
        assert len(data["properties"]) == 2

    def test_get_unknown_collection_returns_404(self, app: TestClient) -> None:
        resp = app.get("/v1/collections/nonexistent/schema")
        assert resp.status_code == 404


class TestHealthEndpoint:
    def test_health_returns_healthy(self, app: TestClient) -> None:
        resp = app.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert len(data["adapters"]) == 1
        assert data["adapters"][0]["status"] == "healthy"

    def test_health_includes_version(self, app: TestClient) -> None:
        resp = app.get("/v1/health")
        data = resp.json()
        assert "version" in data

    def test_health_degraded_on_adapter_error(self, app: TestClient) -> None:
        # Make the adapter health check fail
        app.app.state.agent._adapters["mock"].health_check.side_effect = Exception("connection refused")  # type: ignore[union-attr]
        resp = app.get("/v1/health")
        data = resp.json()
        assert data["status"] in ("degraded", "unhealthy")
        assert data["adapters"][0]["status"] == "unhealthy"
