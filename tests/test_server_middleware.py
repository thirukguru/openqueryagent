"""Tests for server middleware: API key auth, rate limiting, request ID."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from openqueryagent.core.types import (
    AskResponse,
    CollectionSchema,
    PropertySchema,
    QueryIntent,
    QueryPlan,
    SchemaMap,
)
from openqueryagent.server.api import create_app
from openqueryagent.server.config import ServerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_agent() -> MagicMock:
    agent = MagicMock()
    schema_map = SchemaMap(
        collections={
            "products": CollectionSchema(
                name="products", adapter_id="mock",
                properties=[PropertySchema(name="title", data_type="text")],
            )
        },
        adapter_mapping={"products": "mock"},
    )
    agent.schema_inspector = MagicMock()
    agent.schema_inspector.get_schema_map = AsyncMock(return_value=schema_map)
    agent.ask = AsyncMock(
        return_value=AskResponse(
            answer="test answer",
            query_plan=QueryPlan(original_query="test", intent=QueryIntent.SEARCH),
            total_latency_ms=10.0,
        )
    )
    agent._adapters = {}
    return agent


def _create_app_with(api_key: str | None = None, rate_limit: str | None = None) -> TestClient:
    config = ServerConfig(api_key=api_key, rate_limit=rate_limit)
    application = create_app(config)
    application.state.agent = _make_mock_agent()
    return TestClient(application)


# ---------------------------------------------------------------------------
# API Key Middleware Tests
# ---------------------------------------------------------------------------


class TestAPIKeyMiddleware:
    def test_no_key_configured_allows_all(self) -> None:
        app = _create_app_with(api_key=None)
        resp = app.post("/v1/ask", json={"query": "test"})
        assert resp.status_code == 200

    def test_correct_key_passes(self) -> None:
        app = _create_app_with(api_key="secret-123")
        resp = app.post("/v1/ask", json={"query": "test"}, headers={"X-API-Key": "secret-123"})
        assert resp.status_code == 200

    def test_wrong_key_returns_401(self) -> None:
        app = _create_app_with(api_key="secret-123")
        resp = app.post("/v1/ask", json={"query": "test"}, headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401
        assert resp.json()["error"] == "Unauthorized"

    def test_missing_key_returns_401(self) -> None:
        app = _create_app_with(api_key="secret-123")
        resp = app.post("/v1/ask", json={"query": "test"})
        assert resp.status_code == 401

    def test_health_bypasses_auth(self) -> None:
        app = _create_app_with(api_key="secret-123")
        resp = app.get("/v1/health")
        assert resp.status_code == 200

    def test_docs_bypasses_auth(self) -> None:
        app = _create_app_with(api_key="secret-123")
        resp = app.get("/docs")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Rate Limit Middleware Tests
# ---------------------------------------------------------------------------


class TestRateLimitMiddleware:
    def test_under_limit_passes(self) -> None:
        app = _create_app_with(rate_limit="10/minute")
        for _ in range(5):
            resp = app.post("/v1/ask", json={"query": "test"})
            assert resp.status_code == 200

    def test_over_limit_returns_429(self) -> None:
        app = _create_app_with(rate_limit="3/minute")
        for _ in range(3):
            resp = app.post("/v1/ask", json={"query": "test"})
            assert resp.status_code == 200

        resp = app.post("/v1/ask", json={"query": "test"})
        assert resp.status_code == 429
        assert resp.json()["error"] == "Too Many Requests"

    def test_health_not_rate_limited(self) -> None:
        app = _create_app_with(rate_limit="2/minute")
        for _ in range(10):
            resp = app.get("/v1/health")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Request ID Middleware Tests
# ---------------------------------------------------------------------------


class TestRequestIDMiddleware:
    def test_response_has_request_id(self) -> None:
        app = _create_app_with()
        resp = app.get("/v1/health")
        assert "X-Request-ID" in resp.headers

    def test_request_id_preserved_from_client(self) -> None:
        app = _create_app_with()
        resp = app.get("/v1/health", headers={"X-Request-ID": "my-req-123"})
        assert resp.headers["X-Request-ID"] == "my-req-123"

    def test_request_id_generated_when_missing(self) -> None:
        app = _create_app_with()
        resp = app.get("/v1/health")
        req_id = resp.headers.get("X-Request-ID", "")
        assert len(req_id) > 0
        # Should look like a UUID
        assert "-" in req_id
