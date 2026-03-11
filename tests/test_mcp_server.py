"""Tests for MCP stdio server."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types as mcp_types

from openqueryagent.server.mcp_server import create_mcp_server


@pytest.fixture
def mock_agent():
    """Create a mock QueryAgent."""
    agent = AsyncMock()
    agent.list_collections = AsyncMock(return_value=["products", "articles"])

    schema_mock = MagicMock()
    schema_mock.model_dump.return_value = {
        "name": "products",
        "fields": [{"name": "title", "type": "string"}],
    }
    agent.get_collection_schema = AsyncMock(return_value=schema_mock)

    ask_response = MagicMock()
    ask_response.answer = "The answer is 42."
    ask_response.citations = []
    ask_response.confidence = 0.95
    ask_response.total_latency_ms = 123.4
    agent.ask = AsyncMock(return_value=ask_response)

    search_response = MagicMock()
    doc = MagicMock()
    doc.document.id = "doc-1"
    doc.document.content = "Some content"
    doc.document.collection = "products"
    doc.score = 0.92
    search_response.documents = [doc]
    search_response.total_latency_ms = 50.0
    agent.search = AsyncMock(return_value=search_response)

    agg_response = MagicMock()
    agg_result = MagicMock()
    agg_result.model_dump.return_value = {"count": 42}
    agg_response.result = agg_result
    agg_response.total_latency_ms = 30.0
    agent.aggregate = AsyncMock(return_value=agg_response)

    return agent


class TestMCPServerCreation:
    def test_server_name(self) -> None:
        server = create_mcp_server()
        assert server.name == "openqueryagent"

    def test_tools_handler_registered(self) -> None:
        server = create_mcp_server()
        assert mcp_types.ListToolsRequest in server.request_handlers

    def test_call_tool_handler_registered(self) -> None:
        server = create_mcp_server()
        assert mcp_types.CallToolRequest in server.request_handlers

    def test_resources_handler_registered(self) -> None:
        server = create_mcp_server()
        assert mcp_types.ListResourcesRequest in server.request_handlers

    def test_read_resource_handler_registered(self) -> None:
        server = create_mcp_server()
        assert mcp_types.ReadResourceRequest in server.request_handlers

    def test_all_four_handlers_registered(self) -> None:
        server = create_mcp_server()
        expected = [
            mcp_types.ListToolsRequest,
            mcp_types.CallToolRequest,
            mcp_types.ListResourcesRequest,
            mcp_types.ReadResourceRequest,
        ]
        for req_type in expected:
            assert req_type in server.request_handlers, f"{req_type.__name__} not registered"

    def test_server_has_initialization_options(self) -> None:
        server = create_mcp_server()
        options = server.create_initialization_options()
        assert options is not None


