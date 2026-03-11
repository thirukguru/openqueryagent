"""MCP (Model Context Protocol) server for OpenQueryAgent.

Exposes OpenQueryAgent as MCP tools and resources over **stdio** transport,
enabling integration with Claude Desktop, Cursor, and other MCP clients.

Tools:
    - ``openqueryagent_ask`` — natural language Q&A
    - ``openqueryagent_search`` — document retrieval
    - ``openqueryagent_aggregate`` — aggregation queries

Resources:
    - ``oqa://collections`` — list all collections
    - ``oqa://collections/{name}/schema`` — collection schema

Usage::

    python -m openqueryagent.server.mcp_server

Or via the installed script::

    openqueryagent-mcp
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import structlog

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_agent_instance: Any = None


async def _get_or_create_agent() -> Any:
    """Lazily initialise the QueryAgent singleton."""
    global _agent_instance
    if _agent_instance is not None:
        return _agent_instance

    from openqueryagent.core.agent import QueryAgent
    from openqueryagent.core.config import AgentConfig

    adapters: dict[str, Any] = {}

    # Auto-discover adapters from environment — same logic as server/api.py
    qdrant_url = os.environ.get("OQA_QDRANT_URL")
    if qdrant_url:
        try:
            from openqueryagent.adapters.qdrant import QdrantAdapter
            adapter = QdrantAdapter(url=qdrant_url)
            await adapter.connect()
            adapters["qdrant"] = adapter
            logger.info("mcp_adapter_connected", adapter="qdrant")
        except Exception as e:
            logger.warning("mcp_adapter_failed", adapter="qdrant", error=str(e))

    pgvector_dsn = os.environ.get("OQA_PGVECTOR_DSN")
    if pgvector_dsn:
        try:
            from openqueryagent.adapters.pgvector import PgVectorAdapter
            adapter = PgVectorAdapter(dsn=pgvector_dsn)
            await adapter.connect()
            adapters["pgvector"] = adapter
            logger.info("mcp_adapter_connected", adapter="pgvector")
        except Exception as e:
            logger.warning("mcp_adapter_failed", adapter="pgvector", error=str(e))

    milvus_uri = os.environ.get("OQA_MILVUS_URI")
    if milvus_uri:
        try:
            from openqueryagent.adapters.milvus import MilvusAdapter
            adapter = MilvusAdapter(uri=milvus_uri)
            await adapter.connect()
            adapters["milvus"] = adapter
            logger.info("mcp_adapter_connected", adapter="milvus")
        except Exception as e:
            logger.warning("mcp_adapter_failed", adapter="milvus", error=str(e))

    if not adapters:
        logger.warning("mcp_no_adapters", msg="No adapters configured via OQA_* env vars")

    config = AgentConfig(enable_tracing=False)
    agent = QueryAgent(adapters=adapters, config=config)
    await agent.initialize()
    _agent_instance = agent
    return agent


# ---------------------------------------------------------------------------
# MCP Server definition
# ---------------------------------------------------------------------------

def create_mcp_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("openqueryagent")

    # -- Tools ---------------------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="openqueryagent_ask",
                description=(
                    "Ask a natural language question about data stored in connected "
                    "vector databases. Returns an answer with citations."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The natural language question to answer.",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="openqueryagent_search",
                description=(
                    "Search for relevant documents across connected vector databases. "
                    "Returns a ranked list of matching documents."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default 10).",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="openqueryagent_aggregate",
                description=(
                    "Run an aggregation query (count, avg, sum, etc.) across "
                    "connected vector databases."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Aggregation query in natural language.",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
        arguments = arguments or {}
        agent = await _get_or_create_agent()

        try:
            if name == "openqueryagent_ask":
                query = arguments.get("query", "")
                response = await agent.ask(query)
                result = {
                    "answer": response.answer,
                    "citations": [c.model_dump() for c in (response.citations or [])],
                    "confidence": response.confidence,
                    "latency_ms": response.total_latency_ms,
                }
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "openqueryagent_search":
                query = arguments.get("query", "")
                limit = arguments.get("limit", 10)
                response = await agent.search(query, limit=limit)
                docs = []
                for rd in response.documents:
                    docs.append({
                        "id": rd.document.id,
                        "content": rd.document.content[:500] if rd.document.content else "",
                        "score": rd.score,
                        "collection": rd.document.collection,
                    })
                result = {"documents": docs, "total": len(docs), "latency_ms": response.total_latency_ms}
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "openqueryagent_aggregate":
                query = arguments.get("query", "")
                response = await agent.aggregate(query)
                result_data = response.result.model_dump() if response.result else {}
                result = {"result": result_data, "latency_ms": response.total_latency_ms}
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            else:
                return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            logger.error("mcp_tool_error", tool=name, error=str(e))
            return [types.TextContent(type="text", text=f"Error: {e}")]

    # -- Resources -----------------------------------------------------------

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        agent = await _get_or_create_agent()
        resources: list[types.Resource] = [
            types.Resource(
                uri="oqa://collections",
                name="Collections",
                description="List of all collections across connected databases.",
                mimeType="application/json",
            ),
        ]
        # Add per-collection schema resources
        try:
            collections = await agent.list_collections()
            for col in collections:
                resources.append(
                    types.Resource(
                        uri=f"oqa://collections/{col}/schema",
                        name=f"{col} Schema",
                        description=f"Schema for collection '{col}'.",
                        mimeType="application/json",
                    )
                )
        except Exception:
            pass

        return resources

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        agent = await _get_or_create_agent()
        uri_str = str(uri)

        if uri_str == "oqa://collections":
            try:
                collections = await agent.list_collections()
                return json.dumps({"collections": collections}, indent=2)
            except Exception as e:
                return json.dumps({"error": str(e)})

        if uri_str.startswith("oqa://collections/") and uri_str.endswith("/schema"):
            name = uri_str.replace("oqa://collections/", "").replace("/schema", "")
            try:
                schema = await agent.get_collection_schema(name)
                return json.dumps(schema.model_dump(), indent=2) if schema else json.dumps({"error": "not found"})
            except Exception as e:
                return json.dumps({"error": str(e)})

        return json.dumps({"error": f"Unknown resource: {uri_str}"})

    return server


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run the MCP server over stdio."""
    server = create_mcp_server()
    logger.info("mcp_server_starting", transport="stdio")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
