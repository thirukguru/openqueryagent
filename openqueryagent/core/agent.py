"""QueryAgent — the main orchestrator for OpenQueryAgent.

Wires together all pipeline components (planner, router, executor,
reranker, synthesizer) and exposes ``ask()``, ``search()``, and
``aggregate()`` endpoints.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from openqueryagent.core.config import AgentConfig
from openqueryagent.core.executor import QueryExecutor
from openqueryagent.core.memory import ConversationMemory
from openqueryagent.core.planner import LLMQueryPlanner, SimpleQueryPlanner
from openqueryagent.core.reranker import RRFReranker
from openqueryagent.core.router import QueryRouter
from openqueryagent.core.schema import SchemaInspector
from openqueryagent.core.synthesizer import LLMSynthesizer
from openqueryagent.core.types import (
    AggregationResponse,
    AskResponse,
    Document,
    ExecutionStatus,
    RankedDocument,
    SearchResponse,
    TokenUsage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from openqueryagent.core.types import AskResponseChunk

logger = structlog.get_logger(__name__)


class QueryAgent:
    """Main orchestrator for querying vector databases.

    Provides ``ask()``, ``search()``, and ``aggregate()`` methods
    that implement the full pipeline: plan → route → execute → rerank → synthesize.

    Args:
        adapters: Dict of adapter_id → adapter instance.
        llm: LLM provider for planning and synthesis.
        embedding: Optional embedding provider for query vectorization.
        config: Agent configuration.
        planner: Optional custom planner (defaults to LLMQueryPlanner or SimpleQueryPlanner).
        reranker: Optional custom reranker (defaults to RRFReranker).
    """

    def __init__(
        self,
        adapters: dict[str, Any],
        llm: Any | None = None,
        embedding: Any | None = None,
        config: AgentConfig | None = None,
        planner: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        self._config = config or AgentConfig()
        self._adapters = adapters
        self._llm = llm
        self._embedding = embedding

        # Schema inspector
        self._schema_inspector = SchemaInspector(adapters=list(adapters.values()))

        # Planner
        if planner:
            self._planner = planner
        elif llm:
            self._planner = LLMQueryPlanner(llm=llm)
        else:
            self._planner = SimpleQueryPlanner()

        # Executor
        self._executor = QueryExecutor(config=self._config.executor_config)

        # Reranker
        self._reranker = reranker or RRFReranker()

        # Synthesizer
        self._synthesizer = LLMSynthesizer(llm=llm) if llm else None

        # Memory
        self._memory = ConversationMemory(max_tokens=self._config.executor_config.max_aggregation_scroll)

    async def initialize(self) -> None:
        """Initialize the agent by refreshing schemas."""
        await self._schema_inspector.refresh()
        schema_map = await self._schema_inspector.get_schema_map()
        logger.info("agent_initialized", collections=len(schema_map.collections))

    async def ask(
        self,
        query: str,
        *,
        stream: bool = False,
    ) -> AskResponse | AsyncIterator[AskResponseChunk]:
        """Answer a natural language question.

        Full pipeline: plan → route → execute → rerank → synthesize.

        Args:
            query: Natural language question.
            stream: If True, return an async iterator of chunks.

        Returns:
            AskResponse with answer and citations, or async iterator if streaming.
        """
        start = time.monotonic()

        # Add query to memory
        self._memory.add_message("user", query)

        # Ensure schema is fresh
        schema_map = await self._schema_inspector.get_schema_map()

        # Plan
        plan = await self._planner.plan(
            query=query,
            schema_map=schema_map,
            history=self._memory.get_messages(),
        )

        # Route
        router = QueryRouter(adapters=self._adapters, schema_map=schema_map)
        routed = router.route(plan)

        # Embed query if needed
        query_vector = await self._embed_query(query)

        # Execute
        results = await self._executor.execute(routed, query_vector=query_vector)

        # Collect documents
        all_documents: list[Document] = []
        TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        for result in results:
            if result.status == ExecutionStatus.SUCCESS:
                all_documents.extend(result.documents)

        # Rerank
        ranked = await self._reranker.rerank(query, all_documents)

        # Synthesize
        if stream and self._synthesizer:
            return self._stream_synthesis(query, ranked, plan, start)

        if self._synthesizer and plan.requires_synthesis:
            docs_for_synthesis = [r.document for r in ranked[:10]]
            synthesis = await self._synthesizer.synthesize(
                query=query,
                documents=docs_for_synthesis,
                history=self._memory.get_messages(),
            )

            # Add answer to memory
            self._memory.add_message("assistant", synthesis.answer)

            total_latency = (time.monotonic() - start) * 1000
            return AskResponse(
                answer=synthesis.answer,
                citations=synthesis.citations,
                query_plan=plan,
                confidence=synthesis.confidence,
                total_latency_ms=total_latency,
                tokens_used=synthesis.tokens_used,
            )

        # No synthesis, return documents
        total_latency = (time.monotonic() - start) * 1000
        answer = self._format_documents_answer(ranked)
        self._memory.add_message("assistant", answer)
        return AskResponse(
            answer=answer,
            query_plan=plan,
            total_latency_ms=total_latency,
        )

    async def _stream_synthesis(
        self,
        query: str,
        ranked: list[RankedDocument],
        plan: Any,
        start: float,
    ) -> AsyncIterator[AskResponseChunk]:
        """Stream the synthesis part of ask()."""
        from openqueryagent.core.types import AskResponseChunk

        docs = [r.document for r in ranked[:10]]

        yield AskResponseChunk(text="", stage="planning", query_plan=plan)
        yield AskResponseChunk(text="", stage="searching")

        assert self._synthesizer is not None
        full_answer = ""
        async for chunk in self._synthesizer.synthesize_stream(
            query=query, documents=docs, history=self._memory.get_messages(),
        ):
            full_answer += chunk.text
            yield chunk

        self._memory.add_message("assistant", full_answer)

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> SearchResponse:
        """Search without answer synthesis.

        Pipeline: plan → route → execute → rerank.

        Args:
            query: Natural language search query.
            limit: Maximum results.

        Returns:
            SearchResponse with ranked documents.
        """
        start = time.monotonic()

        schema_map = await self._schema_inspector.get_schema_map()

        plan = await self._planner.plan(query=query, schema_map=schema_map)

        router = QueryRouter(adapters=self._adapters, schema_map=schema_map)
        routed = router.route(plan)

        query_vector = await self._embed_query(query)
        results = await self._executor.execute(routed, query_vector=query_vector)

        all_documents: list[Document] = []
        for result in results:
            if result.status == ExecutionStatus.SUCCESS:
                all_documents.extend(result.documents)

        ranked = await self._reranker.rerank(query, all_documents)

        total_latency = (time.monotonic() - start) * 1000
        return SearchResponse(
            documents=ranked[:limit],
            query_plan=plan,
            total_latency_ms=total_latency,
        )

    async def aggregate(
        self,
        query: str,
    ) -> AggregationResponse:
        """Execute an aggregation query.

        Pipeline: plan → route → execute.

        Args:
            query: Natural language aggregation query.

        Returns:
            AggregationResponse with result.
        """
        start = time.monotonic()

        schema_map = await self._schema_inspector.get_schema_map()

        plan = await self._planner.plan(query=query, schema_map=schema_map)

        router = QueryRouter(adapters=self._adapters, schema_map=schema_map)
        routed = router.route(plan)

        results = await self._executor.execute(routed)

        total_latency = (time.monotonic() - start) * 1000
        for result in results:
            if result.status == ExecutionStatus.SUCCESS and result.aggregation_result:
                return AggregationResponse(
                    result=result.aggregation_result,
                    query_plan=plan,
                    total_latency_ms=total_latency,
                )

        return AggregationResponse(
            query_plan=plan,
            total_latency_ms=total_latency,
        )

    @property
    def memory(self) -> ConversationMemory:
        """Access conversation memory."""
        return self._memory

    @property
    def schema_inspector(self) -> SchemaInspector:
        """Access schema inspector."""
        return self._schema_inspector

    async def _embed_query(self, text: str) -> list[float] | None:
        """Embed a query if embedding provider is available."""
        if self._embedding:
            try:
                result: list[float] = await self._embedding.embed_query(text)
                return result
            except Exception as e:
                logger.warning("embedding_error", error=str(e))
        return None

    @staticmethod
    def _format_documents_answer(ranked: list[RankedDocument]) -> str:
        """Format ranked documents as a simple answer when no LLM is available."""
        if not ranked:
            return "No results found."
        lines: list[str] = []
        for i, rd in enumerate(ranked[:5], 1):
            content = rd.document.content or str(rd.document.properties)
            lines.append(f"{i}. {content[:200]}")
        return "\n".join(lines)
