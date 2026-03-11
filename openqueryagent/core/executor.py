"""Query executor — runs sub-queries in parallel with dependency ordering.

Supports parallel execution via asyncio, dependency ordering via topological
sort, per-query timeouts, retry with backoff, and partial result handling.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from openqueryagent.core.config import ExecutorConfig
from openqueryagent.core.types import (
    AggregationResult,
    ExecutionResult,
    ExecutionStatus,
    SearchResult,
    SubQuery,
)
from openqueryagent.observability.metrics import get_metrics
from openqueryagent.observability.tracing import get_tracing

logger = structlog.get_logger(__name__)


class QueryExecutor:
    """Executes sub-queries against vector store adapters.

    Supports parallel execution with configurable concurrency,
    dependency ordering, per-query timeouts, circuit breakers,
    and graceful error handling.

    Args:
        config: Executor configuration.
    """

    def __init__(self, config: ExecutorConfig | None = None) -> None:
        self._config = config or ExecutorConfig()
        # Import here to avoid circular import at module level
        from openqueryagent.core.circuit_breaker import CircuitBreakerRegistry
        self._circuit_breakers = CircuitBreakerRegistry()

    async def execute(
        self,
        routed_queries: list[dict[str, Any]],
        query_vector: list[float] | None = None,
    ) -> list[ExecutionResult]:
        """Execute routed sub-queries.

        Queries without dependencies run in parallel (bounded by semaphore).
        Queries with dependencies wait for their prerequisites.

        Args:
            routed_queries: List of routing info dicts from QueryRouter.
            query_vector: Optional pre-computed query embedding.

        Returns:
            List of ExecutionResult for each sub-query.
        """
        if not routed_queries:
            return []

        # Build dependency graph
        query_map = {rq["sub_query"].id: rq for rq in routed_queries}
        results: dict[str, ExecutionResult] = {}
        semaphore = asyncio.Semaphore(self._config.max_concurrent)

        # Topological sort for dependency ordering
        execution_order = self._topological_sort(routed_queries)

        for batch in execution_order:
            tasks = [
                self._execute_single(
                    query_map[qid], semaphore, results, query_vector,
                )
                for qid in batch
                if qid in query_map
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, qid in enumerate(batch):
                if qid not in query_map:
                    continue
                result = batch_results[i] if i < len(batch_results) else None
                if isinstance(result, ExecutionResult):
                    results[qid] = result
                elif isinstance(result, BaseException):
                    results[qid] = ExecutionResult(
                        sub_query_id=qid,
                        status=ExecutionStatus.ERROR,
                        error=str(result),
                    )

        # Return in original order
        return [
            results.get(rq["sub_query"].id, ExecutionResult(
                sub_query_id=rq["sub_query"].id,
                status=ExecutionStatus.ERROR,
                error="Query not executed",
            ))
            for rq in routed_queries
        ]

    async def _execute_single(
        self,
        routed_query: dict[str, Any],
        semaphore: asyncio.Semaphore,
        completed: dict[str, ExecutionResult],
        query_vector: list[float] | None,
    ) -> ExecutionResult:
        """Execute a single sub-query with timeout and error handling."""
        sq: SubQuery = routed_query["sub_query"]
        adapter = routed_query["adapter"]
        collection = routed_query["collection"]
        filters = routed_query.get("filters")

        import time

        start = time.monotonic()
        adapter_id = getattr(adapter, "adapter_id", "unknown")
        tracing = get_tracing()
        metrics = get_metrics()

        async with semaphore:
            with tracing.span(
                f"oqa.execute.{adapter_id}",
                {"oqa.adapter": adapter_id, "oqa.collection": collection,
                 "oqa.search_type": str(sq.search_type)},
            ):
                # Circuit breaker gate
                breaker = self._circuit_breakers.get(adapter_id)
                try:
                    breaker.pre_call()
                except Exception as e:
                    latency = (time.monotonic() - start) * 1000
                    metrics.observe_adapter_query(adapter_id, latency / 1000, status="circuit_open")
                    return ExecutionResult(
                        sub_query_id=sq.id,
                        status=ExecutionStatus.ERROR,
                        latency_ms=latency,
                        error=str(e),
                    )

                try:
                    result = await asyncio.wait_for(
                        self._run_query(
                            adapter=adapter,
                            sub_query=sq,
                            collection=collection,
                            filters=filters,
                            query_vector=query_vector,
                        ),
                        timeout=self._config.timeout_per_query,
                    )
                    latency = (time.monotonic() - start) * 1000
                    result.latency_ms = latency
                    metrics.observe_adapter_query(adapter_id, latency / 1000)
                    breaker.on_success()
                    return result

                except TimeoutError:
                    latency = (time.monotonic() - start) * 1000
                    logger.warning("query_timeout", sub_query_id=sq.id, latency_ms=latency)
                    metrics.observe_adapter_query(adapter_id, latency / 1000, status="timeout")
                    breaker.on_failure()
                    return ExecutionResult(
                        sub_query_id=sq.id,
                        status=ExecutionStatus.TIMEOUT,
                        latency_ms=latency,
                        error=f"Query timed out after {self._config.timeout_per_query}s",
                    )
                except Exception as e:
                    latency = (time.monotonic() - start) * 1000
                    logger.error("query_error", sub_query_id=sq.id, error=str(e))
                    metrics.observe_adapter_query(adapter_id, latency / 1000, status="error")
                    breaker.on_failure()
                    return ExecutionResult(
                        sub_query_id=sq.id,
                        status=ExecutionStatus.ERROR,
                        latency_ms=latency,
                        error=str(e),
                    )

    async def _run_query(
        self,
        adapter: Any,
        sub_query: SubQuery,
        collection: str,
        filters: Any | None,
        query_vector: list[float] | None,
    ) -> ExecutionResult:
        """Run a search or aggregation query against an adapter."""
        if sub_query.aggregation:
            # Aggregation query
            agg_result: AggregationResult = await adapter.aggregate(
                collection=collection,
                aggregation=sub_query.aggregation,
                filters=filters,
            )
            return ExecutionResult(
                sub_query_id=sub_query.id,
                status=ExecutionStatus.SUCCESS,
                aggregation_result=agg_result,
            )

        # Search query
        search_result: SearchResult = await adapter.search(
            collection=collection,
            query_vector=query_vector,
            query_text=sub_query.query_text,
            filters=filters,
            limit=sub_query.limit,
            search_type=sub_query.search_type,
        )

        return ExecutionResult(
            sub_query_id=sub_query.id,
            status=ExecutionStatus.SUCCESS,
            documents=search_result.documents,
        )

    @staticmethod
    def _topological_sort(
        routed_queries: list[dict[str, Any]],
    ) -> list[list[str]]:
        """Sort queries by dependency order, returning batches.

        Queries in the same batch can run in parallel.

        Returns:
            List of batches, each batch is a list of sub-query IDs.
        """
        # Build adjacency
        all_ids = {rq["sub_query"].id for rq in routed_queries}
        deps: dict[str, list[str]] = {}
        for rq in routed_queries:
            sq: SubQuery = rq["sub_query"]
            sq_deps = [d for d in (sq.depends_on or []) if d in all_ids]
            deps[sq.id] = sq_deps

        # Kahn's algorithm
        in_degree: dict[str, int] = {qid: 0 for qid in all_ids}
        for qid, dep_list in deps.items():
            in_degree[qid] = len(dep_list)

        batches: list[list[str]] = []
        remaining = set(all_ids)

        while remaining:
            # Find all nodes with in_degree 0
            batch = [qid for qid in remaining if in_degree.get(qid, 0) == 0]
            if not batch:
                # Circular dependency — just run remaining
                batches.append(list(remaining))
                break

            batches.append(batch)
            for qid in batch:
                remaining.discard(qid)
                # Reduce in_degree of dependents
                for other_id in remaining:
                    if qid in deps.get(other_id, []):
                        in_degree[other_id] = max(0, in_degree[other_id] - 1)

        return batches
