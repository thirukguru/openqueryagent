"""Tests for QueryExecutor."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from openqueryagent.core.config import ExecutorConfig
from openqueryagent.core.executor import QueryExecutor
from openqueryagent.core.types import (
    Document,
    ExecutionStatus,
    SearchResult,
    SearchType,
    SubQuery,
)


def _make_routed_query(
    sub_query_id: str = "q1",
    collection: str = "products",
    depends_on: list[str] | None = None,
) -> dict:
    adapter = AsyncMock()
    adapter.search.return_value = SearchResult(
        documents=[Document(id="doc-1", content="test doc")],
        total_count=1,
    )
    return {
        "sub_query": SubQuery(
            id=sub_query_id,
            collection=collection,
            query_text="test",
            search_type=SearchType.HYBRID,
            depends_on=depends_on,
        ),
        "collection": collection,
        "adapter": adapter,
        "filters": None,
    }


class TestQueryExecutor:
    @pytest.mark.asyncio
    async def test_single_query_success(self) -> None:
        executor = QueryExecutor()
        rq = _make_routed_query()



        results = await executor.execute([rq])
        assert len(results) == 1
        assert results[0].status == ExecutionStatus.SUCCESS
        assert len(results[0].documents) == 1

    @pytest.mark.asyncio
    async def test_parallel_execution(self) -> None:
        executor = QueryExecutor()
        rq1 = _make_routed_query("q1")
        rq2 = _make_routed_query("q2", "reviews")

        results = await executor.execute([rq1, rq2])
        assert len(results) == 2
        assert all(r.status == ExecutionStatus.SUCCESS for r in results)

    @pytest.mark.asyncio
    async def test_timeout_handling(self) -> None:
        adapter = AsyncMock()

        async def slow_search(**kwargs):
            await asyncio.sleep(10)
            return SearchResult(documents=[])

        adapter.search = slow_search

        executor = QueryExecutor(config=ExecutorConfig(timeout_per_query=0.1))
        rq = {
            "sub_query": SubQuery(id="q1", collection="products", query_text="test"),
            "collection": "products",
            "adapter": adapter,
            "filters": None,
        }

        results = await executor.execute([rq])
        assert len(results) == 1
        assert results[0].status == ExecutionStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        adapter = AsyncMock()
        adapter.search.side_effect = RuntimeError("Search failed")

        executor = QueryExecutor()
        rq = {
            "sub_query": SubQuery(id="q1", collection="products", query_text="test"),
            "collection": "products",
            "adapter": adapter,
            "filters": None,
        }

        results = await executor.execute([rq])
        assert len(results) == 1
        assert results[0].status == ExecutionStatus.ERROR
        assert "Search failed" in (results[0].error or "")

    @pytest.mark.asyncio
    async def test_empty_queries(self) -> None:
        executor = QueryExecutor()
        results = await executor.execute([])
        assert results == []

    @pytest.mark.asyncio
    async def test_dependency_ordering(self) -> None:
        executor = QueryExecutor()
        rq1 = _make_routed_query("q1")
        rq2 = _make_routed_query("q2", depends_on=["q1"])

        results = await executor.execute([rq1, rq2])
        assert len(results) == 2
        assert all(r.status == ExecutionStatus.SUCCESS for r in results)
