"""Tests for reranker implementations — NoopReranker and RRFReranker."""

from __future__ import annotations

import pytest

from openqueryagent.core.reranker import NoopReranker, RRFReranker
from openqueryagent.core.types import Document


def _make_docs(n: int) -> list[Document]:
    return [
        Document(id=f"doc-{i}", content=f"Document {i}", score=1.0 - i * 0.1)
        for i in range(n)
    ]


class TestNoopReranker:
    @pytest.mark.asyncio
    async def test_preserves_order(self) -> None:
        docs = _make_docs(3)
        reranker = NoopReranker()
        ranked = await reranker.rerank("test query", docs)

        assert len(ranked) == 3
        assert ranked[0].document.id == "doc-0"
        assert ranked[1].document.id == "doc-1"
        assert ranked[2].document.id == "doc-2"

    @pytest.mark.asyncio
    async def test_preserves_scores(self) -> None:
        docs = _make_docs(2)
        reranker = NoopReranker()
        ranked = await reranker.rerank("test", docs)

        assert ranked[0].score == docs[0].score
        assert ranked[0].original_rank == 0
        assert ranked[0].new_rank == 0

    @pytest.mark.asyncio
    async def test_empty_docs(self) -> None:
        reranker = NoopReranker()
        ranked = await reranker.rerank("test", [])
        assert ranked == []


class TestRRFReranker:
    @pytest.mark.asyncio
    async def test_single_source(self) -> None:
        docs = _make_docs(3)
        reranker = RRFReranker(k=60)
        ranked = await reranker.rerank("test", docs)

        assert len(ranked) == 3
        # First doc should still be first (highest RRF score)
        assert ranked[0].document.id == "doc-0"
        # Scores should decrease
        assert ranked[0].score > ranked[1].score

    @pytest.mark.asyncio
    async def test_multi_source_fusion(self) -> None:
        source1 = [
            Document(id="a", content="A", score=0.9),
            Document(id="b", content="B", score=0.8),
        ]
        source2 = [
            Document(id="b", content="B", score=0.95),
            Document(id="c", content="C", score=0.85),
        ]
        all_docs = source1 + source2
        reranker = RRFReranker(k=60)
        ranked = await reranker.rerank("test", all_docs, source_groups=[source1, source2])

        assert len(ranked) == 3
        # "b" appears in both sources → should rank higher
        assert ranked[0].document.id == "b"

    @pytest.mark.asyncio
    async def test_empty_docs(self) -> None:
        reranker = RRFReranker()
        ranked = await reranker.rerank("test", [])
        assert ranked == []
