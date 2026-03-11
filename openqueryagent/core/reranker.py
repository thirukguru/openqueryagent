"""Reranker implementations for result ordering.

Provides reranking strategies to reorder search results:
- ``NoopReranker``: Pass-through, no reranking
- ``RRFReranker``: Reciprocal Rank Fusion for multi-source results
"""

from __future__ import annotations

from openqueryagent.core.types import Document, RankedDocument


class NoopReranker:
    """Pass-through reranker that preserves original ordering.

    Wraps documents in RankedDocument with original scores preserved.
    """

    async def rerank(
        self,
        query: str,
        documents: list[Document],
    ) -> list[RankedDocument]:
        """Return documents in original order as RankedDocuments.

        Args:
            query: The original user query (unused).
            documents: Documents to rerank.

        Returns:
            List of RankedDocument preserving original order.
        """
        return [
            RankedDocument(
                document=doc,
                score=doc.score or 0.0,
                original_rank=i,
                new_rank=i,
            )
            for i, doc in enumerate(documents)
        ]


class RRFReranker:
    """Reciprocal Rank Fusion reranker for multi-source results.

    Combines results from multiple sources using RRF scoring:
    ``score(d) = sum(1 / (k + rank_i(d)))`` for each source.

    Args:
        k: RRF constant (higher = more weight to lower-ranked items).
    """

    def __init__(self, k: int = 60) -> None:
        self._k = k

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        source_groups: list[list[Document]] | None = None,
    ) -> list[RankedDocument]:
        """Rerank documents using Reciprocal Rank Fusion.

        If ``source_groups`` is provided, RRF is computed across groups.
        Otherwise, documents are treated as a single source and scored
        by their position.

        Args:
            query: The original user query (for future use).
            documents: All documents to rerank.
            source_groups: Optional grouping of documents by source.

        Returns:
            List of RankedDocument sorted by RRF score.
        """
        if source_groups and len(source_groups) > 1:
            return self._rrf_multi_source(documents, source_groups)

        # Single source: just score by position
        rrf_scores: dict[str, float] = {}
        original_ranks: dict[str, int] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(documents):
            doc_id = doc.id
            rrf_scores[doc_id] = 1.0 / (self._k + rank)
            original_ranks[doc_id] = rank
            doc_map[doc_id] = doc

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        return [
            RankedDocument(
                document=doc_map[doc_id],
                score=rrf_scores[doc_id],
                original_rank=original_ranks[doc_id],
                new_rank=new_rank,
            )
            for new_rank, doc_id in enumerate(sorted_ids)
        ]

    def _rrf_multi_source(
        self,
        documents: list[Document],
        source_groups: list[list[Document]],
    ) -> list[RankedDocument]:
        """Compute RRF across multiple source groups."""
        rrf_scores: dict[str, float] = {}
        original_ranks: dict[str, int] = {}
        doc_map: dict[str, Document] = {}

        # Score from each source
        for group in source_groups:
            for rank, doc in enumerate(group):
                doc_id = doc.id
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                    original_ranks[doc_id] = rank
                    doc_map[doc_id] = doc
                rrf_scores[doc_id] += 1.0 / (self._k + rank)

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        return [
            RankedDocument(
                document=doc_map[doc_id],
                score=rrf_scores[doc_id],
                original_rank=original_ranks[doc_id],
                new_rank=new_rank,
            )
            for new_rank, doc_id in enumerate(sorted_ids)
        ]
