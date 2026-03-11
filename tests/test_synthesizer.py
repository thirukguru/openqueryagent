"""Tests for LLMSynthesizer."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from openqueryagent.core.synthesizer import LLMSynthesizer
from openqueryagent.core.types import Document, TokenUsage
from openqueryagent.llm.base import LLMResponse


class TestLLMSynthesizer:
    @pytest.mark.asyncio
    async def test_synthesize_with_citations(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = LLMResponse(
            content="Based on the results, the best product is Widget A [1] which has great reviews [2].",
            model="gpt-4o-mini",
            usage=TokenUsage(prompt_tokens=200, completion_tokens=50, total_tokens=250),
        )

        synth = LLMSynthesizer(llm=mock_llm)
        docs = [
            Document(id="d1", content="Widget A is excellent", collection="products", score=0.95),
            Document(id="d2", content="Great product reviews", collection="reviews", score=0.9),
        ]

        result = await synth.synthesize("best product", docs)

        assert "Widget A" in result.answer
        assert len(result.citations) == 2
        assert result.citations[0].document_id == "d1"
        assert result.citations[1].document_id == "d2"
        assert result.model_used == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_synthesize_no_documents(self) -> None:
        mock_llm = AsyncMock()
        synth = LLMSynthesizer(llm=mock_llm)

        result = await synth.synthesize("test query", [])

        assert "couldn't find" in result.answer.lower()
        assert result.confidence == 0.0
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesize_no_citations(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = LLMResponse(
            content="I'm not sure about that topic.",
            model="gpt-4o-mini",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        )

        synth = LLMSynthesizer(llm=mock_llm)
        docs = [Document(id="d1", content="Some content", score=0.5)]
        result = await synth.synthesize("unknown topic", docs)

        assert len(result.citations) == 0
        assert result.confidence == 0.3  # Base confidence with no citations

    def test_format_context(self) -> None:
        docs = [
            Document(id="d1", content="First doc", collection="col1", score=0.9),
            Document(id="d2", content="Second doc", collection="col2", score=0.8),
        ]

        context = LLMSynthesizer._format_context(docs)

        assert "[1]" in context
        assert "[2]" in context
        assert "First doc" in context
        assert "Second doc" in context
        assert "col1" in context

    def test_extract_citations(self) -> None:
        docs = [
            Document(id="d1", content="Doc one content", collection="col1", score=0.9),
            Document(id="d2", content="Doc two content", collection="col2", score=0.8),
            Document(id="d3", content="Doc three content", collection="col1", score=0.7),
        ]

        citations = LLMSynthesizer._extract_citations(
            "The answer is [1] and also [3].", docs
        )

        assert len(citations) == 2
        assert citations[0].document_id == "d1"
        assert citations[1].document_id == "d3"

    def test_extract_citations_out_of_range(self) -> None:
        docs = [Document(id="d1", content="Doc", score=0.9)]
        citations = LLMSynthesizer._extract_citations("[1] and [5]", docs)
        assert len(citations) == 1  # [5] is out of range
