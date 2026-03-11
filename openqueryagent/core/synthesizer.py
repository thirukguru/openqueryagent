"""Answer synthesizer — generates natural language answers from retrieved context.

Uses an LLM to synthesize answers from search results, with citation
extraction from ``[N]`` notation in the response.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import structlog

from openqueryagent.core.types import (
    ChatMessage,
    Citation,
    Document,
    SynthesisResult,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from openqueryagent.core.types import AskResponseChunk
    from openqueryagent.llm.base import LLMResponse

logger = structlog.get_logger(__name__)

# Regex to find citation markers like [1], [2], etc.
_CITATION_RE = re.compile(r"\[(\d+)\]")

_SYNTHESIZER_SYSTEM_PROMPT = """You are an AI assistant that answers questions based on the provided context.

Rules:
- Base your answer ONLY on the provided context documents
- Cite sources using [N] notation where N is the document number
- If the context doesn't contain enough information, say so honestly
- Be concise and direct
- If the question asks for specific data, include the relevant numbers/facts
"""


class LLMSynthesizer:
    """Synthesizes natural language answers from search results using an LLM.

    Formats retrieved documents as numbered context, builds a synthesis
    prompt, calls the LLM, and extracts citations from [N] markers.

    Args:
        llm: LLM provider for answer generation.
        max_context_docs: Maximum number of documents to include in context.
    """

    def __init__(self, llm: Any, max_context_docs: int = 10) -> None:
        self._llm = llm
        self._max_context_docs = max_context_docs

    async def synthesize(
        self,
        query: str,
        documents: list[Document],
        history: list[ChatMessage] | None = None,
    ) -> SynthesisResult:
        """Generate a synthesized answer from documents.

        Args:
            query: The user's original query.
            documents: Retrieved documents for context.
            history: Optional conversation history.

        Returns:
            SynthesisResult with answer, citations, and usage.
        """
        if not documents:
            return SynthesisResult(
                answer="I couldn't find any relevant documents to answer your question.",
                confidence=0.0,
            )

        # Format context
        context = self._format_context(documents[:self._max_context_docs])

        # Build messages
        messages = [
            ChatMessage(role="system", content=_SYNTHESIZER_SYSTEM_PROMPT),
        ]

        if history:
            messages.extend(history[-4:])  # Last 2 exchanges

        user_content = f"""Context Documents:
{context}

Question: {query}

Provide a comprehensive answer based on the context above. Cite your sources using [N] notation."""

        messages.append(ChatMessage(role="user", content=user_content))

        # Call LLM
        response: LLMResponse = await self._llm.complete(
            messages=messages,
            temperature=0.3,
        )

        # Extract citations
        citations = self._extract_citations(response.content, documents[:self._max_context_docs])

        return SynthesisResult(
            answer=response.content,
            citations=citations,
            confidence=min(1.0, len(citations) * 0.2 + 0.3) if citations else 0.3,
            model_used=response.model,
            tokens_used=response.usage,
        )

    async def synthesize_stream(
        self,
        query: str,
        documents: list[Document],
        history: list[ChatMessage] | None = None,
    ) -> AsyncIterator[AskResponseChunk]:
        """Generate a streaming synthesized answer.

        Args:
            query: The user's original query.
            documents: Retrieved documents.
            history: Optional conversation history.

        Yields:
            AskResponseChunk objects with incremental content.
        """
        from openqueryagent.core.types import AskResponseChunk

        if not documents:
            yield AskResponseChunk(
                text="I couldn't find any relevant documents to answer your question.",
                stage="synthesizing",
                is_final=True,
            )
            return

        context = self._format_context(documents[:self._max_context_docs])
        messages = [
            ChatMessage(role="system", content=_SYNTHESIZER_SYSTEM_PROMPT),
        ]
        if history:
            messages.extend(history[-4:])

        user_content = f"""Context Documents:
{context}

Question: {query}

Provide a comprehensive answer based on the context above. Cite your sources using [N] notation."""

        messages.append(ChatMessage(role="user", content=user_content))

        full_content = ""
        async for chunk in self._llm.complete_stream(messages=messages, temperature=0.3):
            full_content += chunk.content
            yield AskResponseChunk(
                text=chunk.content,
                stage="synthesizing",
            )

        # Final chunk with citations
        citations = self._extract_citations(full_content, documents[:self._max_context_docs])
        yield AskResponseChunk(
            text="",
            stage="synthesizing",
            is_final=True,
            citations=citations,
        )

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        """Format documents as numbered context for the LLM."""
        lines: list[str] = []
        for i, doc in enumerate(documents, 1):
            content = doc.content or ""
            if not content and doc.properties:
                content = str(doc.properties)
            meta_parts: list[str] = []
            if doc.collection:
                meta_parts.append(f"collection: {doc.collection}")
            if doc.score is not None:
                meta_parts.append(f"score: {doc.score:.3f}")
            meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
            lines.append(f"[{i}]{meta}: {content}")
        return "\n\n".join(lines)

    @staticmethod
    def _extract_citations(
        answer: str,
        documents: list[Document],
    ) -> list[Citation]:
        """Extract citation markers [N] from the answer text."""
        cited_indices: set[int] = set()
        for match in _CITATION_RE.finditer(answer):
            idx = int(match.group(1))
            if 1 <= idx <= len(documents):
                cited_indices.add(idx)

        citations: list[Citation] = []
        for idx in sorted(cited_indices):
            doc = documents[idx - 1]
            snippet = (doc.content or "")[:200]
            citations.append(Citation(
                document_id=doc.id,
                collection=doc.collection,
                text_snippet=snippet,
                relevance_score=doc.score or 0.0,
            ))

        return citations
