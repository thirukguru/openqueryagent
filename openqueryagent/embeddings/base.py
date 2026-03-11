"""Base protocol for embedding providers.

Embedding providers generate vector representations of text for use
with vector stores that require pre-computed embeddings (e.g., pgvector,
Qdrant without server-side embedding).

Adapters with built-in embedding (Weaviate, Pinecone) may not need
an external embedding provider.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface for embedding model providers.

    Implementations: OpenAIEmbedding (Phase 1), CohereEmbedding,
    HuggingFaceEmbedding, BedrockEmbedding (Phase 2).
    """

    @property
    def dimensions(self) -> int:
        """The dimensionality of the embedding vectors."""
        ...

    @property
    def model_name(self) -> str:
        """The model identifier (e.g., 'text-embedding-3-small')."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Generate an embedding vector for a single query text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            RateLimitError: If the provider rate-limits the request.
            OpenQueryAgentError: If embedding fails.
        """
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            A list of embedding vectors, one per input text.

        Raises:
            RateLimitError: If the provider rate-limits the request.
            OpenQueryAgentError: If embedding fails.
        """
        ...
