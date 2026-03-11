"""OpenAI Embedding provider.

Implements the EmbeddingProvider protocol using the OpenAI Embeddings API.
Supports ``text-embedding-3-small``, ``text-embedding-3-large``,
and ``text-embedding-ada-002``.

Requires ``openai`` (install with ``pip install openqueryagent[openai]``).
"""

from __future__ import annotations

from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

# Known model → dimension mapping
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Maximum batch size for OpenAI embeddings API
_MAX_BATCH_SIZE = 2048


class OpenAIEmbedding:
    """OpenAI embedding provider.

    Args:
        model: Embedding model to use.
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        api_base: Custom API base URL (for Azure or local proxies).
        dimensions: Override vector dimensions (only for v3 models).
        batch_size: Maximum number of texts per API call.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        api_base: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 512,
    ) -> None:
        self._model = model
        self._batch_size = min(batch_size, _MAX_BATCH_SIZE)
        self._client: Any = None

        # Resolve dimensions
        if dimensions:
            self._dimensions = dimensions
        else:
            self._dimensions = _MODEL_DIMENSIONS.get(model, 1536)

        self._init_client(api_key, api_base)

    def _init_client(self, api_key: str | None, api_base: str | None) -> None:
        """Initialize the OpenAI client."""
        try:
            import openai
        except ImportError as e:
            msg = "openai is not installed. Install with: pip install openqueryagent[openai]"
            raise ImportError(msg) from e

        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    @property
    def dimensions(self) -> int:
        """Vector dimensionality of the embedding model."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Model identifier."""
        return self._model

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: The query text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": text,
        }

        # Only v3 models support custom dimensions
        if self._model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self._dimensions

        response = await self._client.embeddings.create(**kwargs)
        embedding: list[float] = response.data[0].embedding
        return embedding

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents.

        Splits into batches of ``batch_size`` for API limits.

        Args:
            texts: List of document texts.

        Returns:
            List of embedding vectors, one per input text.
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch of texts."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": texts,
        }

        if self._model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self._dimensions

        response = await self._client.embeddings.create(**kwargs)

        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
