"""Cohere embedding provider.

Implements the EmbeddingProvider protocol using the Cohere API.
Requires ``cohere`` (install with ``pip install openqueryagent[cohere]``).
"""

from __future__ import annotations

from typing import Any

import structlog

from openqueryagent.core.exceptions import OpenQueryAgentError

logger = structlog.get_logger(__name__)


class CohereEmbedding:
    """Cohere embedding provider.

    Args:
        model: Model name (e.g., ``embed-english-v3.0``).
        api_key: Cohere API key. Falls back to ``CO_API_KEY`` env var.
        input_type: Input type for the embedding model.
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: str | None = None,
        input_type: str = "search_query",
    ) -> None:
        self._model = model
        self._input_type = input_type
        self._dimensions = 1024  # Default for embed-english-v3.0
        self._client: Any = None
        self._init_client(api_key)

    def _init_client(self, api_key: str | None) -> None:
        """Initialize the Cohere client."""
        try:
            import cohere
        except ImportError as e:
            msg = "cohere is not installed. Install with: pip install openqueryagent[cohere]"
            raise ImportError(msg) from e
        self._client = cohere.AsyncClientV2(api_key=api_key)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        try:
            response = await self._client.embed(
                texts=[text],
                model=self._model,
                input_type=self._input_type,
                embedding_types=["float"],
            )
            embeddings = response.embeddings
            # V2 response: embeddings.float_ is a list of lists
            float_embeddings: list[list[float]] = getattr(embeddings, "float_", [[]])
            return float_embeddings[0] if float_embeddings else []
        except Exception as e:
            raise OpenQueryAgentError(f"Cohere embedding failed: {e}") from e

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple document texts."""
        try:
            response = await self._client.embed(
                texts=texts,
                model=self._model,
                input_type="search_document",
                embedding_types=["float"],
            )
            float_embeddings: list[list[float]] = getattr(response.embeddings, "float_", [])
            return float_embeddings
        except Exception as e:
            raise OpenQueryAgentError(f"Cohere embedding failed: {e}") from e
