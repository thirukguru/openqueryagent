"""HuggingFace embedding provider.

Implements the EmbeddingProvider protocol using local HuggingFace
``sentence-transformers`` models. Runs entirely locally — no API calls.

Requires ``sentence-transformers`` (install with ``pip install openqueryagent[huggingface]``).
"""

from __future__ import annotations

from typing import Any

import structlog

from openqueryagent.core.exceptions import OpenQueryAgentError

logger = structlog.get_logger(__name__)


class HuggingFaceEmbedding:
    """HuggingFace local embedding provider.

    Args:
        model: Model name (e.g., ``all-MiniLM-L6-v2``).
        device: Torch device (``cpu``, ``cuda``, ``mps``).
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self._model_name = model
        self._device = device
        self._model: Any = None
        self._dimensions = 0
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as e:
            msg = "sentence-transformers is not installed. Install with: pip install openqueryagent[huggingface]"
            raise ImportError(msg) from e
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._dimensions = self._model.get_sentence_embedding_dimension() or 384

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text (runs synchronously in thread)."""
        try:
            import asyncio
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, self._encode_single, text
            )
            return embedding
        except Exception as e:
            raise OpenQueryAgentError(f"HuggingFace embedding failed: {e}") from e

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (runs synchronously in thread)."""
        try:
            import asyncio
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, self._encode_batch, texts
            )
            return embeddings
        except Exception as e:
            raise OpenQueryAgentError(f"HuggingFace embedding failed: {e}") from e

    def _encode_single(self, text: str) -> list[float]:
        """Encode a single text."""
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()  # type: ignore[no-any-return]

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts."""
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()  # type: ignore[no-any-return]
