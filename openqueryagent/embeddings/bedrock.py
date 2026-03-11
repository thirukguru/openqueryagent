"""AWS Bedrock embedding provider.

Implements the EmbeddingProvider protocol using AWS Bedrock Runtime.
Supports Titan Embedding and Cohere Embed models.

Requires ``aiobotocore`` (install with ``pip install openqueryagent[bedrock]``).
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from openqueryagent.core.exceptions import OpenQueryAgentError

logger = structlog.get_logger(__name__)


class BedrockEmbedding:
    """AWS Bedrock embedding provider.

    Args:
        model: Bedrock model ID (e.g., ``amazon.titan-embed-text-v2:0``).
        region: AWS region.
        dimensions: Embedding dimensions for the model.
        aws_access_key_id: Optional AWS access key.
        aws_secret_access_key: Optional AWS secret key.
    """

    def __init__(
        self,
        model: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        dimensions: int = 1024,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        self._model = model
        self._region = region
        self._dimensions = dimensions
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._session: Any = None
        self._init_session()

    def _init_session(self) -> None:
        """Initialize aiobotocore session."""
        try:
            from aiobotocore.session import AioSession
        except ImportError as e:
            msg = "aiobotocore is not installed. Install with: pip install openqueryagent[bedrock]"
            raise ImportError(msg) from e
        self._session = AioSession()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client_kwargs(self) -> dict[str, Any]:
        """Build kwargs for creating the Bedrock client."""
        kwargs: dict[str, Any] = {"region_name": self._region}
        if self._aws_access_key_id and self._aws_secret_access_key:
            kwargs["aws_access_key_id"] = self._aws_access_key_id
            kwargs["aws_secret_access_key"] = self._aws_secret_access_key
        return kwargs

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text via Bedrock."""
        results = await self._invoke_model([text])
        return results[0] if results else []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts via Bedrock (one at a time)."""
        return await self._invoke_model(texts)

    async def _invoke_model(self, texts: list[str]) -> list[list[float]]:
        """Invoke the Bedrock model for embedding."""
        embeddings: list[list[float]] = []
        try:
            async with self._session.create_client(
                "bedrock-runtime", **self._get_client_kwargs()
            ) as client:
                for text in texts:
                    body = self._build_body(text)
                    response = await client.invoke_model(
                        modelId=self._model,
                        body=json.dumps(body),
                        contentType="application/json",
                        accept="application/json",
                    )
                    response_body = json.loads(await response["body"].read())
                    embedding = self._extract_embedding(response_body)
                    embeddings.append(embedding)
        except Exception as e:
            raise OpenQueryAgentError(f"Bedrock embedding failed: {e}") from e
        return embeddings

    def _build_body(self, text: str) -> dict[str, Any]:
        """Build request body based on model family."""
        if "titan" in self._model:
            return {
                "inputText": text,
                "dimensions": self._dimensions,
            }
        if "cohere" in self._model:
            return {
                "texts": [text],
                "input_type": "search_query",
            }
        return {"inputText": text}

    def _extract_embedding(self, response: dict[str, Any]) -> list[float]:
        """Extract embedding from response based on model family."""
        if "titan" in self._model:
            return list(response.get("embedding", []))
        if "cohere" in self._model:
            embeddings = response.get("embeddings", [[]])
            return list(embeddings[0]) if embeddings else []
        return list(response.get("embedding", []))
