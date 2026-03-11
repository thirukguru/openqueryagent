"""Ollama LLM provider.

Implements the LLMProvider protocol for local/self-hosted models via Ollama.
Supports streaming, works with llama3, mistral, mixtral, etc.

Requires ``httpx`` (already a dependency via openai).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from openqueryagent.core.exceptions import OpenQueryAgentError
from openqueryagent.core.types import ChatMessage, TokenUsage
from openqueryagent.llm.base import LLMChunk, LLMResponse, ResponseFormat

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = structlog.get_logger(__name__)


class OllamaProvider:
    """Ollama LLM provider for local model inference.

    Args:
        model: Model name (e.g., ``llama3``, ``mistral``, ``mixtral``).
        base_url: Ollama API base URL.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the httpx async client."""
        try:
            import httpx
        except ImportError as e:
            msg = "httpx is not installed. Install with: pip install httpx"
            raise ImportError(msg) from e
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse:
        """Generate a completion from Ollama."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if response_format == ResponseFormat.JSON:
            payload["format"] = "json"

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            content: str = data.get("message", {}).get("content", "")
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)

            return LLMResponse(
                content=content,
                model=self._model,
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
                finish_reason="stop",
            )
        except Exception as e:
            raise OpenQueryAgentError(
                f"Ollama completion failed: {e}",
            ) from e

    async def complete_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[LLMChunk]:
        """Generate a streaming completion from Ollama."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    import json
                    data = json.loads(line)
                    chunk_content = data.get("message", {}).get("content", "")
                    done = data.get("done", False)
                    yield LLMChunk(
                        content=chunk_content,
                        model=self._model,
                        finish_reason="stop" if done else None,
                    )
        except Exception as e:
            raise OpenQueryAgentError(
                f"Ollama streaming failed: {e}",
            ) from e
