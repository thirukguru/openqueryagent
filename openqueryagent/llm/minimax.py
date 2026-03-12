"""MiniMax LLM provider.

Implements the LLMProvider protocol using the MiniMax API (OpenAI-compatible).
Supports Chat Completions with streaming, JSON response format, and
MiniMax-M2.5 / MiniMax-M2.5-highspeed models with 204K context window.

Uses ``httpx`` (already a core dependency) — no extra install required.

Usage::

    from openqueryagent.llm.minimax import MiniMaxProvider

    llm = MiniMaxProvider(model="MiniMax-M2.5", api_key="your-key")
    response = await llm.complete(messages=[...])
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from openqueryagent.core.exceptions import OpenQueryAgentError, RateLimitError
from openqueryagent.core.types import ChatMessage, TokenUsage
from openqueryagent.llm.base import LLMChunk, LLMResponse, ResponseFormat

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = structlog.get_logger(__name__)

# MiniMax API base URL (OpenAI-compatible endpoint)
_DEFAULT_BASE_URL = "https://api.minimax.io/v1"


class MiniMaxProvider:
    """MiniMax LLM provider.

    Connects to the MiniMax API using its OpenAI-compatible Chat Completions
    endpoint.  Supports ``MiniMax-M2.5`` and ``MiniMax-M2.5-highspeed`` models
    (204K context window).

    Args:
        model: Model identifier (e.g., ``MiniMax-M2.5``,
            ``MiniMax-M2.5-highspeed``).
        api_key: MiniMax API key. Falls back to ``MINIMAX_API_KEY`` env var.
        base_url: API base URL. Defaults to ``https://api.minimax.io/v1``.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts for transient errors.
    """

    def __init__(
        self,
        model: str = "MiniMax-M2.5",
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
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
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse:
        """Generate a completion via MiniMax Chat API.

        Args:
            messages: Conversation messages.
            temperature: Sampling temperature. MiniMax requires a value
                in (0.0, 1.0], so 0.0 is clamped to 0.01.
            max_tokens: Maximum response tokens.
            response_format: TEXT or JSON output format.

        Returns:
            LLMResponse with content and token usage.

        Raises:
            RateLimitError: If rate-limited after retries.
            OpenQueryAgentError: If the completion fails.
        """
        # MiniMax rejects temperature=0; clamp to small positive value
        safe_temperature = max(temperature, 0.01)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": safe_temperature,
            "max_tokens": max_tokens,
        }

        if response_format == ResponseFormat.JSON:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = await self._client.post("/chat/completions", json=payload)

            if response.status_code == 429:
                raise RateLimitError(
                    "MiniMax rate limit exceeded",
                    provider="minimax",
                    model=self._model,
                )

            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            usage = data.get("usage", {})

            return LLMResponse(
                content=choice["message"]["content"] or "",
                model=data.get("model", self._model),
                usage=TokenUsage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                ),
                finish_reason=choice.get("finish_reason", ""),
            )
        except RateLimitError:
            raise
        except Exception as e:
            if "429" in str(e):
                raise RateLimitError(
                    f"MiniMax rate limit exceeded: {e}",
                    provider="minimax",
                    model=self._model,
                ) from e
            raise OpenQueryAgentError(
                f"MiniMax completion failed: {e}",
            ) from e

    async def complete_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[LLMChunk]:
        """Generate a streaming completion via MiniMax Chat API.

        Yields:
            LLMChunk objects with incremental content.
        """
        safe_temperature = max(temperature, 0.01)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": safe_temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                if response.status_code == 429:
                    raise RateLimitError(
                        "MiniMax rate limit exceeded",
                        provider="minimax",
                        model=self._model,
                    )
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break

                    data = json.loads(data_str)
                    if not data.get("choices"):
                        continue

                    delta = data["choices"][0].get("delta", {})
                    yield LLMChunk(
                        content=delta.get("content", ""),
                        model=data.get("model", self._model),
                        finish_reason=data["choices"][0].get("finish_reason"),
                    )
        except RateLimitError:
            raise
        except Exception as e:
            raise OpenQueryAgentError(
                f"MiniMax streaming failed: {e}",
            ) from e
