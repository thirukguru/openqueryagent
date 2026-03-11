"""OpenAI LLM provider.

Implements the LLMProvider protocol using the OpenAI Python SDK.
Supports Chat Completions API with streaming and JSON response format.
Also supports Azure OpenAI via ``api_base`` and ``api_version`` config.

Requires ``openai`` (install with ``pip install openqueryagent[openai]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from openqueryagent.core.exceptions import RateLimitError
from openqueryagent.core.types import ChatMessage, TokenUsage
from openqueryagent.llm.base import LLMChunk, LLMResponse, ResponseFormat

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = structlog.get_logger(__name__)


class OpenAIProvider:
    """OpenAI-compatible LLM provider.

    Works with OpenAI API and Azure OpenAI. Configure via constructor params.

    Args:
        model: Model identifier (e.g., ``gpt-4o``, ``gpt-4o-mini``).
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
        api_base: Custom API base URL (for Azure or local proxies).
        api_version: Azure OpenAI API version (e.g., ``2024-06-01``).
        organization: OpenAI organization ID.
        max_retries: Maximum retry attempts for transient errors.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        organization: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._client: Any = None
        self._init_client(api_key, api_base, api_version, organization)

    def _init_client(
        self,
        api_key: str | None,
        api_base: str | None,
        api_version: str | None,
        organization: str | None,
    ) -> None:
        """Initialize the OpenAI client (lazy import)."""
        try:
            import openai
        except ImportError as e:
            msg = "openai is not installed. Install with: pip install openqueryagent[openai]"
            raise ImportError(msg) from e

        if api_base and api_version:
            # Azure OpenAI
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=api_base,
                api_version=api_version,
            )
        else:
            self._client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
                organization=organization,
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
        """Generate a completion via OpenAI Chat API.

        Args:
            messages: Conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            response_format: TEXT or JSON output format.

        Returns:
            LLMResponse with content and token usage.

        Raises:
            RateLimitError: If rate-limited after retries.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format == ResponseFormat.JSON:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            if _is_rate_limit_error(e):
                raise RateLimitError(
                    f"OpenAI rate limit exceeded: {e}",
                    provider="openai",
                    model=self._model,
                ) from e
            raise

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            finish_reason=choice.finish_reason or "",
        )

    async def complete_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[LLMChunk]:
        """Generate a streaming completion via OpenAI Chat API.

        Yields:
            LLMChunk objects with incremental content.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    yield LLMChunk(
                        content=delta.content or "",
                        model=chunk.model or self._model,
                        finish_reason=chunk.choices[0].finish_reason,
                    )
        except Exception as e:
            if _is_rate_limit_error(e):
                raise RateLimitError(
                    f"OpenAI rate limit exceeded: {e}",
                    provider="openai",
                    model=self._model,
                ) from e
            raise


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if exception is a rate limit error."""
    try:
        import openai

        return isinstance(exc, openai.RateLimitError)
    except ImportError:
        return "rate_limit" in str(exc).lower() or "429" in str(exc)
