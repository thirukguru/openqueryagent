"""Anthropic LLM provider.

Implements the LLMProvider protocol using the Anthropic Python SDK.
Supports Messages API with streaming and JSON extraction.

Requires ``anthropic`` (install with ``pip install openqueryagent[anthropic]``).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from openqueryagent.core.exceptions import RateLimitError
from openqueryagent.core.types import ChatMessage, TokenUsage
from openqueryagent.llm.base import LLMChunk, LLMResponse, ResponseFormat

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = structlog.get_logger(__name__)


# Regex to extract JSON from markdown code blocks
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


class AnthropicProvider:
    """Anthropic Claude LLM provider.

    Args:
        model: Model identifier (e.g., ``claude-sonnet-4-20250514``).
        api_key: Anthropic API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
        max_retries: Maximum retry attempts for transient errors.
        max_tokens: Default max tokens for responses.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_retries: int = 3,
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._default_max_tokens = max_tokens
        self._client: Any = None
        self._init_client(api_key)

    def _init_client(self, api_key: str | None) -> None:
        """Initialize the Anthropic client (lazy import)."""
        try:
            import anthropic
        except ImportError as e:
            msg = (
                "anthropic is not installed. "
                "Install with: pip install openqueryagent[anthropic]"
            )
            raise ImportError(msg) from e

        self._client = anthropic.AsyncAnthropic(api_key=api_key)

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
        """Generate a completion via Anthropic Messages API.

        For JSON response format, the system prompt is augmented to request
        JSON output, and the response is parsed to extract JSON content
        (including from markdown code blocks).

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
        system_msg, api_messages = self._prepare_messages(messages)

        # Augment system prompt for JSON mode
        if response_format == ResponseFormat.JSON:
            json_instruction = (
                "\n\nYou MUST respond with valid JSON only. "
                "Do not include any text before or after the JSON object."
            )
            system_msg = (system_msg or "") + json_instruction

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
        }
        if system_msg:
            kwargs["system"] = system_msg

        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as e:
            if _is_rate_limit_error(e):
                raise RateLimitError(
                    f"Anthropic rate limit exceeded: {e}",
                    provider="anthropic",
                    model=self._model,
                ) from e
            raise

        content = response.content[0].text if response.content else ""

        # For JSON mode, try to extract JSON from markdown blocks
        if response_format == ResponseFormat.JSON:
            content = _extract_json(content)

        return LLMResponse(
            content=content,
            model=response.model,
            usage=TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            finish_reason=response.stop_reason or "",
        )

    async def complete_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[LLMChunk]:
        """Generate a streaming completion via Anthropic Messages API.

        Yields:
            LLMChunk objects with incremental content.
        """
        system_msg, api_messages = self._prepare_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": max_tokens or self._default_max_tokens,
            "temperature": temperature,
        }
        if system_msg:
            kwargs["system"] = system_msg

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield LLMChunk(
                        content=text,
                        model=self._model,
                    )
        except Exception as e:
            if _is_rate_limit_error(e):
                raise RateLimitError(
                    f"Anthropic rate limit exceeded: {e}",
                    provider="anthropic",
                    model=self._model,
                ) from e
            raise

    @staticmethod
    def _prepare_messages(
        messages: list[ChatMessage],
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Split system message from user/assistant messages.

        Anthropic requires the system message as a separate parameter.

        Returns:
            (system_message, [{"role": ..., "content": ...}])
        """
        system_msg: str | None = None
        api_messages: list[dict[str, str]] = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        return system_msg, api_messages


def _extract_json(text: str) -> str:
    """Extract JSON from text, handling markdown code blocks.

    Claude often wraps JSON in ```json ... ``` blocks. This function
    extracts the JSON content from such blocks, or returns the raw
    text if no blocks are found.
    """
    # Try markdown code block extraction
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()

    # Try raw JSON parsing
    text = text.strip()
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass

    return text


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if exception is a rate limit error."""
    try:
        import anthropic

        return isinstance(exc, anthropic.RateLimitError)
    except ImportError:
        return "rate_limit" in str(exc).lower() or "429" in str(exc)
