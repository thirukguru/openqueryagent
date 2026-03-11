"""Base protocol and types for LLM providers.

All LLM provider implementations must conform to the LLMProvider protocol.
Providers are used by the QueryPlanner (for query decomposition) and the
Synthesizer (for answer generation).
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel

from openqueryagent.core.types import ChatMessage, TokenUsage  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# ---------------------------------------------------------------------------
# Supporting Types
# ---------------------------------------------------------------------------


class ResponseFormat(StrEnum):
    """Response format for LLM completions."""

    TEXT = "text"
    JSON = "json"


class LLMResponse(BaseModel):
    """Response from an LLM completion call."""

    content: str
    model: str
    usage: TokenUsage
    finish_reason: str = ""


class LLMChunk(BaseModel):
    """A single chunk in a streaming LLM response."""

    content: str = ""
    model: str = ""
    finish_reason: str | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Interface for LLM providers used by planner and synthesizer.

    Implementations: OpenAIProvider, AnthropicProvider, OllamaProvider,
    BedrockProvider. Azure OpenAI uses OpenAIProvider with Azure-specific config.
    """

    @property
    def model_name(self) -> str:
        """The model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')."""
        ...

    async def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse:
        """Generate a completion from the LLM.

        Args:
            messages: Conversation messages (system, user, assistant).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
            response_format: Optional format constraint (TEXT or JSON).

        Returns:
            LLMResponse with generated content and usage stats.

        Raises:
            RateLimitError: If the provider rate-limits the request.
            OpenQueryAgentError: If the completion fails.
        """
        ...

    async def complete_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[LLMChunk]:
        """Generate a streaming completion from the LLM.

        Args:
            messages: Conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Yields:
            LLMChunk objects with incremental content.

        Raises:
            RateLimitError: If the provider rate-limits the request.
            OpenQueryAgentError: If the completion fails.
        """
        ...
