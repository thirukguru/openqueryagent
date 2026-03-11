"""Conversation memory for multi-turn dialogue.

Tracks conversation history with token counting for context window
management. Supports truncation to fit within LLM token budgets.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from openqueryagent.core.types import ChatMessage


class ConversationMemory(BaseModel):
    """In-memory conversation history with token budget management.

    Stores messages and truncates older messages when the token budget
    is exceeded, keeping the system prompt and most recent messages.

    Args:
        max_tokens: Maximum token budget for conversation history.
        messages: List of conversation messages.
    """

    max_tokens: int = 4096
    messages: list[ChatMessage] = Field(default_factory=list)
    _token_counts: list[int] = []

    def model_post_init(self, __context: Any) -> None:
        """Initialize token counts for existing messages."""
        self._token_counts = [self._estimate_tokens(m.content) for m in self.messages]

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: Message role ('system', 'user', 'assistant').
            content: Message content.
        """
        msg = ChatMessage(role=role, content=content)  # type: ignore[arg-type]
        tokens = self._estimate_tokens(content)
        self.messages.append(msg)
        self._token_counts.append(tokens)
        self._truncate_if_needed()

    def get_messages(self) -> list[ChatMessage]:
        """Get all messages in the conversation history.

        Returns:
            List of ChatMessage objects.
        """
        return list(self.messages)

    def get_recent_messages(self, n: int) -> list[ChatMessage]:
        """Get the most recent N messages.

        Args:
            n: Number of recent messages to return.

        Returns:
            List of the most recent N messages.
        """
        return list(self.messages[-n:]) if n > 0 else []

    def clear(self) -> None:
        """Clear all messages except the system prompt."""
        system_msgs = [m for m in self.messages if m.role == "system"]
        self.messages = system_msgs
        self._token_counts = [self._estimate_tokens(m.content) for m in self.messages]

    @property
    def total_tokens(self) -> int:
        """Total estimated tokens across all messages."""
        return sum(self._token_counts)

    @property
    def message_count(self) -> int:
        """Number of messages in history."""
        return len(self.messages)

    def _truncate_if_needed(self) -> None:
        """Remove oldest non-system messages to fit within token budget."""
        while self.total_tokens > self.max_tokens and len(self.messages) > 1:
            # Find oldest non-system message
            for i, msg in enumerate(self.messages):
                if msg.role != "system":
                    self.messages.pop(i)
                    self._token_counts.pop(i)
                    break
            else:
                break  # Only system messages remain

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count using simple heuristic.

        Uses ~4 chars per token as a rough estimate. For more accurate
        counting, integrate tiktoken.

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        return max(1, len(text) // 4)
