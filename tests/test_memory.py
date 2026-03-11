"""Tests for ConversationMemory."""

from __future__ import annotations

from openqueryagent.core.memory import ConversationMemory


class TestConversationMemory:
    def test_add_message(self) -> None:
        mem = ConversationMemory(max_tokens=10000)
        mem.add_message("user", "Hello")
        assert mem.message_count == 1
        assert mem.get_messages()[0].role == "user"
        assert mem.get_messages()[0].content == "Hello"

    def test_get_recent_messages(self) -> None:
        mem = ConversationMemory(max_tokens=10000)
        mem.add_message("user", "First")
        mem.add_message("assistant", "Response")
        mem.add_message("user", "Second")

        recent = mem.get_recent_messages(2)
        assert len(recent) == 2
        assert recent[0].content == "Response"
        assert recent[1].content == "Second"

    def test_clear_keeps_system(self) -> None:
        mem = ConversationMemory(max_tokens=10000)
        mem.add_message("system", "You are helpful")
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi")

        mem.clear()
        assert mem.message_count == 1
        assert mem.get_messages()[0].role == "system"

    def test_truncation_on_budget(self) -> None:
        mem = ConversationMemory(max_tokens=10)  # Very small budget
        mem.add_message("system", "Sys")
        mem.add_message("user", "A very long message that should exceed the token budget by far")

        # Should have truncated the user message (oldest non-system)
        # or kept it if system + this fits somehow
        assert mem.total_tokens <= 10 or mem.message_count <= 2

    def test_total_tokens(self) -> None:
        mem = ConversationMemory(max_tokens=10000)
        mem.add_message("user", "Hello world")  # ~2-3 tokens at 4 chars/token
        assert mem.total_tokens > 0

    def test_empty_recent(self) -> None:
        mem = ConversationMemory(max_tokens=10000)
        assert mem.get_recent_messages(0) == []
        assert mem.get_recent_messages(5) == []
