"""Tests for LLM providers — OpenAI and Anthropic.

Uses mocked API clients to test provider logic without real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openqueryagent.core.types import ChatMessage
from openqueryagent.llm.base import LLMResponse, ResponseFormat

# ===========================================================================
# OpenAI Provider Tests
# ===========================================================================


class TestOpenAIProvider:
    def test_model_name(self) -> None:
        with patch("openai.AsyncOpenAI"):
            from openqueryagent.llm.openai import OpenAIProvider

            provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
            assert provider.model_name == "gpt-4o"

    @pytest.mark.asyncio
    async def test_complete(self) -> None:
        with patch("openai.AsyncOpenAI") as mock_cls:
            from openqueryagent.llm.openai import OpenAIProvider

            mock_client = AsyncMock()

            # Build mock response
            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 20
            mock_usage.total_tokens = 30

            mock_choice = MagicMock()
            mock_choice.message.content = "Hello, world!"
            mock_choice.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.model = "gpt-4o"
            mock_response.usage = mock_usage

            mock_client.chat.completions.create.return_value = mock_response
            mock_cls.return_value = mock_client

            provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
            provider._client = mock_client

            result = await provider.complete(
                messages=[ChatMessage(role="user", content="Hi")],
            )

            assert isinstance(result, LLMResponse)
            assert result.content == "Hello, world!"
            assert result.model == "gpt-4o"
            assert result.usage.total_tokens == 30
            assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_complete_json_format(self) -> None:
        with patch("openai.AsyncOpenAI") as mock_cls:
            from openqueryagent.llm.openai import OpenAIProvider

            mock_client = AsyncMock()

            mock_usage = MagicMock()
            mock_usage.prompt_tokens = 10
            mock_usage.completion_tokens = 20
            mock_usage.total_tokens = 30

            mock_choice = MagicMock()
            mock_choice.message.content = '{"answer": "test"}'
            mock_choice.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.model = "gpt-4o"
            mock_response.usage = mock_usage

            mock_client.chat.completions.create.return_value = mock_response
            mock_cls.return_value = mock_client

            provider = OpenAIProvider(model="gpt-4o", api_key="test-key")
            provider._client = mock_client

            result = await provider.complete(
                messages=[ChatMessage(role="user", content="Return JSON")],
                response_format=ResponseFormat.JSON,
            )

            assert result.content == '{"answer": "test"}'

            # Verify response_format was passed
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["response_format"] == {"type": "json_object"}


# ===========================================================================
# Anthropic Provider Tests
# ===========================================================================


class TestAnthropicProvider:
    def test_model_name(self) -> None:
        with patch("anthropic.AsyncAnthropic"):
            from openqueryagent.llm.anthropic import AnthropicProvider

            provider = AnthropicProvider(
                model="claude-sonnet-4-20250514", api_key="test-key"
            )
            assert provider.model_name == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_complete(self) -> None:
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            from openqueryagent.llm.anthropic import AnthropicProvider

            mock_client = AsyncMock()

            mock_usage = MagicMock()
            mock_usage.input_tokens = 15
            mock_usage.output_tokens = 25

            mock_content = MagicMock()
            mock_content.text = "Hello from Claude!"

            mock_response = MagicMock()
            mock_response.content = [mock_content]
            mock_response.model = "claude-sonnet-4-20250514"
            mock_response.usage = mock_usage
            mock_response.stop_reason = "end_turn"

            mock_client.messages.create.return_value = mock_response
            mock_cls.return_value = mock_client

            provider = AnthropicProvider(
                model="claude-sonnet-4-20250514", api_key="test-key"
            )
            provider._client = mock_client

            result = await provider.complete(
                messages=[ChatMessage(role="user", content="Hi")],
            )

            assert isinstance(result, LLMResponse)
            assert result.content == "Hello from Claude!"
            assert result.usage.prompt_tokens == 15
            assert result.usage.completion_tokens == 25
            assert result.usage.total_tokens == 40

    @pytest.mark.asyncio
    async def test_complete_json_extraction_from_markdown(self) -> None:
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            from openqueryagent.llm.anthropic import AnthropicProvider

            mock_client = AsyncMock()

            mock_usage = MagicMock()
            mock_usage.input_tokens = 15
            mock_usage.output_tokens = 25

            # Claude wraps JSON in markdown code block
            mock_content = MagicMock()
            mock_content.text = 'Here is the JSON:\n```json\n{"answer": "test"}\n```'

            mock_response = MagicMock()
            mock_response.content = [mock_content]
            mock_response.model = "claude-sonnet-4-20250514"
            mock_response.usage = mock_usage
            mock_response.stop_reason = "end_turn"

            mock_client.messages.create.return_value = mock_response
            mock_cls.return_value = mock_client

            provider = AnthropicProvider(
                model="claude-sonnet-4-20250514", api_key="test-key"
            )
            provider._client = mock_client

            result = await provider.complete(
                messages=[ChatMessage(role="user", content="Return JSON")],
                response_format=ResponseFormat.JSON,
            )

            assert result.content == '{"answer": "test"}'

    def test_prepare_messages_system_separation(self) -> None:
        from openqueryagent.llm.anthropic import AnthropicProvider

        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello!"),
        ]

        system_msg, api_msgs = AnthropicProvider._prepare_messages(messages)
        assert system_msg == "You are helpful."
        assert len(api_msgs) == 2
        assert api_msgs[0]["role"] == "user"
        assert api_msgs[1]["role"] == "assistant"


# ===========================================================================
# JSON Extraction Tests
# ===========================================================================


class TestJsonExtraction:
    def test_extract_from_code_block(self) -> None:
        from openqueryagent.llm.anthropic import _extract_json

        text = '```json\n{"key": "value"}\n```'
        assert _extract_json(text) == '{"key": "value"}'

    def test_extract_raw_json(self) -> None:
        from openqueryagent.llm.anthropic import _extract_json

        text = '{"key": "value"}'
        assert _extract_json(text) == '{"key": "value"}'

    def test_extract_from_code_block_no_language(self) -> None:
        from openqueryagent.llm.anthropic import _extract_json

        text = '```\n{"key": "value"}\n```'
        assert _extract_json(text) == '{"key": "value"}'

    def test_extract_returns_raw_when_no_json(self) -> None:
        from openqueryagent.llm.anthropic import _extract_json

        text = "not json at all"
        assert _extract_json(text) == "not json at all"
