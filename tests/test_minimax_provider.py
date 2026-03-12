"""Tests for MiniMax LLM provider.

Uses mocked httpx client to test provider logic without real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openqueryagent.core.types import ChatMessage
from openqueryagent.llm.base import LLMResponse, ResponseFormat


class TestMiniMaxProvider:
    def test_model_name(self) -> None:
        from openqueryagent.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider(model="MiniMax-M2.5", api_key="test-key")
        assert provider.model_name == "MiniMax-M2.5"

    def test_default_model(self) -> None:
        from openqueryagent.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider(api_key="test-key")
        assert provider.model_name == "MiniMax-M2.5"

    def test_highspeed_model(self) -> None:
        from openqueryagent.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider(model="MiniMax-M2.5-highspeed", api_key="test-key")
        assert provider.model_name == "MiniMax-M2.5-highspeed"

    def test_api_key_from_env(self) -> None:
        with patch.dict("os.environ", {"MINIMAX_API_KEY": "env-key"}):
            from openqueryagent.llm.minimax import MiniMaxProvider

            provider = MiniMaxProvider(model="MiniMax-M2.5")
            assert provider._api_key == "env-key"

    def test_custom_base_url(self) -> None:
        from openqueryagent.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider(
            api_key="test-key", base_url="https://custom.api.example.com/v1/"
        )
        assert provider._base_url == "https://custom.api.example.com/v1"

    @pytest.mark.asyncio
    async def test_complete(self) -> None:
        from openqueryagent.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider(model="MiniMax-M2.5", api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "Hello from MiniMax!"},
                    "finish_reason": "stop",
                }
            ],
            "model": "MiniMax-M2.5",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        mock_response.raise_for_status = MagicMock()

        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_response)

        result = await provider.complete(
            messages=[ChatMessage(role="user", content="Hi")],
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from MiniMax!"
        assert result.model == "MiniMax-M2.5"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_complete_json_format(self) -> None:
        from openqueryagent.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider(model="MiniMax-M2.5", api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": '{"answer": "test"}'},
                    "finish_reason": "stop",
                }
            ],
            "model": "MiniMax-M2.5",
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }
        mock_response.raise_for_status = MagicMock()

        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_response)

        result = await provider.complete(
            messages=[ChatMessage(role="user", content="Return JSON")],
            response_format=ResponseFormat.JSON,
        )

        assert result.content == '{"answer": "test"}'

        # Verify response_format was passed in payload
        call_args = provider._client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_temperature_clamping(self) -> None:
        """MiniMax requires temperature > 0; verify 0.0 is clamped to 0.01."""
        from openqueryagent.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider(model="MiniMax-M2.5", api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "MiniMax-M2.5",
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
        mock_response.raise_for_status = MagicMock()

        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_response)

        await provider.complete(
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=0.0,
        )

        call_args = provider._client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["temperature"] == 0.01

    @pytest.mark.asyncio
    async def test_rate_limit_error(self) -> None:
        from openqueryagent.core.exceptions import RateLimitError
        from openqueryagent.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider(model="MiniMax-M2.5", api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 429

        provider._client = AsyncMock()
        provider._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(RateLimitError, match="MiniMax rate limit"):
            await provider.complete(
                messages=[ChatMessage(role="user", content="Hi")],
            )

    @pytest.mark.asyncio
    async def test_complete_stream(self) -> None:
        from openqueryagent.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider(model="MiniMax-M2.5", api_key="test-key")

        # Build SSE lines
        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}],"model":"MiniMax-M2.5"}',
            'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}],"model":"MiniMax-M2.5"}',
            'data: {"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}],"model":"MiniMax-M2.5"}',
            "data: [DONE]",
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = AsyncMock(return_value=_async_iter(lines))

        # Use an async context manager mock
        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=None)

        provider._client = MagicMock()
        provider._client.stream = MagicMock(return_value=mock_stream_cm)

        chunks = []
        async for chunk in provider.complete_stream(
            messages=[ChatMessage(role="user", content="Hi")],
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[2].content == "!"
        assert chunks[2].finish_reason == "stop"


async def _async_iter(items: list[str]):  # type: ignore[no-untyped-def]
    """Helper to create an async iterator from a list."""
    for item in items:
        yield item
