"""Tests for the OpenAI Embedding provider.

Uses mocked OpenAI client to test batching and dimension logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestOpenAIEmbedding:
    def test_model_name(self) -> None:
        with patch("openai.AsyncOpenAI"):
            from openqueryagent.embeddings.openai import OpenAIEmbedding

            emb = OpenAIEmbedding(model="text-embedding-3-small", api_key="test")
            assert emb.model_name == "text-embedding-3-small"

    def test_default_dimensions(self) -> None:
        with patch("openai.AsyncOpenAI"):
            from openqueryagent.embeddings.openai import OpenAIEmbedding

            emb_small = OpenAIEmbedding(model="text-embedding-3-small", api_key="test")
            assert emb_small.dimensions == 1536

            emb_large = OpenAIEmbedding(model="text-embedding-3-large", api_key="test")
            assert emb_large.dimensions == 3072

            emb_ada = OpenAIEmbedding(model="text-embedding-ada-002", api_key="test")
            assert emb_ada.dimensions == 1536

    def test_custom_dimensions(self) -> None:
        with patch("openai.AsyncOpenAI"):
            from openqueryagent.embeddings.openai import OpenAIEmbedding

            emb = OpenAIEmbedding(
                model="text-embedding-3-small", api_key="test", dimensions=512
            )
            assert emb.dimensions == 512

    @pytest.mark.asyncio
    async def test_embed_query(self) -> None:
        with patch("openai.AsyncOpenAI") as mock_cls:
            from openqueryagent.embeddings.openai import OpenAIEmbedding

            mock_client = AsyncMock()
            mock_data = MagicMock()
            mock_data.embedding = [0.1, 0.2, 0.3]
            mock_data.index = 0

            mock_response = MagicMock()
            mock_response.data = [mock_data]
            mock_client.embeddings.create.return_value = mock_response
            mock_cls.return_value = mock_client

            emb = OpenAIEmbedding(model="text-embedding-3-small", api_key="test")
            emb._client = mock_client

            result = await emb.embed_query("test text")
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_documents_batching(self) -> None:
        with patch("openai.AsyncOpenAI") as mock_cls:
            from openqueryagent.embeddings.openai import OpenAIEmbedding

            mock_client = AsyncMock()

            def make_batch_response(n: int) -> MagicMock:
                response = MagicMock()
                data_items = []
                for i in range(n):
                    item = MagicMock()
                    item.embedding = [0.1 * (i + 1)] * 3
                    item.index = i
                    data_items.append(item)
                response.data = data_items
                return response

            # Create embedding provider with small batch size
            mock_cls.return_value = mock_client
            emb = OpenAIEmbedding(model="text-embedding-3-small", api_key="test", batch_size=2)
            emb._client = mock_client

            # Mock two batch calls
            mock_client.embeddings.create.side_effect = [
                make_batch_response(2),
                make_batch_response(1),
            ]

            texts = ["text1", "text2", "text3"]
            result = await emb.embed_documents(texts)

            assert len(result) == 3
            assert mock_client.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_query_with_custom_dimensions(self) -> None:
        with patch("openai.AsyncOpenAI") as mock_cls:
            from openqueryagent.embeddings.openai import OpenAIEmbedding

            mock_client = AsyncMock()
            mock_data = MagicMock()
            mock_data.embedding = [0.1, 0.2]
            mock_data.index = 0

            mock_response = MagicMock()
            mock_response.data = [mock_data]
            mock_client.embeddings.create.return_value = mock_response
            mock_cls.return_value = mock_client

            emb = OpenAIEmbedding(
                model="text-embedding-3-small", api_key="test", dimensions=256
            )
            emb._client = mock_client

            await emb.embed_query("test")

            # Verify dimensions param was passed for v3 model
            call_kwargs = mock_client.embeddings.create.call_args.kwargs
            assert call_kwargs["dimensions"] == 256
