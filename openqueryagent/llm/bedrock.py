"""AWS Bedrock LLM provider.

Implements the LLMProvider protocol for AWS Bedrock Runtime.
Supports Claude, Titan, Llama models via Bedrock.

Requires ``aiobotocore`` (install with ``pip install openqueryagent[bedrock]``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from openqueryagent.core.exceptions import OpenQueryAgentError
from openqueryagent.core.types import ChatMessage, TokenUsage
from openqueryagent.llm.base import LLMChunk, LLMResponse, ResponseFormat

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = structlog.get_logger(__name__)


class BedrockProvider:
    """AWS Bedrock LLM provider.

    Args:
        model: Bedrock model ID (e.g., ``anthropic.claude-3-sonnet-20240229-v1:0``).
        region: AWS region.
        aws_access_key_id: Optional AWS access key.
        aws_secret_access_key: Optional AWS secret key.
    """

    def __init__(
        self,
        model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        region: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        self._model = model
        self._region = region
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._session: Any = None
        self._init_session()

    def _init_session(self) -> None:
        """Initialize aiobotocore session."""
        try:
            from aiobotocore.session import AioSession
        except ImportError as e:
            msg = "aiobotocore is not installed. Install with: pip install openqueryagent[bedrock]"
            raise ImportError(msg) from e
        self._session = AioSession()

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client_kwargs(self) -> dict[str, Any]:
        """Build kwargs for creating the Bedrock client."""
        kwargs: dict[str, Any] = {"region_name": self._region}
        if self._aws_access_key_id and self._aws_secret_access_key:
            kwargs["aws_access_key_id"] = self._aws_access_key_id
            kwargs["aws_secret_access_key"] = self._aws_secret_access_key
        return kwargs

    async def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse:
        """Generate a completion from Bedrock."""
        body = self._build_request_body(messages, temperature, max_tokens)

        try:
            async with self._session.create_client(
                "bedrock-runtime", **self._get_client_kwargs()
            ) as client:
                response = await client.invoke_model(
                    modelId=self._model,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
                response_body = json.loads(
                    await response["body"].read(),
                )

            return self._parse_response(response_body)
        except Exception as e:
            raise OpenQueryAgentError(
                f"Bedrock completion failed: {e}",
            ) from e

    async def complete_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[LLMChunk]:
        """Generate a streaming completion from Bedrock."""
        body = self._build_request_body(messages, temperature, max_tokens)

        try:
            async with self._session.create_client(
                "bedrock-runtime", **self._get_client_kwargs()
            ) as client:
                response = await client.invoke_model_with_response_stream(
                    modelId=self._model,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
                async for event in response["body"]:
                    chunk_bytes = event.get("chunk", {}).get("bytes", b"{}")
                    chunk_data = json.loads(chunk_bytes)
                    if "anthropic" in self._model:
                        delta = chunk_data.get("delta", {})
                        content = delta.get("text", "")
                        stop = chunk_data.get("type") == "message_stop"
                    else:
                        content = chunk_data.get("outputText", "")
                        stop = chunk_data.get("completionReason") == "FINISH"

                    if content:
                        yield LLMChunk(
                            content=content,
                            model=self._model,
                            finish_reason="stop" if stop else None,
                        )
        except Exception as e:
            raise OpenQueryAgentError(
                f"Bedrock streaming failed: {e}",
            ) from e

    def _build_request_body(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Build request body based on model family."""
        if "anthropic" in self._model:
            system_msgs = [m for m in messages if m.role == "system"]
            chat_msgs = [m for m in messages if m.role != "system"]
            body: dict[str, Any] = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": m.role, "content": m.content} for m in chat_msgs
                ],
            }
            if system_msgs:
                body["system"] = system_msgs[0].content
            return body

        if "titan" in self._model:
            prompt = "\n".join(m.content for m in messages)
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                },
            }

        # Generic format (Llama, etc.)
        prompt = "\n".join(m.content for m in messages)
        return {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
        }

    def _parse_response(self, response_body: dict[str, Any]) -> LLMResponse:
        """Parse response body based on model family."""
        if "anthropic" in self._model:
            content_blocks = response_body.get("content", [])
            content = "".join(b.get("text", "") for b in content_blocks)
            usage = response_body.get("usage", {})
            return LLMResponse(
                content=content,
                model=self._model,
                usage=TokenUsage(
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                ),
                finish_reason=response_body.get("stop_reason", "end_turn"),
            )

        if "titan" in self._model:
            results = response_body.get("results", [{}])
            content = results[0].get("outputText", "") if results else ""
            return LLMResponse(
                content=content,
                model=self._model,
                usage=TokenUsage(
                    prompt_tokens=response_body.get("inputTextTokenCount", 0),
                    completion_tokens=results[0].get("tokenCount", 0) if results else 0,
                    total_tokens=response_body.get("inputTextTokenCount", 0) + (
                        results[0].get("tokenCount", 0) if results else 0
                    ),
                ),
                finish_reason=results[0].get("completionReason", "FINISH") if results else "FINISH",
            )

        # Generic (Llama, etc.)
        content = response_body.get("generation", "")
        return LLMResponse(
            content=content,
            model=self._model,
            usage=TokenUsage(
                prompt_tokens=response_body.get("prompt_token_count", 0),
                completion_tokens=response_body.get("generation_token_count", 0),
                total_tokens=response_body.get("prompt_token_count", 0) + response_body.get(
                    "generation_token_count", 0
                ),
            ),
            finish_reason=response_body.get("stop_reason", "stop"),
        )
