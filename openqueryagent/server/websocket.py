"""WebSocket endpoint for streaming ask responses."""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from openqueryagent.server.models import AskRequest

logger = structlog.get_logger(__name__)


async def ask_stream_handler(websocket: WebSocket) -> None:
    """Handle a streaming ask request over WebSocket.

    Protocol:
    1. Client sends JSON matching ``AskRequest`` schema.
    2. Server yields ``AskResponseChunk`` JSON frames.
    3. Final frame has ``is_final=True``.
    """
    await websocket.accept()

    try:
        # 1. Receive request
        raw: Any = await websocket.receive_json()
        request = AskRequest.model_validate(raw)

        # 2. Get agent from app state
        agent = websocket.app.state.agent

        # 3. Stream response
        async for chunk in await agent.ask(request.query, stream=True):  # type: ignore[union-attr]
            await websocket.send_json(chunk.model_dump())

        # 4. Send final marker (the last chunk from the generator already has
        #    is_final=True when synthesizer finishes, but we send an explicit
        #    close-clean signal as well).
        await websocket.close()

    except WebSocketDisconnect:
        logger.info("ws_client_disconnected")
    except json.JSONDecodeError:
        await websocket.send_json({"error": "Invalid JSON"})
        await websocket.close(code=1003)
    except Exception as exc:
        logger.error("ws_error", error=str(exc))
        try:
            await websocket.send_json({"error": str(exc)})
            await websocket.close(code=1011)
        except Exception:
            pass
