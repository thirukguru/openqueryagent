"""Server middleware: API-key auth, rate limiting, request-ID injection."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)

# Paths that bypass authentication
_PUBLIC_PATHS = frozenset({"/v1/health", "/docs", "/openapi.json", "/redoc"})


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject ``X-Request-ID`` header on every response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        import uuid

        import structlog

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id  # type: ignore[attr-defined]

        # Bind correlation ID into structlog context for all downstream logs
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate ``X-API-Key`` header when an API key is configured."""

    def __init__(self, app: Any, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip auth for public endpoints
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if provided != self._api_key:
            logger.warning("auth_rejected", path=request.url.path)
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": "Invalid or missing API key"},
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory token-bucket rate limiter per client IP.

    ``rate_limit`` format: ``"<max_requests>/<period>"`` where period is
    one of ``second``, ``minute``, ``hour``.
    """

    _PERIOD_MAP: dict[str, float] = {
        "second": 1.0,
        "minute": 60.0,
        "hour": 3600.0,
    }

    def __init__(self, app: Any, rate_limit: str) -> None:
        super().__init__(app)
        count_str, period_str = rate_limit.strip().split("/")
        self._max_requests = int(count_str)
        self._period = self._PERIOD_MAP[period_str.strip().lower()]
        # IP -> list of request timestamps
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        # Prune old entries
        window_start = now - self._period
        timestamps = self._requests[client_ip]
        self._requests[client_ip] = [t for t in timestamps if t > window_start]

        if len(self._requests[client_ip]) >= self._max_requests:
            logger.warning("rate_limited", client_ip=client_ip)
            return JSONResponse(
                status_code=429,
                content={"error": "Too Many Requests", "detail": "Rate limit exceeded"},
            )

        self._requests[client_ip].append(now)
        return await call_next(request)
