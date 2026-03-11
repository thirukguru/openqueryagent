"""Exception hierarchy for OpenQueryAgent.

All exceptions include structured context for debugging.
"""

from __future__ import annotations

from typing import Any


class OpenQueryAgentError(Exception):
    """Base exception for all OpenQueryAgent errors."""


class AdapterConnectionError(OpenQueryAgentError):
    """Raised when an adapter fails to connect to its backend."""

    def __init__(
        self,
        message: str,
        adapter_id: str,
        adapter_name: str = "",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.adapter_id = adapter_id
        self.adapter_name = adapter_name
        self.original_error = original_error


class AdapterQueryError(OpenQueryAgentError):
    """Raised when a query against an adapter fails."""

    def __init__(
        self,
        message: str,
        adapter_id: str,
        collection: str,
        adapter_name: str = "",
        query: Any = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.adapter_id = adapter_id
        self.collection = collection
        self.adapter_name = adapter_name
        self.query = query
        self.original_error = original_error


class PlannerError(OpenQueryAgentError):
    """Raised when query planning fails."""

    def __init__(
        self,
        message: str,
        query: str = "",
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.query = query
        self.original_error = original_error


class FilterCompilationError(OpenQueryAgentError):
    """Raised when filter compilation fails.

    This includes cases where an extended operator is not supported
    by the target backend.
    """

    def __init__(
        self,
        message: str,
        operator: str = "",
        field: str = "",
        adapter_id: str = "",
    ) -> None:
        super().__init__(message)
        self.operator = operator
        self.field = field
        self.adapter_id = adapter_id


class SynthesisError(OpenQueryAgentError):
    """Raised when answer synthesis fails."""

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.original_error = original_error


class QueryTimeoutError(OpenQueryAgentError):
    """Raised when a query exceeds the configured timeout."""

    def __init__(
        self,
        message: str,
        timeout_seconds: float = 0.0,
        adapter_id: str = "",
    ) -> None:
        super().__init__(message)
        self.timeout_seconds = timeout_seconds
        self.adapter_id = adapter_id


class SchemaError(OpenQueryAgentError):
    """Raised when schema introspection or validation fails."""

    def __init__(
        self,
        message: str,
        collection: str = "",
        adapter_id: str = "",
    ) -> None:
        super().__init__(message)
        self.collection = collection
        self.adapter_id = adapter_id


class RateLimitError(OpenQueryAgentError):
    """Raised when an external service returns a rate limit error."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        model: str = "",
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.retry_after_seconds = retry_after_seconds
