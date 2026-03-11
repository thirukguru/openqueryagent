"""Tests for exception hierarchy."""

from __future__ import annotations

from openqueryagent.core.exceptions import (
    AdapterConnectionError,
    AdapterQueryError,
    FilterCompilationError,
    OpenQueryAgentError,
    PlannerError,
    QueryTimeoutError,
    RateLimitError,
    SchemaError,
    SynthesisError,
)


class TestExceptionHierarchy:
    """All exceptions inherit from OpenQueryAgentError."""

    def test_base_exception(self) -> None:
        err = OpenQueryAgentError("test")
        assert str(err) == "test"
        assert isinstance(err, Exception)

    def test_adapter_connection_error(self) -> None:
        original = ConnectionError("refused")
        err = AdapterConnectionError(
            "Failed to connect", adapter_id="qdrant-1", original_error=original,
        )
        assert isinstance(err, OpenQueryAgentError)
        assert err.adapter_id == "qdrant-1"
        assert err.original_error is original

    def test_adapter_query_error(self) -> None:
        err = AdapterQueryError(
            "Query failed", adapter_id="qdrant-1",
            collection="products", query="test query",
        )
        assert isinstance(err, OpenQueryAgentError)
        assert err.collection == "products"

    def test_planner_error(self) -> None:
        err = PlannerError("Invalid plan", query="find shoes")
        assert isinstance(err, OpenQueryAgentError)
        assert err.query == "find shoes"

    def test_filter_compilation_error(self) -> None:
        err = FilterCompilationError(
            "Operator $starts_with not supported on Pinecone",
            operator="$starts_with", field="name", adapter_id="pinecone-1",
        )
        assert isinstance(err, OpenQueryAgentError)
        assert err.operator == "$starts_with"

    def test_synthesis_error(self) -> None:
        err = SynthesisError("LLM returned empty")
        assert isinstance(err, OpenQueryAgentError)

    def test_query_timeout_error(self) -> None:
        err = QueryTimeoutError(
            "Query timed out", timeout_seconds=30.0, adapter_id="milvus-1",
        )
        assert isinstance(err, OpenQueryAgentError)
        assert err.timeout_seconds == 30.0

    def test_schema_error(self) -> None:
        err = SchemaError(
            "Collection not found", collection="unknown", adapter_id="qdrant-1",
        )
        assert isinstance(err, OpenQueryAgentError)
        assert err.collection == "unknown"

    def test_rate_limit_error(self) -> None:
        err = RateLimitError(
            "Rate limited", provider="openai", retry_after_seconds=5.0,
        )
        assert isinstance(err, OpenQueryAgentError)
        assert err.retry_after_seconds == 5.0

    def test_all_catchable_by_base(self) -> None:
        """All errors can be caught by catching OpenQueryAgentError."""
        errors = [
            AdapterConnectionError("x", adapter_id="a"),
            AdapterQueryError("x", adapter_id="a", collection="c"),
            PlannerError("x"),
            FilterCompilationError("x"),
            SynthesisError("x"),
            QueryTimeoutError("x"),
            SchemaError("x"),
            RateLimitError("x"),
        ]
        for err in errors:
            try:
                raise err
            except OpenQueryAgentError:
                pass  # All caught correctly
