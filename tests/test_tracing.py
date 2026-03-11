"""Tests for OpenTelemetry tracing manager."""

from __future__ import annotations

import pytest

from openqueryagent.observability.tracing import TracingManager, configure_tracing, get_tracing


class TestTracingManagerDisabled:
    """Tests when tracing is disabled (no OTel dependency needed)."""

    def test_disabled_span_is_noop(self) -> None:
        tracing = TracingManager(enabled=False)
        with tracing.span("test.span") as span:
            assert span is None

    def test_disabled_record_latency_is_noop(self) -> None:
        tracing = TracingManager(enabled=False)
        # Should not raise
        tracing.record_latency(None, 123.0)

    def test_disabled_record_result_count_is_noop(self) -> None:
        tracing = TracingManager(enabled=False)
        tracing.record_result_count(None, 42)

    def test_enabled_property(self) -> None:
        tracing = TracingManager(enabled=False)
        assert tracing.enabled is False


class TestTracingGlobalConfig:
    """Tests for global tracing configuration."""

    def test_configure_and_get(self) -> None:
        tracing = configure_tracing(enabled=False)
        assert tracing is get_tracing()
        assert tracing.enabled is False

    def test_reconfigure(self) -> None:
        configure_tracing(enabled=False)
        t1 = get_tracing()
        configure_tracing(enabled=False, service_name="custom")
        t2 = get_tracing()
        assert t1 is not t2
