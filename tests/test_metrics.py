"""Tests for Prometheus metrics manager."""

from __future__ import annotations

import pytest

from openqueryagent.observability.metrics import MetricsManager, configure_metrics, get_metrics


class TestMetricsManagerDisabled:
    """Tests when metrics are disabled."""

    def test_disabled_inc_request_is_noop(self) -> None:
        metrics = MetricsManager(enabled=False)
        # Should not raise
        metrics.inc_request("ask")

    def test_disabled_track_request_is_noop(self) -> None:
        metrics = MetricsManager(enabled=False)
        with metrics.track_request("search"):
            pass

    def test_disabled_observe_adapter_query_is_noop(self) -> None:
        metrics = MetricsManager(enabled=False)
        metrics.observe_adapter_query("qdrant", 0.5)

    def test_disabled_generate_latest_returns_empty(self) -> None:
        metrics = MetricsManager(enabled=False)
        assert metrics.generate_latest() == b""

    def test_enabled_property(self) -> None:
        metrics = MetricsManager(enabled=False)
        assert metrics.enabled is False


class TestMetricsGlobalConfig:
    """Tests for global metrics configuration."""

    def test_configure_and_get(self) -> None:
        metrics = configure_metrics(enabled=False)
        assert metrics is get_metrics()

    def test_reconfigure(self) -> None:
        configure_metrics(enabled=False)
        m1 = get_metrics()
        configure_metrics(enabled=False)
        m2 = get_metrics()
        assert m1 is not m2
