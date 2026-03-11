"""Prometheus metrics for request and adapter monitoring.

When ``prometheus_client`` is not installed, all operations are **no-ops**.

Metrics exposed:

- ``oqa_requests_total``           — Counter(endpoint, status)
- ``oqa_request_duration_seconds`` — Histogram(endpoint)
- ``oqa_active_requests``          — Gauge
- ``oqa_adapter_queries_total``    — Counter(adapter, status)
- ``oqa_adapter_query_duration_seconds`` — Histogram(adapter)
"""

from __future__ import annotations

import contextlib
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

_HAS_PROMETHEUS = True
try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
except ImportError:  # pragma: no cover
    _HAS_PROMETHEUS = False

if TYPE_CHECKING:
    from collections.abc import Iterator


class MetricsManager:
    """Manages Prometheus metrics.

    All methods are safe no-ops when ``prometheus_client`` is not installed.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled and _HAS_PROMETHEUS

        if self._enabled:
            self._requests_total = Counter(
                "oqa_requests_total",
                "Total API requests",
                ["endpoint", "status"],
            )
            self._request_duration = Histogram(
                "oqa_request_duration_seconds",
                "Request duration in seconds",
                ["endpoint"],
                buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            )
            self._active_requests = Gauge(
                "oqa_active_requests",
                "Currently active requests",
            )
            self._adapter_queries_total = Counter(
                "oqa_adapter_queries_total",
                "Total adapter queries",
                ["adapter", "status"],
            )
            self._adapter_query_duration = Histogram(
                "oqa_adapter_query_duration_seconds",
                "Adapter query duration in seconds",
                ["adapter"],
                buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0),
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def inc_request(self, endpoint: str, status: str = "success") -> None:
        """Increment the request counter."""
        if self._enabled:
            self._requests_total.labels(endpoint=endpoint, status=status).inc()

    @contextmanager
    def track_request(self, endpoint: str) -> Iterator[None]:
        """Track a request: duration histogram + active gauge."""
        if not self._enabled:
            yield
            return

        self._active_requests.inc()
        start = time.monotonic()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.monotonic() - start
            self._request_duration.labels(endpoint=endpoint).observe(duration)
            self._requests_total.labels(endpoint=endpoint, status=status).inc()
            self._active_requests.dec()

    def observe_adapter_query(
        self, adapter: str, duration_seconds: float, status: str = "success"
    ) -> None:
        """Record an adapter query."""
        if self._enabled:
            self._adapter_queries_total.labels(adapter=adapter, status=status).inc()
            self._adapter_query_duration.labels(adapter=adapter).observe(duration_seconds)

    def generate_latest(self) -> bytes:
        """Generate Prometheus exposition format."""
        if self._enabled:
            return generate_latest()  # type: ignore[no-any-return]
        return b""


# Module-level singleton
_metrics = MetricsManager(enabled=False)


def get_metrics() -> MetricsManager:
    """Return the global metrics manager."""
    return _metrics


def configure_metrics(enabled: bool = True) -> MetricsManager:
    """Configure the global metrics manager."""
    global _metrics
    _metrics = MetricsManager(enabled=enabled)
    return _metrics
