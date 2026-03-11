"""OpenTelemetry tracing for all pipeline stages.

Creates spans for each stage of the query pipeline:
``oqa.ask``, ``oqa.plan``, ``oqa.route``, ``oqa.execute``,
``oqa.execute.{adapter_id}``, ``oqa.rerank``, ``oqa.synthesize``.

When ``opentelemetry`` is not installed, all operations are **no-ops**
so the library works without the optional dependency.
"""

from __future__ import annotations

import contextlib
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

_HAS_OTEL = True
try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode, Tracer
except ImportError:  # pragma: no cover
    _HAS_OTEL = False

if TYPE_CHECKING:
    from collections.abc import Iterator


_TRACER_NAME = "openqueryagent"


class TracingManager:
    """Manages OpenTelemetry tracing for the pipeline.

    If ``opentelemetry`` is not installed, all methods are safe no-ops.

    Args:
        enabled: Whether tracing is active.  Even when ``True`` the
            library degrades gracefully if the SDK is not installed.
        service_name: OTEL service name (default ``openqueryagent``).
    """

    def __init__(self, enabled: bool = True, service_name: str = "openqueryagent") -> None:
        self._enabled = enabled and _HAS_OTEL
        self._tracer: Tracer | None = None
        self._service_name = service_name

        if self._enabled:
            self._tracer = trace.get_tracer(_TRACER_NAME)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
        """Start a tracing span as a context-manager.

        Args:
            name: Span name (e.g. ``oqa.ask``).
            attributes: Optional key-value attributes to set on the span.

        Yields:
            The span object (or ``None`` if tracing is disabled).
        """
        if not self._enabled or self._tracer is None:
            yield None
            return

        with self._tracer.start_as_current_span(name) as otel_span:
            if attributes:
                for key, value in attributes.items():
                    # OTel only accepts str, int, float, bool, or sequences thereof
                    if isinstance(value, (str, int, float, bool)):
                        otel_span.set_attribute(key, value)
                    elif isinstance(value, (list, tuple)):
                        otel_span.set_attribute(key, list(value))
                    else:
                        otel_span.set_attribute(key, str(value))
            try:
                yield otel_span
            except Exception as exc:
                otel_span.set_status(StatusCode.ERROR, str(exc))
                otel_span.record_exception(exc)
                raise

    def record_latency(self, span: Any, latency_ms: float) -> None:
        """Record latency on a span."""
        if span is not None and self._enabled:
            with contextlib.suppress(Exception):
                span.set_attribute("oqa.latency_ms", latency_ms)

    def record_result_count(self, span: Any, count: int) -> None:
        """Record result count on a span."""
        if span is not None and self._enabled:
            with contextlib.suppress(Exception):
                span.set_attribute("oqa.result_count", count)


# Module-level singleton — disabled by default, agent activates it.
_tracing = TracingManager(enabled=False)


def get_tracing() -> TracingManager:
    """Return the global tracing manager."""
    return _tracing


def configure_tracing(enabled: bool = True, service_name: str = "openqueryagent") -> TracingManager:
    """Configure the global tracing manager.

    Called once during agent initialization.
    """
    global _tracing
    _tracing = TracingManager(enabled=enabled, service_name=service_name)
    return _tracing
