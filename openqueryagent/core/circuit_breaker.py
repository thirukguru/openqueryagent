"""Per-adapter circuit breaker.

Prevents hammering a failing adapter by short-circuiting calls
when the adapter has exceeded a failure threshold.

States:
- **Closed** — normal operation, errors are counted.
- **Open** — adapter is considered down, calls are immediately rejected.
- **Half-open** — one trial call is allowed to test recovery.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

import structlog

from openqueryagent.core.exceptions import AdapterConnectionError

logger = structlog.get_logger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-adapter circuit breaker.

    Args:
        adapter_id: Identifier for logging.
        failure_threshold: Consecutive failures before opening.
        recovery_timeout: Seconds to wait before half-opening.
        half_open_max_calls: Max trial calls in half-open state.
    """

    def __init__(
        self,
        adapter_id: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self._adapter_id = adapter_id
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state, accounting for automatic transitions."""
        if (
            self._state == CircuitState.OPEN
            and (time.monotonic() - self._last_failure_time) >= self._recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            logger.info("circuit_half_open", adapter=self._adapter_id)
        return self._state

    def pre_call(self) -> None:
        """Check if a call is allowed. Raises if circuit is open.

        Raises:
            AdapterConnectionError: If the circuit is open.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise AdapterConnectionError(
                f"Circuit breaker OPEN for adapter '{self._adapter_id}'. "
                f"Retry after {self._recovery_timeout}s.",
                adapter_id=self._adapter_id,
            )

        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self._half_open_max_calls:
                raise AdapterConnectionError(
                    f"Circuit breaker HALF_OPEN for adapter '{self._adapter_id}': "
                    f"max trial calls ({self._half_open_max_calls}) reached.",
                    adapter_id=self._adapter_id,
                )
            self._half_open_calls += 1

    def on_success(self) -> None:
        """Record a successful call. Resets the circuit to closed."""
        if self._state in (CircuitState.HALF_OPEN, CircuitState.CLOSED):
            if self._failure_count > 0 or self._state == CircuitState.HALF_OPEN:
                logger.info(
                    "circuit_closed",
                    adapter=self._adapter_id,
                    previous_failures=self._failure_count,
                )
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0

    def on_failure(self) -> None:
        """Record a failed call. May transition to open."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Trial call failed — immediately re-open
            self._state = CircuitState.OPEN
            logger.warning(
                "circuit_reopened",
                adapter=self._adapter_id,
                failure_count=self._failure_count,
            )
        elif self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "circuit_opened",
                adapter=self._adapter_id,
                failure_count=self._failure_count,
            )

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0


class CircuitBreakerRegistry:
    """Registry of circuit breakers, one per adapter."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, adapter_id: str) -> CircuitBreaker:
        """Get or create a circuit breaker for an adapter."""
        if adapter_id not in self._breakers:
            self._breakers[adapter_id] = CircuitBreaker(
                adapter_id=adapter_id,
                failure_threshold=self._failure_threshold,
                recovery_timeout=self._recovery_timeout,
            )
        return self._breakers[adapter_id]

    @property
    def breakers(self) -> dict[str, CircuitBreaker]:
        """All registered circuit breakers."""
        return dict(self._breakers)
