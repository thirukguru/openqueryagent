"""Tests for circuit breaker state machine."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from openqueryagent.core.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitState
from openqueryagent.core.exceptions import AdapterConnectionError


class TestCircuitBreaker:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_stays_closed_below_threshold(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.on_failure()
        cb.on_failure()
        assert cb.state == CircuitState.CLOSED

    def test_opens_at_threshold(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=3)
        for _ in range(3):
            cb.on_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_rejects_calls(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        with pytest.raises(AdapterConnectionError, match="Circuit breaker OPEN"):
            cb.pre_call()

    def test_transitions_to_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_trial_call(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        cb.on_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.pre_call()  # Should not raise

    def test_half_open_limits_trial_calls(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1, half_open_max_calls=1)
        cb.on_failure()
        time.sleep(0.15)
        cb.pre_call()  # First trial — allowed
        with pytest.raises(AdapterConnectionError, match="HALF_OPEN"):
            cb.pre_call()  # Second trial — rejected

    def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        cb.on_failure()
        time.sleep(0.15)
        cb.pre_call()
        cb.on_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        cb.on_failure()
        time.sleep(0.15)
        cb.pre_call()
        cb.on_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.on_failure()
        cb.on_failure()
        cb.on_success()
        cb.on_failure()
        cb.on_failure()
        # Should still be closed — count reset after success
        assert cb.state == CircuitState.CLOSED

    def test_reset(self) -> None:
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerRegistry:
    def test_creates_breakers_on_demand(self) -> None:
        registry = CircuitBreakerRegistry()
        cb = registry.get("qdrant")
        assert isinstance(cb, CircuitBreaker)

    def test_returns_same_breaker(self) -> None:
        registry = CircuitBreakerRegistry()
        cb1 = registry.get("qdrant")
        cb2 = registry.get("qdrant")
        assert cb1 is cb2

    def test_different_adapters_get_different_breakers(self) -> None:
        registry = CircuitBreakerRegistry()
        cb1 = registry.get("qdrant")
        cb2 = registry.get("pgvector")
        assert cb1 is not cb2

    def test_breakers_property(self) -> None:
        registry = CircuitBreakerRegistry()
        registry.get("a")
        registry.get("b")
        assert set(registry.breakers.keys()) == {"a", "b"}
