"""Tests for configuration types — defaults and serialization."""

from __future__ import annotations

from openqueryagent.core.config import AgentConfig, ExecutorConfig, RetryConfig


class TestRetryConfig:
    def test_defaults(self) -> None:
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.backoff_base == 0.5
        assert config.backoff_max == 30.0
        assert config.retry_on_timeout is True

    def test_custom_values(self) -> None:
        config = RetryConfig(max_attempts=5, backoff_base=1.0, backoff_max=60.0)
        data = config.model_dump()
        restored = RetryConfig.model_validate(data)
        assert restored.max_attempts == 5
        assert restored.backoff_max == 60.0


class TestExecutorConfig:
    def test_defaults(self) -> None:
        config = ExecutorConfig()
        assert config.max_concurrent == 10
        assert config.timeout_per_query == 30.0
        assert config.max_aggregation_scroll == 100_000
        assert config.retry_config.max_attempts == 3

    def test_round_trip(self) -> None:
        config = ExecutorConfig(
            max_concurrent=5,
            timeout_per_query=15.0,
            retry_config=RetryConfig(max_attempts=5),
            max_aggregation_scroll=50_000,
        )
        json_str = config.model_dump_json()
        restored = ExecutorConfig.model_validate_json(json_str)
        assert restored.max_concurrent == 5
        assert restored.retry_config.max_attempts == 5
        assert restored.max_aggregation_scroll == 50_000


class TestAgentConfig:
    def test_defaults(self) -> None:
        config = AgentConfig()
        assert config.max_sub_queries == 5
        assert config.default_limit == 10
        assert config.timeout_seconds == 60.0
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 300
        assert config.enable_tracing is True
        assert config.log_level == "INFO"
        assert config.max_concurrent_queries == 10
        assert config.retry_config.max_attempts == 3
        assert config.executor_config.max_concurrent == 10

    def test_nested_round_trip(self) -> None:
        config = AgentConfig(
            max_sub_queries=3,
            timeout_seconds=30.0,
            enable_caching=False,
            log_level="DEBUG",
            retry_config=RetryConfig(max_attempts=5),
            executor_config=ExecutorConfig(max_concurrent=20),
        )
        json_str = config.model_dump_json()
        restored = AgentConfig.model_validate_json(json_str)
        assert restored.max_sub_queries == 3
        assert restored.enable_caching is False
        assert restored.log_level == "DEBUG"
        assert restored.retry_config.max_attempts == 5
        assert restored.executor_config.max_concurrent == 20
