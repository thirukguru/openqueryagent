"""Configuration types for OpenQueryAgent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_attempts: int = 3
    backoff_base: float = 0.5
    backoff_max: float = 30.0
    retry_on_timeout: bool = True


class ExecutorConfig(BaseModel):
    """Configuration for the query executor."""

    max_concurrent: int = 10
    timeout_per_query: float = 30.0
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    max_aggregation_scroll: int = 100_000


class AgentConfig(BaseModel):
    """Top-level configuration for the QueryAgent."""

    max_sub_queries: int = 5
    default_limit: int = 10
    timeout_seconds: float = 60.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_tracing: bool = True
    log_level: str = "INFO"
    max_concurrent_queries: int = 10
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    executor_config: ExecutorConfig = Field(default_factory=ExecutorConfig)
