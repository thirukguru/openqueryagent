"""Server configuration loaded from environment variables or config file."""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AdapterEntry(BaseModel):
    """Configuration for a single adapter in the server config."""

    type: str
    url: str | None = None
    dsn: str | None = None
    api_key: str | None = None
    region: str | None = None
    bucket: str | None = None
    host: str | None = None
    port: int | None = None
    collections: list[str] = Field(default_factory=list)
    indexes: list[str] = Field(default_factory=list)


class ServerConfig(BaseSettings):
    """Server configuration with env-var support.

    All settings can be overridden via ``OQA_`` prefixed environment
    variables (e.g. ``OQA_HOST=0.0.0.0``).
    """

    model_config = {"env_prefix": "OQA_"}

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    grpc_port: int = 50051
    workers: int = 1

    # Security
    api_key: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Rate limiting  (format: "<count>/<period>" e.g. "100/minute")
    rate_limit: str | None = None

    # LLM
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None

    # Embedding
    embedding_provider: str | None = None
    embedding_model: str | None = None

    # Adapters (populated from config.yaml; env-based wiring uses
    # individual OQA_QDRANT_URL / OQA_PGVECTOR_DSN / … vars).
    qdrant_url: str | None = None
    pgvector_dsn: str | None = None
    milvus_url: str | None = None
    weaviate_url: str | None = None
    pinecone_api_key: str | None = None
    chroma_url: str | None = None
    elasticsearch_url: str | None = None
    s3vectors_region: str | None = None
    s3vectors_bucket: str | None = None
