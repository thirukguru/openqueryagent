# ---- Build stage ----
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY openqueryagent/ ./openqueryagent/

RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /build/dist

# ---- Runtime stage ----
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/thirukguru/openqueryagent"
LABEL org.opencontainers.image.description="OpenQueryAgent — database-agnostic query agent for vector databases"

WORKDIR /app

# Install the wheel with server extras
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl[server,openai,anthropic,qdrant,pgvector,milvus] \
    && rm -rf /tmp/*.whl

# Non-root user
RUN useradd --create-home --shell /bin/bash oqa
USER oqa

EXPOSE 8000 50051

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/v1/health').raise_for_status()"

ENTRYPOINT ["python", "-m", "openqueryagent.server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
