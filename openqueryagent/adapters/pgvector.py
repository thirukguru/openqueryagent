"""pgvector adapter — connects to PostgreSQL with pgvector via asyncpg.

Implements the VectorStoreAdapter protocol for PostgreSQL with pgvector extension.
Requires ``asyncpg`` (install with ``pip install openqueryagent[pgvector]``).
"""

from __future__ import annotations

import re
import time
from typing import Any

import structlog

from openqueryagent.adapters.base import (
    ConnectionConfig,
    FilterCompiler,
    HealthStatus,
)
from openqueryagent.adapters.pgvector_filters import PgvectorFilterCompiler, PgvectorFilterResult
from openqueryagent.core.exceptions import (
    AdapterConnectionError,
    AdapterQueryError,
    SchemaError,
)
from openqueryagent.core.types import (
    AggregationQuery,
    AggregationResult,
    CollectionSchema,
    DataType,
    DistanceMetric,
    Document,
    PropertySchema,
    SearchResult,
    SearchType,
    VectorConfig,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Connection Config
# ---------------------------------------------------------------------------


class PgvectorConnectionConfig(ConnectionConfig):
    """Connection configuration for PostgreSQL + pgvector."""

    dsn: str = "postgresql://localhost:5432/vectordb"
    min_pool_size: int = 2
    max_pool_size: int = 10


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

# PostgreSQL column type → DataType mapping
_PG_TYPE_MAP: dict[str, DataType] = {
    "text": DataType.TEXT,
    "varchar": DataType.TEXT,
    "character varying": DataType.TEXT,
    "integer": DataType.INT,
    "bigint": DataType.INT,
    "smallint": DataType.INT,
    "real": DataType.FLOAT,
    "double precision": DataType.FLOAT,
    "numeric": DataType.FLOAT,
    "boolean": DataType.BOOL,
    "timestamp": DataType.DATE,
    "timestamp with time zone": DataType.DATE,
    "timestamp without time zone": DataType.DATE,
    "date": DataType.DATE,
    "json": DataType.TEXT,
    "jsonb": DataType.TEXT,
    "vector": DataType.FLOAT_ARRAY,
    "uuid": DataType.TEXT,
}


class PgvectorAdapter:
    """PostgreSQL + pgvector adapter.

    Uses ``asyncpg`` connection pool for async operations.
    Supports vector search (via <=> operator), keyword search
    (via tsvector/tsquery), and hybrid search (via CTE-based RRF).
    """

    def __init__(self, adapter_id: str = "pgvector") -> None:
        self._adapter_id = adapter_id
        self._pool: Any = None
        self._filter_compiler = PgvectorFilterCompiler()

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def adapter_name(self) -> str:
        return "pgvector"

    @property
    def supports_native_aggregation(self) -> bool:
        return True  # PostgreSQL has full SQL aggregation

    async def connect(self, config: ConnectionConfig) -> None:
        """Connect to PostgreSQL and create connection pool."""
        try:
            import asyncpg
        except ImportError as e:
            raise AdapterConnectionError(
                "asyncpg is not installed. Install with: pip install openqueryagent[pgvector]",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

        if not isinstance(config, PgvectorConnectionConfig):
            config = PgvectorConnectionConfig(**config.model_dump())

        try:
            self._pool = await asyncpg.create_pool(
                dsn=config.dsn,
                min_size=config.min_pool_size,
                max_size=config.max_pool_size,
                timeout=config.timeout_seconds,
            )
            logger.info("pgvector_connected", dsn=_redact_dsn(config.dsn))
        except Exception as e:
            raise AdapterConnectionError(
                f"Failed to connect to PostgreSQL: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            ) from e

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("pgvector_disconnected")

    async def health_check(self) -> HealthStatus:
        """Check PostgreSQL health."""
        self._ensure_connected()
        start = time.monotonic()
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            latency = (time.monotonic() - start) * 1000
            return HealthStatus(
                healthy=True,
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return HealthStatus(
                healthy=False,
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                message=str(e),
                latency_ms=latency,
            )

    async def get_collections(self) -> list[str]:
        """List tables that have vector columns."""
        self._ensure_connected()
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT DISTINCT table_name
                    FROM information_schema.columns
                    WHERE data_type = 'USER-DEFINED'
                      AND udt_name = 'vector'
                      AND table_schema = 'public'
                    ORDER BY table_name
                """)
                return [row["table_name"] for row in rows]
        except Exception as e:
            raise AdapterQueryError(
                f"Failed to list collections: {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection="",
            ) from e

    async def get_schema(self, collection: str) -> CollectionSchema:
        """Introspect table schema from information_schema."""
        self._ensure_connected()
        try:
            async with self._pool.acquire() as conn:
                # Get column info
                columns = await conn.fetch("""
                    SELECT column_name, data_type, udt_name,
                           character_maximum_length, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = $1 AND table_schema = 'public'
                    ORDER BY ordinal_position
                """, collection)

                if not columns:
                    raise SchemaError(
                        f"Table '{collection}' not found or has no columns",
                        collection=collection,
                        adapter_id=self._adapter_id,
                    )

                # Get vector dimensions
                vector_dim = await conn.fetchval("""
                    SELECT atttypmod
                    FROM pg_attribute
                    WHERE attrelid = $1::regclass
                      AND atttypid = (SELECT oid FROM pg_type WHERE typname = 'vector')
                    LIMIT 1
                """, collection)

                properties: list[PropertySchema] = []
                vector_config: VectorConfig | None = None

                for col in columns:
                    col_name = col["column_name"]
                    data_type_str = col["data_type"].lower()
                    udt_name = col["udt_name"].lower()

                    if udt_name == "vector":
                        dim = vector_dim if vector_dim and vector_dim > 0 else 0
                        vector_config = VectorConfig(
                            dimensions=dim,
                            distance_metric=DistanceMetric.COSINE,  # Default; detect from index
                        )
                        continue

                    dt = _PG_TYPE_MAP.get(data_type_str, _PG_TYPE_MAP.get(udt_name, DataType.TEXT))
                    properties.append(PropertySchema(
                        name=col_name,
                        data_type=dt,
                        filterable=True,
                    ))

                return CollectionSchema(
                    name=collection,
                    adapter_id=self._adapter_id,
                    vector_config=vector_config,
                    properties=properties,
                )
        except SchemaError:
            raise
        except Exception as e:
            raise SchemaError(
                f"Failed to get schema for '{collection}': {e}",
                collection=collection,
                adapter_id=self._adapter_id,
            ) from e

    async def search(
        self,
        collection: str,
        query_vector: list[float] | None = None,
        query_text: str | None = None,
        filters: Any | None = None,
        limit: int = 10,
        offset: int = 0,
        search_type: SearchType = SearchType.HYBRID,
        search_params: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Execute search on PostgreSQL table."""
        self._ensure_connected()
        try:
            # Validate and clamp limit/offset
            limit = max(1, min(limit, 1000))
            offset = max(0, min(offset, 100_000))

            # Validate column names against allowlist
            embedding_col = _validate_identifier(
                (search_params or {}).get("embedding_column", "embedding"),
            )
            content_col = _validate_identifier(
                (search_params or {}).get("content_column", "content"),
            )
            search_vector_col = _validate_identifier(
                (search_params or {}).get("search_vector_column", "search_vector"),
            )
            safe_collection = _validate_identifier(collection)

            where_clause = ""
            params: list[Any] = []

            if isinstance(filters, PgvectorFilterResult):
                where_clause = f"WHERE {filters.sql}"
                params = list(filters.params)

            param_offset = len(params)

            if search_type == SearchType.VECTOR and query_vector:
                params.append(str(query_vector))
                vec_param = f"${param_offset + 1}"
                params.append(limit)
                params.append(offset)
                sql = f"""
                    SELECT *, ({embedding_col} <=> {vec_param}::vector) AS _distance
                    FROM "{safe_collection}"
                    {where_clause}
                    ORDER BY {embedding_col} <=> {vec_param}::vector
                    LIMIT ${param_offset + 2} OFFSET ${param_offset + 3}
                """
            elif search_type == SearchType.KEYWORD and query_text:
                params.append(query_text)
                text_param = f"${param_offset + 1}"
                where_prefix = f"{where_clause.replace('WHERE ', '')} AND" if where_clause else ""
                params.append(limit)
                params.append(offset)
                sql = f"""
                    SELECT *, ts_rank({search_vector_col}, plainto_tsquery({text_param})) AS _rank
                    FROM "{safe_collection}"
                    WHERE {where_prefix}
                     {search_vector_col} @@ plainto_tsquery({text_param})
                    ORDER BY _rank DESC
                    LIMIT ${param_offset + 2} OFFSET ${param_offset + 3}
                """
            elif search_type == SearchType.HYBRID and query_vector and query_text:
                # CTE-based Reciprocal Rank Fusion
                params.append(str(query_vector))
                params.append(query_text)
                vec_param = f"${param_offset + 1}"
                text_param = f"${param_offset + 2}"
                rrf_k = 60
                rrf_limit = limit * 3
                where_prefix = f"{where_clause.replace('WHERE ', '')} AND" if where_clause else ""
                params.append(limit)
                params.append(offset)
                sql = f"""
                    WITH vector_results AS (
                        SELECT *, ROW_NUMBER() OVER (
                            ORDER BY {embedding_col} <=> {vec_param}::vector
                        ) AS vec_rank
                        FROM "{safe_collection}"
                        {where_clause}
                        ORDER BY {embedding_col} <=> {vec_param}::vector
                        LIMIT {rrf_limit}
                    ),
                    keyword_results AS (
                        SELECT *, ROW_NUMBER() OVER (
                            ORDER BY ts_rank({search_vector_col}, plainto_tsquery({text_param})) DESC
                        ) AS kw_rank
                        FROM "{safe_collection}"
                        WHERE {where_prefix}
                         {search_vector_col} @@ plainto_tsquery({text_param})
                        LIMIT {rrf_limit}
                    )
                    SELECT v.*,
                           (1.0 / ({rrf_k} + COALESCE(v.vec_rank, {rrf_limit}))
                          + 1.0 / ({rrf_k} + COALESCE(k.kw_rank, {rrf_limit}))) AS _rrf_score
                    FROM vector_results v
                    FULL OUTER JOIN keyword_results k ON v.id = k.id
                    ORDER BY _rrf_score DESC
                    LIMIT ${param_offset + 3} OFFSET ${param_offset + 4}
                """
            else:
                # Fallback to vector search if vector available
                if query_vector:
                    params.append(str(query_vector))
                    vec_param = f"${param_offset + 1}"
                    params.append(limit)
                    params.append(offset)
                    sql = f"""
                        SELECT *, ({embedding_col} <=> {vec_param}::vector) AS _distance
                        FROM "{safe_collection}"
                        {where_clause}
                        ORDER BY {embedding_col} <=> {vec_param}::vector
                        LIMIT ${param_offset + 2} OFFSET ${param_offset + 3}
                    """
                else:
                    params.append(limit)
                    params.append(offset)
                    sql = f"""
                        SELECT *
                        FROM "{safe_collection}"
                        {where_clause}
                        LIMIT ${param_offset + 1} OFFSET ${param_offset + 2}
                    """

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)

            documents: list[Document] = []
            for row in rows:
                row_dict = dict(row)
                doc_id = str(row_dict.pop("id", ""))
                content = str(row_dict.pop(content_col, ""))
                # Remove internal columns
                for key in ("_distance", "_rank", "_rrf_score", "vec_rank", "kw_rank",
                            embedding_col, search_vector_col):
                    row_dict.pop(key, None)
                score = row.get("_rrf_score") or (1.0 - row.get("_distance", 1.0))
                documents.append(Document(
                    id=doc_id,
                    content=content,
                    metadata=row_dict,
                    score=float(score) if score is not None else 0.0,
                ))

            return SearchResult(documents=documents, total_count=len(documents))
        except Exception as e:
            raise AdapterQueryError(
                f"Search failed on '{collection}': {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection=collection,
            ) from e

    async def aggregate(
        self,
        collection: str,
        aggregation: AggregationQuery,
        filters: Any | None = None,
    ) -> AggregationResult:
        """Execute native SQL aggregation."""
        self._ensure_connected()
        try:
            where_clause = ""
            params: list[Any] = []
            if isinstance(filters, PgvectorFilterResult):
                where_clause = f"WHERE {filters.sql}"
                params = list(filters.params)

            field = f'"{aggregation.field}"' if aggregation.field else "*"
            op = aggregation.operation.lower()

            # Map operation to SQL function
            sql_func = {
                "count": f"COUNT({field})",
                "sum": f"SUM({field})",
                "avg": f"AVG({field})",
                "min": f"MIN({field})",
                "max": f"MAX({field})",
            }.get(op, f"COUNT({field})")

            sql = f'SELECT {sql_func} AS result FROM "{collection}" {where_clause}'

            async with self._pool.acquire() as conn:
                result = await conn.fetchval(sql, *params)

            return AggregationResult(
                values={op: result},
                is_approximate=False,
            )
        except Exception as e:
            raise AdapterQueryError(
                f"Aggregation failed on '{collection}': {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection=collection,
            ) from e

    async def get_by_ids(self, collection: str, ids: list[str]) -> list[Document]:
        """Retrieve documents by ID from PostgreSQL."""
        self._ensure_connected()
        try:
            safe_collection = _validate_identifier(collection)
            sql = f'SELECT * FROM "{safe_collection}" WHERE id = ANY($1)'
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, ids)

            return [
                Document(
                    id=str(dict(row).pop("id", "")),
                    content=str(dict(row).get("content", "")),
                    metadata={
                        k: v for k, v in dict(row).items()
                        if k not in ("id", "content", "embedding")
                    },
                )
                for row in rows
            ]
        except Exception as e:
            raise AdapterQueryError(
                f"get_by_ids failed on '{collection}': {e}",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
                collection=collection,
            ) from e

    def get_filter_compiler(self) -> FilterCompiler:
        """Get the pgvector filter compiler."""
        return self._filter_compiler

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if self._pool is None:
            raise AdapterConnectionError(
                "Not connected to PostgreSQL. Call connect() first.",
                adapter_id=self._adapter_id,
                adapter_name=self.adapter_name,
            )


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_identifier(name: str) -> str:
    """Validate and sanitize a SQL identifier (table/column name).

    Rejects any name that doesn't match ``[a-zA-Z_][a-zA-Z0-9_]*``.
    """
    cleaned = name.replace('"', "")
    if not _IDENTIFIER_RE.match(cleaned):
        msg = f"Invalid SQL identifier: {cleaned!r}"
        raise AdapterQueryError(
            msg,
            adapter_id="pgvector",
            adapter_name="pgvector",
            collection="",
        )
    return cleaned


def _redact_dsn(dsn: str) -> str:
    """Redact password from a PostgreSQL DSN."""
    return re.sub(r"://([^:]+):[^@]+@", r"://\1:***@", dsn)
