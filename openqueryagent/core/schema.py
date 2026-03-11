"""Schema Inspector — aggregates and caches collection schemas from all adapters.

The SchemaInspector builds a unified SchemaMap from multiple adapters,
caches it with a configurable TTL, and provides schema lookup by
collection name.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from openqueryagent.core.types import CollectionSchema, SchemaMap

if TYPE_CHECKING:
    from openqueryagent.adapters.base import VectorStoreAdapter

logger = structlog.get_logger(__name__)


class SchemaInspector:
    """Aggregates and caches schemas from multiple adapters.

    Args:
        adapters: List of connected VectorStoreAdapter instances.
        cache_ttl_seconds: How long to cache schemas before refreshing.
    """

    def __init__(
        self,
        adapters: list[VectorStoreAdapter],
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        self._adapters = adapters
        self._cache_ttl = cache_ttl_seconds
        self._schema_map: SchemaMap | None = None
        self._last_refresh: float = 0.0

    @property
    def is_stale(self) -> bool:
        """Whether the cached schema map is stale (older than TTL)."""
        if self._schema_map is None:
            return True
        return (time.monotonic() - self._last_refresh) > self._cache_ttl

    async def get_schema_map(self) -> SchemaMap:
        """Get the unified schema map, refreshing if stale.

        Returns:
            SchemaMap with all collections across all adapters.
        """
        if self.is_stale:
            await self.refresh()
        assert self._schema_map is not None
        return self._schema_map

    async def get_schema(self, collection: str) -> CollectionSchema | None:
        """Look up a single collection schema by name.

        Args:
            collection: The collection name.

        Returns:
            CollectionSchema if found, None otherwise.
        """
        schema_map = await self.get_schema_map()
        return schema_map.collections.get(collection)

    async def get_adapter_for_collection(self, collection: str) -> str | None:
        """Get the adapter_id that owns a collection.

        Args:
            collection: The collection name.

        Returns:
            adapter_id string if found, None otherwise.
        """
        schema_map = await self.get_schema_map()
        return schema_map.adapter_mapping.get(collection)

    async def refresh(self) -> SchemaMap:
        """Force a full schema refresh from all adapters.

        Queries all adapters in parallel for their collections and schemas.

        Returns:
            The freshly built SchemaMap.
        """
        logger.info("schema_refresh_start", adapter_count=len(self._adapters))

        collections: dict[str, CollectionSchema] = {}
        adapter_mapping: dict[str, str] = {}

        # Gather collection lists from all adapters in parallel
        adapter_collections: list[tuple[VectorStoreAdapter, list[str]]] = []

        results = await asyncio.gather(
            *[self._get_adapter_collections(a) for a in self._adapters],
            return_exceptions=True,
        )

        for adapter, result in zip(self._adapters, results, strict=False):
            if isinstance(result, BaseException):
                logger.warning(
                    "schema_refresh_adapter_error",
                    adapter_id=adapter.adapter_id,
                    error=str(result),
                )
                continue
            adapter_collections.append((adapter, result))

        # Gather schemas for all collections in parallel
        schema_tasks = []
        task_metadata: list[tuple[str, str]] = []  # (adapter_id, collection_name)

        for adapter, collection_names in adapter_collections:
            for name in collection_names:
                schema_tasks.append(adapter.get_schema(name))
                task_metadata.append((adapter.adapter_id, name))

        schema_results = await asyncio.gather(*schema_tasks, return_exceptions=True)

        for (adapter_id, name), schema_result in zip(task_metadata, schema_results, strict=False):
            if isinstance(schema_result, BaseException):
                logger.warning(
                    "schema_fetch_error",
                    adapter_id=adapter_id,
                    collection=name,
                    error=str(schema_result),
                )
                continue
            collections[name] = schema_result
            adapter_mapping[name] = adapter_id

        self._schema_map = SchemaMap(
            collections=collections,
            adapter_mapping=adapter_mapping,
        )
        self._last_refresh = time.monotonic()

        logger.info(
            "schema_refresh_complete",
            total_collections=len(collections),
            adapters_succeeded=len(adapter_collections),
        )

        return self._schema_map

    def invalidate(self) -> None:
        """Invalidate the schema cache, forcing a refresh on next access."""
        self._schema_map = None
        self._last_refresh = 0.0

    @staticmethod
    async def _get_adapter_collections(
        adapter: VectorStoreAdapter,
    ) -> list[str]:
        """Get collections from a single adapter."""
        return await adapter.get_collections()
