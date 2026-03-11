"""Query router — resolves collections and compiles filters.

Takes a QueryPlan and resolves collection names, validates fields,
and compiles filter expressions into native formats using the
appropriate adapter's FilterCompiler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from openqueryagent.core.types import QueryPlan, SchemaMap

logger = structlog.get_logger(__name__)


class QueryRouter:
    """Routes query plans to the correct adapters with compiled filters.

    Resolves collection names from the SchemaMap, validates that
    collections exist, and compiles any filter expressions.

    Args:
        adapters: Mapping of adapter_id → adapter instance.
        schema_map: The current unified schema map.
    """

    def __init__(
        self,
        adapters: dict[str, Any],
        schema_map: SchemaMap,
    ) -> None:
        self._adapters = adapters
        self._schema_map = schema_map

    def resolve_collection(self, collection_name: str) -> str | None:
        """Resolve a collection name, supporting fuzzy matching.

        Args:
            collection_name: The collection name from the query plan.

        Returns:
            The resolved collection name, or None if not found.
        """
        # Exact match
        if collection_name in self._schema_map.collections:
            return collection_name

        # Case-insensitive match
        lower_name = collection_name.lower()
        for name in self._schema_map.collections:
            if name.lower() == lower_name:
                return name

        # Substring match
        for name in self._schema_map.collections:
            if lower_name in name.lower() or name.lower() in lower_name:
                return name

        return None

    def get_adapter_for_collection(self, collection_name: str) -> Any | None:
        """Get the adapter instance for a given collection.

        Args:
            collection_name: The collection name.

        Returns:
            The adapter instance, or None if not found.
        """
        adapter_id = self._schema_map.adapter_mapping.get(collection_name)
        if adapter_id:
            return self._adapters.get(adapter_id)
        return None

    def compile_filters(
        self,
        collection_name: str,
        filters: Any | None,
    ) -> Any | None:
        """Compile filter expressions using the adapter's FilterCompiler.

        Args:
            collection_name: Target collection.
            filters: Universal FilterExpression or None.

        Returns:
            Native compiled filter or None.
        """
        if filters is None:
            return None

        adapter = self.get_adapter_for_collection(collection_name)
        if adapter is None:
            logger.warning("no_adapter_for_collection", collection=collection_name)
            return None

        schema = self._schema_map.collections.get(collection_name)
        if schema is None:
            return None

        compiler = adapter.get_filter_compiler()
        return compiler.compile(filters, schema)

    def route(self, plan: QueryPlan) -> list[dict[str, Any]]:
        """Route a query plan, resolving collections and preparing execution info.

        Args:
            plan: The query plan to route.

        Returns:
            List of routing info dicts with resolved collection, adapter, and filters.
        """
        routed: list[dict[str, Any]] = []

        for sq in plan.sub_queries:
            resolved_name = self.resolve_collection(sq.collection)
            if resolved_name is None:
                logger.warning("collection_not_found", collection=sq.collection)
                continue

            adapter = self.get_adapter_for_collection(resolved_name)
            if adapter is None:
                logger.warning("adapter_not_found", collection=resolved_name)
                continue

            compiled_filters = self.compile_filters(resolved_name, sq.filters)

            routed.append({
                "sub_query": sq,
                "collection": resolved_name,
                "adapter": adapter,
                "filters": compiled_filters,
            })

        return routed
