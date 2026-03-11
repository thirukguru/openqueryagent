"""Rule-based query planner.

A planner that uses pattern matching and keyword analysis to route
queries without LLM calls. Good for cost-sensitive deployments or
when queries follow predictable patterns.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from openqueryagent.core.types import (
    AggregationQuery,
    QueryIntent,
    QueryPlan,
    SchemaMap,
    SearchType,
    SubQuery,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Aggregation keywords → operations
# ---------------------------------------------------------------------------

_AGG_KEYWORDS: dict[str, str] = {
    "how many": "count",
    "count": "count",
    "total number": "count",
    "average": "avg",
    "mean": "avg",
    "sum": "sum",
    "total": "sum",
    "minimum": "min",
    "lowest": "min",
    "maximum": "max",
    "highest": "max",
}


class RuleBasedPlanner:
    """Rule-based query planner using keyword analysis.

    Routes queries to collections based on keyword matching against
    collection names and property names. Detects aggregation intent
    via keyword patterns.

    Args:
        default_collection: Fallback collection when no match is found.
        default_search_type: Default search type.
        default_limit: Default result limit.
    """

    def __init__(
        self,
        default_collection: str = "",
        default_search_type: SearchType = SearchType.HYBRID,
        default_limit: int = 10,
    ) -> None:
        self._default_collection = default_collection
        self._default_search_type = default_search_type
        self._default_limit = default_limit

    async def plan(
        self,
        query: str,
        schema_map: SchemaMap,
        history: list[Any] | None = None,
    ) -> QueryPlan:
        """Plan a query using rule-based analysis.

        Args:
            query: User's natural language query.
            schema_map: Available collection schemas.
            history: Ignored in rule-based planner.

        Returns:
            A QueryPlan with sub-queries.
        """
        query_lower = query.lower()

        # Detect aggregation intent
        agg_op = self._detect_aggregation(query_lower)
        intent = QueryIntent.AGGREGATE if agg_op else QueryIntent.SEARCH

        # Find matching collection(s)
        matched = self._match_collections(query_lower, schema_map)
        collection = matched[0] if matched else self._default_collection
        if not collection and schema_map.collections:
            collection = next(iter(schema_map.collections))

        # Detect search type
        search_type = self._detect_search_type(query_lower)

        # Build aggregation if detected
        aggregation = None
        if agg_op:
            agg_field = self._detect_agg_field(query_lower, schema_map, collection)
            aggregation = AggregationQuery(operation=agg_op, field=agg_field)

        sub_queries = [
            SubQuery(
                id="q1",
                collection=collection,
                query_text=query,
                search_type=search_type,
                limit=self._default_limit,
                aggregation=aggregation,
            ),
        ]

        # Multi-collection support: if query mentions multiple collections
        if len(matched) > 1:
            for i, coll in enumerate(matched[1:3], start=2):  # Max 3 collections
                sub_queries.append(SubQuery(
                    id=f"q{i}",
                    collection=coll,
                    query_text=query,
                    search_type=search_type,
                    limit=self._default_limit,
                ))

        plan = QueryPlan(
            original_query=query,
            intent=intent,
            sub_queries=sub_queries,
            reasoning=f"Rule-based: matched {len(matched)} collection(s), "
            f"search_type={search_type}, agg={agg_op or 'none'}",
            requires_synthesis=len(sub_queries) > 1 or intent != QueryIntent.AGGREGATE,
        )
        logger.info(
            "rule_based_plan",
            collections=matched,
            intent=intent,
            agg=agg_op,
        )
        return plan

    @staticmethod
    def _detect_aggregation(query_lower: str) -> str | None:
        """Detect aggregation operation from keywords."""
        for keyword, op in _AGG_KEYWORDS.items():
            if keyword in query_lower:
                return op
        return None

    @staticmethod
    def _match_collections(query_lower: str, schema_map: SchemaMap) -> list[str]:
        """Match query text to collection names."""
        matched: list[str] = []
        for name in schema_map.collections:
            # Check if collection name or synonyms appear in query
            name_words = re.split(r"[_\-\s]+", name.lower())
            for word in name_words:
                if len(word) >= 3 and word in query_lower:
                    matched.append(name)
                    break
        return matched

    @staticmethod
    def _detect_search_type(query_lower: str) -> SearchType:
        """Detect search type from query keywords."""
        exact_keywords = {"exact", "precisely", "filter", "where", "equals"}
        if any(kw in query_lower for kw in exact_keywords):
            return SearchType.KEYWORD
        semantic_keywords = {"similar to", "like", "related to", "resembles"}
        if any(kw in query_lower for kw in semantic_keywords):
            return SearchType.VECTOR
        return SearchType.HYBRID

    @staticmethod
    def _detect_agg_field(
        query_lower: str,
        schema_map: SchemaMap,
        collection: str,
    ) -> str | None:
        """Try to detect which field the aggregation targets."""
        schema = schema_map.collections.get(collection)
        if not schema:
            return None
        for prop in schema.properties:
            if prop.name.lower() in query_lower:
                return prop.name
        return None
