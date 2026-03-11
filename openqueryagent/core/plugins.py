"""Entry-point based plugin discovery for adapters, rerankers, and LLM providers.

Third-party packages register plugins via ``pyproject.toml`` entry points::

    [project.entry-points."openqueryagent.adapters"]
    my_adapter = "my_package:MyAdapter"

The agent discovers and registers these plugins automatically at init time.
"""

from __future__ import annotations

import sys
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Entry point group names
_ADAPTER_GROUP = "openqueryagent.adapters"
_RERANKER_GROUP = "openqueryagent.rerankers"
_LLM_PROVIDER_GROUP = "openqueryagent.llm_providers"
_EMBEDDING_PROVIDER_GROUP = "openqueryagent.embedding_providers"


def _load_entry_points(group: str) -> dict[str, Any]:
    """Load all entry points for a given group.

    Returns:
        Dict mapping entry point name → loaded object (class or factory).
    """
    from importlib.metadata import entry_points

    eps = entry_points(group=group)

    plugins: dict[str, Any] = {}
    for ep in eps:
        try:
            plugins[ep.name] = ep.load()
            logger.info("plugin_loaded", group=group, name=ep.name)
        except Exception as exc:
            logger.warning("plugin_load_failed", group=group, name=ep.name, error=str(exc))
    return plugins


class PluginRegistry:
    """Registry of discovered plugins.

    Call :meth:`discover` to scan all entry point groups and populate
    the registry.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, Any] = {}
        self._rerankers: dict[str, Any] = {}
        self._llm_providers: dict[str, Any] = {}
        self._embedding_providers: dict[str, Any] = {}

    def discover(self) -> None:
        """Scan entry points and load all plugins."""
        self._adapters = _load_entry_points(_ADAPTER_GROUP)
        self._rerankers = _load_entry_points(_RERANKER_GROUP)
        self._llm_providers = _load_entry_points(_LLM_PROVIDER_GROUP)
        self._embedding_providers = _load_entry_points(_EMBEDDING_PROVIDER_GROUP)

        total = (
            len(self._adapters)
            + len(self._rerankers)
            + len(self._llm_providers)
            + len(self._embedding_providers)
        )
        if total > 0:
            logger.info(
                "plugins_discovered",
                adapters=list(self._adapters.keys()),
                rerankers=list(self._rerankers.keys()),
                llm_providers=list(self._llm_providers.keys()),
                embedding_providers=list(self._embedding_providers.keys()),
            )

    @property
    def adapters(self) -> dict[str, Any]:
        """Discovered adapter classes, keyed by entry point name."""
        return dict(self._adapters)

    @property
    def rerankers(self) -> dict[str, Any]:
        """Discovered reranker classes."""
        return dict(self._rerankers)

    @property
    def llm_providers(self) -> dict[str, Any]:
        """Discovered LLM provider classes."""
        return dict(self._llm_providers)

    @property
    def embedding_providers(self) -> dict[str, Any]:
        """Discovered embedding provider classes."""
        return dict(self._embedding_providers)

    def get_adapter(self, name: str) -> Any | None:
        """Get an adapter class by name."""
        return self._adapters.get(name)

    def get_reranker(self, name: str) -> Any | None:
        """Get a reranker class by name."""
        return self._rerankers.get(name)
