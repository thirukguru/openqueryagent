"""Tests for plugin discovery system."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from openqueryagent.core.plugins import PluginRegistry, _load_entry_points

_MOCK_TARGET = "importlib.metadata.entry_points"


class TestLoadEntryPoints:
    def test_loads_available_plugins(self) -> None:
        mock_ep = MagicMock()
        mock_ep.name = "test_adapter"
        mock_ep.load.return_value = MagicMock

        with patch(_MOCK_TARGET, return_value=[mock_ep]):
            plugins = _load_entry_points("openqueryagent.adapters")

        assert "test_adapter" in plugins

    def test_handles_load_failure(self) -> None:
        mock_ep = MagicMock()
        mock_ep.name = "bad_plugin"
        mock_ep.load.side_effect = ImportError("no module")

        with patch(_MOCK_TARGET, return_value=[mock_ep]):
            plugins = _load_entry_points("openqueryagent.adapters")

        assert "bad_plugin" not in plugins

    def test_empty_group(self) -> None:
        with patch(_MOCK_TARGET, return_value=[]):
            plugins = _load_entry_points("openqueryagent.adapters")
        assert plugins == {}


class TestPluginRegistry:
    def test_discover_populates_registry(self) -> None:
        mock_ep = MagicMock()
        mock_ep.name = "custom_adapter"
        mock_ep.load.return_value = type("CustomAdapter", (), {})

        with patch(_MOCK_TARGET, return_value=[mock_ep]):
            registry = PluginRegistry()
            registry.discover()

        assert "custom_adapter" in registry.adapters

    def test_empty_discover(self) -> None:
        with patch(_MOCK_TARGET, return_value=[]):
            registry = PluginRegistry()
            registry.discover()

        assert registry.adapters == {}
        assert registry.rerankers == {}
        assert registry.llm_providers == {}

    def test_get_adapter(self) -> None:
        mock_ep = MagicMock()
        mock_ep.name = "my_adapter"
        mock_cls = type("MyAdapter", (), {})
        mock_ep.load.return_value = mock_cls

        with patch(_MOCK_TARGET, return_value=[mock_ep]):
            registry = PluginRegistry()
            registry.discover()

        assert registry.get_adapter("my_adapter") is mock_cls
        assert registry.get_adapter("nonexistent") is None
