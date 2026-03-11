# Contributing to OpenQueryAgent

Thank you for your interest in contributing! This guide covers how to add new components, code style, and the PR process.

## Development Setup

```bash
git clone https://github.com/thirukguru/openqueryagent.git
cd openqueryagent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,server,observability]"
pytest tests/ -v
```

## Adding a New Adapter

1. **Create** `openqueryagent/adapters/my_db.py` inheriting from `BaseAdapter`
2. **Implement** all abstract methods: `connect()`, `disconnect()`, `health_check()`, `search()`, `aggregate()`, `list_collections()`, `get_schema()`
3. **Create** `openqueryagent/adapters/my_db_filters.py` with a `FilterCompiler` subclass
4. **Add** an optional dependency group in `pyproject.toml`
5. **Register** via entry point (optional, for plugin-based discovery):
   ```toml
   [project.entry-points."openqueryagent.adapters"]
   my_db = "openqueryagent.adapters.my_db:MyDbAdapter"
   ```
6. **Write tests** in `tests/test_my_db_adapter.py`

## Adding a New Reranker

1. **Create** `openqueryagent/core/rerankers/my_reranker.py` implementing the `Reranker` protocol
2. **Register** via entry point:
   ```toml
   [project.entry-points."openqueryagent.rerankers"]
   my_reranker = "openqueryagent.core.rerankers.my_reranker:MyReranker"
   ```

## Adding a New LLM Provider

1. **Create** `openqueryagent/llm/my_provider.py` inheriting from `BaseLLMProvider`
2. **Add** optional dependency in `pyproject.toml`
3. **Write tests** with mocked API responses

## Code Style

- **Formatter & linter**: [Ruff](https://docs.astral.sh/ruff/) (`ruff check --fix . && ruff format .`)
- **Type checking**: [mypy](http://mypy-lang.org/) (`mypy openqueryagent/`)
- **Line length**: 100 characters
- **Docstrings**: Google style
- **Imports**: `from __future__ import annotations` at top of every module

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_my_adapter.py -v

# Run with coverage
pytest tests/ --cov=openqueryagent --cov-report=term-missing
```

## Pull Request Process

1. Fork the repo and create a feature branch from `main`
2. Keep commits focused and atomic
3. Add tests for any new functionality
4. Ensure all tests pass and linting is clean
5. Update documentation if needed
6. Submit PR with a clear description of what and why

## Reporting Issues

Use the GitHub issue templates:
- **Bug Report** — reproducible bugs with environment details
- **Feature Request** — new feature proposals
- **New Adapter** — proposals for new database adapters
