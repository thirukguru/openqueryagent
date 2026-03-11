"""OpenQueryAgent REST API server.

Provides a FastAPI-based HTTP server wrapping the QueryAgent pipeline.
"""

from openqueryagent.server.api import create_app

__all__ = ["create_app"]
