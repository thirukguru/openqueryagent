"""CLI entry point: ``python -m openqueryagent.server``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and start the uvicorn server."""
    parser = argparse.ArgumentParser(
        prog="openqueryagent-server",
        description="Start the OpenQueryAgent REST API server",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (default: 1)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (optional)")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is required to run the server. "
            "Install with: pip install openqueryagent[server]",
            file=sys.stderr,
        )
        sys.exit(1)

    uvicorn.run(
        "openqueryagent.server.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
