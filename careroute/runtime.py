"""
careroute/runtime.py — event-loop compatibility (Windows).

psycopg's async mode (the Postgres checkpointer and the PostGIS pool) cannot
run on Windows' default ProactorEventLoop; it needs a SelectorEventLoop.
Linux — Docker, CI, Azure — is unaffected, and so is memory mode anywhere.

Running shared mode directly on Windows:

    uvicorn --factory careroute.api.app:create_app --loop careroute.runtime:new_event_loop

Without that flag, the first Postgres use raises the explanatory error below
instead of psycopg's terse one.

Tests: every place that starts an event loop must use new_event_loop —
pytest's anyio backend (tests/conftest.py), Starlette's TestClient
(tests/test_api.py) and the replica subprocesses (tests/procs.py). They all
take ANYIO_BACKEND_OPTIONS / the --loop flag UNCONDITIONALLY, so the code path
Windows needs is the same one CI exercises on Linux.
"""

from __future__ import annotations

import asyncio
import sys

UVICORN_HINT = ("uvicorn --factory careroute.api.app:create_app "
                "--loop careroute.runtime:new_event_loop")


def new_event_loop() -> asyncio.AbstractEventLoop:
    """A loop psycopg's async mode accepts on every platform. uvicorn calls
    this when started with --loop careroute.runtime:new_event_loop."""
    return asyncio.SelectorEventLoop()


#: For anything built on anyio (pytest's anyio plugin, Starlette TestClient).
ANYIO_BACKEND_OPTIONS = {"loop_factory": new_event_loop}


def ensure_psycopg_compatible_loop() -> None:
    """Raise a clear error if the running loop can't host async psycopg."""
    if sys.platform != "win32":
        return
    loop = asyncio.get_running_loop()
    proactor = getattr(asyncio, "ProactorEventLoop", None)
    if proactor is not None and isinstance(loop, proactor):
        raise RuntimeError(
            "Postgres (DATABASE_URL / GEO_DATABASE_URL) needs a SelectorEventLoop "
            "on Windows, but this process runs the default ProactorEventLoop. "
            f"Start the server with:\n    {UVICORN_HINT}\n"
            "or run it in Docker (docker compose up).")
