"""
tests/conftest.py — run async tests on asyncio.

Async tests are marked with `pytest.mark.anyio` (the anyio plugin ships with
anyio, which FastAPI and httpx already depend on — no extra package). Each
test gets its own event loop, so everything a test opens (Redis client,
Postgres pools, HTTP client) must be closed in that test; tests/fakes.py
tracks containers for that.
"""

import pytest

from careroute.runtime import ANYIO_BACKEND_OPTIONS


@pytest.fixture
def anyio_backend():
    # Windows' default ProactorEventLoop can't host async psycopg (shared-mode
    # tests). A SelectorEventLoop works everywhere, so it is used on every
    # platform: CI on Linux runs the exact loop setup Windows needs.
    return "asyncio", ANYIO_BACKEND_OPTIONS
