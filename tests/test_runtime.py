"""
tests/test_runtime.py — every test event loop can host async psycopg.

On Windows the default loop (Proactor) can't; these pin that pytest, the
TestClient and the guard all behave, and they run identically on Linux CI.
"""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from careroute import runtime
from careroute.runtime import ANYIO_BACKEND_OPTIONS


def _is_selector(loop) -> bool:
    return isinstance(loop, asyncio.SelectorEventLoop)


@pytest.mark.anyio
async def test_async_tests_run_on_a_selector_loop():
    assert _is_selector(asyncio.get_running_loop())


def test_testclient_runs_on_a_selector_loop():
    app = FastAPI()

    @app.get("/")
    async def which():
        return {"selector": _is_selector(asyncio.get_running_loop())}

    with TestClient(app, backend_options=ANYIO_BACKEND_OPTIONS) as http:
        assert http.get("/").json() == {"selector": True}


@pytest.mark.anyio
async def test_guard_explains_the_windows_fix(monkeypatch):
    """Simulate Windows with the current loop standing in for Proactor."""
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime.asyncio, "ProactorEventLoop",
                        type(asyncio.get_running_loop()), raising=False)
    with pytest.raises(RuntimeError, match="careroute.runtime:new_event_loop"):
        runtime.ensure_psycopg_compatible_loop()


@pytest.mark.anyio
async def test_guard_is_silent_off_windows():
    runtime.ensure_psycopg_compatible_loop()       # Linux / macOS: no-op
