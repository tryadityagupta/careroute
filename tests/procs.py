"""tests/procs.py — start real uvicorn processes for multi-process tests."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import httpx

from careroute.config import PROJECT_ROOT

APP = ["--factory", "careroute.api.app:create_app"]
MOCK = ["mock_llm.app:app"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(target: list[str], port: int, env: dict, probe: str = "/healthz") -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", *target, "--port", str(port),
                          "--log-level", "warning"],
                         cwd=PROJECT_ROOT, env=env, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE)
    for _ in range(150):
        try:
            if httpx.get(f"http://127.0.0.1:{port}{probe}", timeout=1).status_code == 200:
                return p
        except httpx.HTTPError:
            pass
        if p.poll() is not None:
            raise RuntimeError(f"{target} on :{port} died: {p.stderr.read().decode()[-800:]}")
        time.sleep(0.2)
    raise RuntimeError(f"{target} on :{port} never became healthy")


def stop(*procs: subprocess.Popen) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()


def base_env(**extra) -> dict:
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    env.update({k: str(v) for k, v in extra.items()})
    return env
