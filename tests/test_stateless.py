"""
tests/test_stateless.py — proves replicas are disposable. Real processes, real
Redis, real Postgres, mocked LLM (free).

    REDIS_URL=redis://localhost:6379/13 \\
    DATABASE_URL=postgresql://careroute:careroute@localhost:5432/careroute \\
    python -m pytest tests/test_stateless.py -s

  1. A conversation hops replicas: turn 1 on A, turn 2 on B — B sees the
     history AND the medications added on turn 1.
  2. A replica dies mid-conversation: hard-kill A; the conversation continues
     on B, then on a brand-new C that never saw turns 1-3.
  3. The rate limit is global: requests alternating between two replicas
     share ONE bucket.
  4. Turns of one conversation are serialised: two simultaneous turns on two
     replicas -> one runs, the other gets 409, and none is lost.

Each replica is a separate OS process: nothing can be shared through Python
memory by accident, so anything that works here works across machines.
"""

import concurrent.futures as cf
import os
import subprocess
import sys
import time

import httpx
import pytest

from careroute.config import PROJECT_ROOT
from tests import procs

REDIS_URL = os.getenv("REDIS_URL", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

pytestmark = pytest.mark.skipif(not (REDIS_URL and DATABASE_URL),
                                reason="needs REDIS_URL and DATABASE_URL")


def chat(port, ip, **body):
    return httpx.post(f"http://127.0.0.1:{port}/chat", json=body, timeout=30,
                      headers={"X-Forwarded-For": ip})


def test_replicas_are_disposable():
    import redis
    redis.Redis.from_url(REDIS_URL).flushdb()            # dedicated test DB only
    mport = procs.free_port()
    env = procs.base_env(
        OPENAI_API_KEY="mock", OPENAI_BASE_URL=f"http://127.0.0.1:{mport}/v1",
        USE_REAL_PROVIDERS="", CAREROUTE_REQUIRE_SHARED_STATE="1",
        CAREROUTE_PG_AUTO_MIGRATE="0",                # migrate once, like prod
        CAREROUTE_RATE_PER_MIN="1", CAREROUTE_BURST="5",
        MOCK_LLM_LATENCY_S="1.0", MOCK_LLM_MODE="echo")
    subprocess.run([sys.executable, "-m", "careroute.storage.checkpoints", "migrate"],
                   cwd=PROJECT_ROOT, env=env, check=True, stdout=subprocess.DEVNULL)

    running = {"mock": procs.start(procs.MOCK, mport, env)}
    A, B, C = procs.free_port(), procs.free_port(), procs.free_port()
    try:
        running["A"] = procs.start(procs.APP, A, env)
        running["B"] = procs.start(procs.APP, B, env)

        # 1) hop replicas
        ip = "198.51.100.1"
        t1 = chat(A, ip, message="itchy rash on my arms", lat=12.9352, lng=77.6245,
                  meds="ibuprofen")
        assert t1.status_code == 200, t1.text
        sid = t1.json()["session_id"]
        assert "turns_seen=1" in t1.json()["answer"]
        a2 = chat(B, ip, message="it's spreading", session_id=sid, meds="cetirizine").json()["answer"]
        assert "turns_seen=2" in a2 and "ibuprofen, cetirizine" in a2, a2
        print(f"\nPASS  turn 1 on A, turn 2 on B: B saw history + meds -> {a2!r}")

        # 2) a replica dies (SIGKILL: no graceful shutdown, like a crash)
        running["A"].kill()
        running["A"].wait()
        assert "turns_seen=3" in chat(B, ip, message="also swelling", session_id=sid).json()["answer"]
        running["C"] = procs.start(procs.APP, C, env)             # born just now
        t4 = chat(C, ip, message="what should I do?", session_id=sid)
        assert t4.status_code == 200 and "turns_seen=4" in t4.json()["answer"], t4.text
        print(f"PASS  A killed; turn 3 on B, turn 4 on brand-new C -> {t4.json()['answer']!r}")

        # 3) one bucket across replicas (blank message: 400 AFTER the limiter)
        codes = [chat(B if i % 2 else C, "203.0.113.9", message=" ").status_code
                 for i in range(10)]
        assert codes.count(400) == 5 and codes.count(429) == 5, codes
        print(f"PASS  10 requests alternating B/C: 5 allowed in total -> {codes}")

        # 4) concurrent turns of ONE conversation are serialised
        ip2 = "198.51.100.2"
        sid2 = chat(B, ip2, message="headache", lat=12.97, lng=77.64).json()["session_id"]
        with cf.ThreadPoolExecutor(2) as ex:
            f1 = ex.submit(chat, B, ip2, message="worse now", session_id=sid2)
            time.sleep(0.15)
            f2 = ex.submit(chat, C, ip2, message="and nausea", session_id=sid2)
            got = sorted([f1.result().status_code, f2.result().status_code])
        assert got == [200, 409], got
        after = chat(C, ip2, message="still there?", session_id=sid2).json()
        assert "turns_seen=3" in after["answer"], after
        print(f"PASS  simultaneous turns on B and C -> {got}; next turn sees exactly 3")
    finally:
        procs.stop(*running.values())
