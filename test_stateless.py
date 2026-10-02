"""
test_stateless.py — proves replicas are disposable. Real processes, real Redis,
real Postgres, mocked LLM (free).

    REDIS_URL=redis://localhost:6379/13 \\
    DATABASE_URL=postgresql://careroute:careroute@localhost:5432/careroute \\
    python test_stateless.py

What it shows (each is a sentence you can say in an interview):
  1. A conversation can hop replicas: turn 1 on A, turn 2 on B — B sees the
     full history AND the medications added on turn 1.
  2. A replica can die mid-conversation: hard-kill A, the conversation continues
     on B, then on a brand-new replica C that never saw turns 1-3.
  3. The rate limit is global: requests alternating between two replicas
     share ONE bucket (5 allowed in total, not 5 per replica).
  4. Turns of one conversation are serialised: two simultaneous turns of the
     same session on two replicas -> one runs, the other gets 409.

Each replica is a separate OS process, so nothing can be shared through
Python memory by accident; anything that works here works across machines.
"""

import concurrent.futures as cf
import os
import subprocess
import sys
import time

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
MOCK_PORT = 9100
ENV = {
    **os.environ,
    "OPENAI_API_KEY": "mock",
    "OPENAI_BASE_URL": f"http://127.0.0.1:{MOCK_PORT}/v1",
    "OPENAI_API_BASE": f"http://127.0.0.1:{MOCK_PORT}/v1",
    "USE_REAL_PROVIDERS": "",                 # offline dummy provider data
    "CAREROUTE_REQUIRE_SHARED_STATE": "1",    # boot fails if misconfigured
    "CAREROUTE_PG_AUTO_MIGRATE": "0",         # migrate once, like prod (below)
    "CAREROUTE_RATE_PER_MIN": "1",
    "CAREROUTE_BURST": "5",
    "MOCK_LLM_LATENCY_S": "1.0",
    # Echo mode: replies "turns_seen=N | meds=..." so we can see whether
    # conversation history survived a replica hop.
    "MOCK_LLM_MODE": "echo",
}


def start(module_app: str, port: int) -> subprocess.Popen:
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", module_app, "--port", str(port),
         "--log-level", "warning"],
        cwd=HERE, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{port}"
    probe = "/healthz" if module_app.startswith("server") else "/docs"
    for _ in range(100):
        try:
            if httpx.get(url + probe, timeout=1).status_code == 200:
                return p
        except httpx.HTTPError:
            pass
        if p.poll() is not None:
            raise RuntimeError(f"{module_app} on :{port} died: "
                               f"{p.stderr.read().decode()[-800:]}")
        time.sleep(0.2)
    raise RuntimeError(f"{module_app} on :{port} never became healthy")


def chat(port, ip, **body):
    return httpx.post(f"http://127.0.0.1:{port}/chat", json=body, timeout=30,
                      headers={"X-Forwarded-For": ip})


def main():
    if not (os.getenv("REDIS_URL") and os.getenv("DATABASE_URL")):
        sys.exit("Set REDIS_URL and DATABASE_URL to run this test.")

    import redis
    r = redis.Redis.from_url(os.environ["REDIS_URL"])
    r.flushdb()                                   # dedicated test DB only
    subprocess.run([sys.executable, "checkpoints.py", "migrate"], cwd=HERE,
                   env=ENV, check=True, stdout=subprocess.DEVNULL)

    procs = {"mock": start("mock_llm:app", MOCK_PORT)}
    A, B, C = 8101, 8102, 8103
    try:
        procs["A"] = start("server:app", A)
        procs["B"] = start("server:app", B)

        # 1) Conversation hops replicas ------------------------------------
        ip = "198.51.100.1"
        t1 = chat(A, ip, message="itchy rash on my arms", lat=12.9352,
                  lng=77.6245, meds="ibuprofen")
        assert t1.status_code == 200, t1.text
        sid = t1.json()["session_id"]
        assert "turns_seen=1" in t1.json()["answer"], t1.json()

        t2 = chat(B, ip, message="it's spreading", session_id=sid,
                  meds="cetirizine")
        a2 = t2.json()["answer"]
        assert "turns_seen=2" in a2 and "ibuprofen, cetirizine" in a2, a2
        print(
            f"PASS  turn 1 on A, turn 2 on B: B saw history + meds -> {a2!r}")

        # 2) A replica dies mid-conversation ---------------------------------
        # .kill() = SIGKILL on Linux/macOS, TerminateProcess on Windows: an
        # abrupt death, no graceful shutdown — like a crashed container.
        procs["A"].kill()
        procs["A"].wait()
        t3 = chat(B, ip, message="also some swelling", session_id=sid)
        assert "turns_seen=3" in t3.json()["answer"], t3.json()
        procs["C"] = start("server:app", C)       # a replica born just now
        t4 = chat(C, ip, message="what should I do?", session_id=sid)
        a4 = t4.json()["answer"]
        assert t4.status_code == 200 and "turns_seen=4" in a4, t4.text
        print(f"PASS  A killed (hard kill); turn 3 on B, turn 4 on brand-new C "
              f"-> {a4!r}")

        # 3) One rate-limit bucket across replicas ---------------------------
        # A blank message is rejected (400) AFTER the rate limiter runs, so
        # this costs no LLM calls. Burst 5, refill 1/min: 5 pass in TOTAL.
        flood_ip = "203.0.113.9"
        codes = [chat(B if i % 2 else C, flood_ip, message=" ").status_code
                 for i in range(10)]
        assert codes.count(400) == 5 and codes.count(429) == 5, codes
        print(f"PASS  10 requests alternating B/C: 5 allowed in total, "
              f"then 429 -> {codes}")

        # 4) Concurrent turns of ONE conversation are serialised -------------
        ip2 = "198.51.100.2"
        sid2 = chat(B, ip2, message="headache", lat=12.97,
                    lng=77.64).json()["session_id"]
        with cf.ThreadPoolExecutor(2) as ex:
            f1 = ex.submit(chat, B, ip2, message="worse now", session_id=sid2)
            time.sleep(0.15)
            f2 = ex.submit(chat, C, ip2, message="and nausea", session_id=sid2)
            got = sorted([f1.result().status_code, f2.result().status_code])
        assert got == [200, 409], got
        after = chat(C, ip2, message="still there?", session_id=sid2).json()
        assert "turns_seen=3" in after["answer"], after  # no lost/forked turn
        print(f"PASS  two simultaneous turns on B and C -> {got}; the next "
              f"turn sees exactly 3 turns (none lost or duplicated)")

        print("\nAll 4 statelessness checks green.")
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
                p.wait(5)


if __name__ == "__main__":
    main()
