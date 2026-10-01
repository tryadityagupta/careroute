"""
patient_store.py — live patient records that must survive across requests.

Before: server.py wrote each live user's record into tools._PATIENTS, a
module-level dict, and the agent's update_patient_record tool mutated it in
place. With two replicas, turn 2 landing on the other replica found no record
and answered "Session expired".

Now there are two kinds of patient record, kept apart on purpose:
  * STATIC demo patients (P001..., from Synthea) — read-only reference data,
    loaded once into tools._PATIENTS on every replica. Never written.
  * LIVE records (LIVE-xxxx, plus any edits the agent makes to a demo
    patient) — stored here, in Redis with a TTL, so every replica sees them.

Semantics are copy-on-read / write-back in BOTH backends: get() hands you a
copy, and nothing changes until you put() it. That's how Redis behaves anyway
(you get a decoded JSON copy), and making the memory backend behave the same
means "I mutated it but forgot to save it" bugs fail in the offline tests,
not only in production.
"""

import copy
import json
import os
import threading
import time

import shared_state

# Live records expire with their conversation (same default as sessions).
TTL_SECONDS = int(os.getenv("CAREROUTE_SESSION_TTL", "1800"))


class MemoryPatients:
    def __init__(self):
        self._recs: dict[str, tuple[dict, float]] = {}
        self._lock = threading.Lock()

    def get(self, pid: str) -> dict | None:
        with self._lock:
            item = self._recs.get(pid)
            if item is None:
                return None
            rec, expires = item
            if expires < time.monotonic():
                self._recs.pop(pid, None)
                return None
            return copy.deepcopy(rec)

    def put(self, pid: str, rec: dict, ttl: int = TTL_SECONDS) -> None:
        with self._lock:
            self._recs[pid] = (copy.deepcopy(rec), time.monotonic() + ttl)

    def delete(self, pid: str) -> None:
        with self._lock:
            self._recs.pop(pid, None)


class RedisPatients:
    def __init__(self, client):
        self.r = client

    @staticmethod
    def _k(pid: str) -> str:
        return shared_state.key("patient", pid)

    def get(self, pid: str) -> dict | None:
        raw = self.r.get(self._k(pid))
        return json.loads(raw) if raw else None

    def put(self, pid: str, rec: dict, ttl: int = TTL_SECONDS) -> None:
        self.r.set(self._k(pid), json.dumps(rec), ex=ttl)

    def delete(self, pid: str) -> None:
        self.r.delete(self._k(pid))


_backend = (RedisPatients(shared_state.redis_client())
            if shared_state.USE_REDIS else MemoryPatients())

get = _backend.get
put = _backend.put
delete = _backend.delete


if __name__ == "__main__":
    put("LIVE-test", {"name": "A", "current_medications": ["x"]}, ttl=60)
    rec = get("LIVE-test")
    rec["current_medications"].append("y")      # mutate the COPY...
    assert get("LIVE-test")["current_medications"] == ["x"], "leaked mutation"
    put("LIVE-test", rec)                       # ...nothing changes until put
    assert get("LIVE-test")["current_medications"] == ["x", "y"]
    delete("LIVE-test")
    assert get("LIVE-test") is None
    print(f"patient_store.py self-test passed ({type(_backend).__name__}).")
