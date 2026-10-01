"""
sessions.py — conversation sessions for the multi-turn /chat endpoint.

A "session" ties together, under one id, what a running conversation needs to
persist between HTTP requests:
  * the LangGraph thread_id (the session id IS the thread id) — so the agent
    remembers the transcript (stored by the checkpointer, see agent_langgraph);
  * a stable patient_id (record stored in patient_store) — so location and the
    accumulating medication list survive across turns;
  * a turn counter, for logging and ordering.

TWO BACKENDS, SAME API (picked by shared_state):
  * RedisSessions  — a Redis hash per session with a native TTL. Expiry is
    Redis's job, so there is no sweep loop and no cap needed: an idle session
    simply disappears. Any replica can serve any turn.
  * MemorySessions — the original dict + TTL sweep, for dev and offline tests.

THE TURN LOCK (good interview point)
------------------------------------
Statelessness creates a new race. With one replica, two messages for the same
conversation were serialised by accident. With N replicas, a double-click or a
retry can run two turns of ONE conversation at the same time on two replicas:
both read the same checkpoint, both append, and one turn's messages are lost
(or the transcript forks). So a conversation may run only one turn at a time:
acquire_turn_lock() is a Redis SET NX EX lock, released with a compare-and-
delete so a replica can never release a lock that expired and was re-taken by
someone else. The EX is a safety net: a replica killed mid-turn can't wedge the
conversation forever.
"""

import os
import threading
import time
import uuid

import shared_state

# Sessions idle longer than this (seconds) expire. Default 30 minutes.
TTL_SECONDS = int(os.getenv("CAREROUTE_SESSION_TTL", "1800"))

# Memory backend only: hard ceiling so a flood can't grow memory without bound.
# (Redis needs no cap: memory is bounded by TTL and by Redis's maxmemory.)
MAX_SESSIONS = int(os.getenv("CAREROUTE_MAX_SESSIONS", "5000"))

# Longest a single turn may hold the lock. Must exceed the worst-case turn
# (8 LLM calls + map lookups); after this a crashed replica's lock self-clears.
TURN_LOCK_SECONDS = int(os.getenv("CAREROUTE_TURN_LOCK_S", "180"))


class MemorySessions:
    """Single-replica backend: the original dict, now behind the shared API."""

    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._locks: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def create_session(self, patient_id: str) -> str:
        sid = uuid.uuid4().hex
        now = time.monotonic()
        with self._lock:
            self._sessions[sid] = {"patient_id": patient_id, "created": now,
                                   "last_seen": now, "turn": 1}
        return sid

    def next_turn(self, session_id: str) -> int | None:
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return None
            s["turn"] = s.get("turn", 1) + 1
            return s["turn"]

    def get_session(self, session_id: str | None) -> dict | None:
        if not session_id:
            return None
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return None
            if time.monotonic() - s["last_seen"] > TTL_SECONDS:
                # Expired but not swept yet: behave like Redis, where an
                # expired key is simply gone.
                return None
            s["last_seen"] = time.monotonic()
            return dict(s)

    def sweep(self) -> list[str]:
        now = time.monotonic()
        removed: list[str] = []
        with self._lock:
            for sid in list(self._sessions.keys()):
                if now - self._sessions[sid]["last_seen"] > TTL_SECONDS:
                    removed.append(self._sessions.pop(sid)["patient_id"])
            overflow = len(self._sessions) - MAX_SESSIONS
            if overflow > 0:
                oldest = sorted(self._sessions.items(),
                                key=lambda kv: kv[1]["last_seen"])
                for sid, meta in oldest[:overflow]:
                    self._sessions.pop(sid, None)
                    removed.append(meta["patient_id"])
        return removed

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def acquire_turn_lock(self, session_id: str) -> str | None:
        token = uuid.uuid4().hex
        now = time.monotonic()
        with self._lock:
            held = self._locks.get(session_id)
            if held and held[1] > now:
                return None
            self._locks[session_id] = (token, now + TURN_LOCK_SECONDS)
        return token

    def release_turn_lock(self, session_id: str, token: str) -> None:
        with self._lock:
            held = self._locks.get(session_id)
            if held and held[0] == token:
                self._locks.pop(session_id, None)


# Compare-and-delete: only the holder of `token` may release the lock.
_RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


class RedisSessions:
    """Shared backend: any replica can serve any turn of any conversation."""

    def __init__(self, client):
        self.r = client
        self._release = client.register_script(_RELEASE_LUA)

    @staticmethod
    def _k(sid: str) -> str:
        return shared_state.key("session", sid)

    def create_session(self, patient_id: str) -> str:
        sid = uuid.uuid4().hex
        k = self._k(sid)
        pipe = self.r.pipeline()
        pipe.hset(k, mapping={"patient_id": patient_id,
                              "created": time.time(), "turn": 1})
        pipe.expire(k, TTL_SECONDS)
        pipe.execute()
        return sid

    def next_turn(self, session_id: str) -> int | None:
        k = self._k(session_id)
        # HINCRBY on a missing key would CREATE it (without a TTL), so only
        # increment a session that still exists.
        if not self.r.exists(k):
            return None
        return int(self.r.hincrby(k, "turn", 1))

    def get_session(self, session_id: str | None) -> dict | None:
        if not session_id:
            return None
        k = self._k(session_id)
        pipe = self.r.pipeline()
        pipe.hgetall(k)
        pipe.expire(k, TTL_SECONDS)       # sliding expiry: touch on every turn
        data, _ = pipe.execute()
        if not data:
            return None
        return {"patient_id": data["patient_id"],
                "created": float(data.get("created", 0)),
                "turn": int(data.get("turn", 1))}

    def sweep(self) -> list[str]:
        return []                         # Redis expires keys itself

    def count(self) -> int:
        return sum(1 for _ in self.r.scan_iter(
            match=shared_state.key("session", "*"), count=1000))

    def acquire_turn_lock(self, session_id: str) -> str | None:
        token = uuid.uuid4().hex
        ok = self.r.set(shared_state.key("turnlock", session_id), token,
                        nx=True, ex=TURN_LOCK_SECONDS)
        return token if ok else None

    def release_turn_lock(self, session_id: str, token: str) -> None:
        self._release(keys=[shared_state.key("turnlock", session_id)],
                      args=[token])


_backend = (RedisSessions(shared_state.redis_client())
            if shared_state.USE_REDIS else MemorySessions())

# Module-level API, unchanged for callers (server.py, tests).
create_session = _backend.create_session
next_turn = _backend.next_turn
get_session = _backend.get_session
sweep = _backend.sweep
count = _backend.count
acquire_turn_lock = _backend.acquire_turn_lock
release_turn_lock = _backend.release_turn_lock


if __name__ == "__main__":
    # Self-test of whichever backend is configured. Run it twice to cover both:
    #   python sessions.py                              (memory)
    #   REDIS_URL=redis://localhost:6379/15 python sessions.py   (redis)
    import sys

    s = _backend
    a = s.create_session("pA")
    assert s.get_session(a)["patient_id"] == "pA"
    assert s.get_session("nope") is None and s.get_session(None) is None
    assert s.next_turn(a) == 2 and s.get_session(a)["turn"] == 2
    assert s.next_turn("nope") is None

    # Turn lock: second acquirer is refused; a stale token can't release it.
    t1 = s.acquire_turn_lock(a)
    assert t1 and s.acquire_turn_lock(a) is None
    s.release_turn_lock(a, "not-the-owner")
    assert s.acquire_turn_lock(a) is None, "foreign token released the lock"
    s.release_turn_lock(a, t1)
    t2 = s.acquire_turn_lock(a)
    assert t2, "lock not released by its owner"
    s.release_turn_lock(a, t2)

    if isinstance(s, MemorySessions):
        # TTL + cap behaviour of the memory backend.
        TTL_SECONDS = 0
        assert s.get_session(a) is None           # expired reads as gone
        assert "pA" in s.sweep() and s.count() == 0
        TTL_SECONDS, MAX_SESSIONS = 10_000, 2
        for p in ("p1", "p2", "p3"):
            s.create_session(p)
            time.sleep(0.01)
        assert s.sweep() == ["p1"] and s.count() == 2
    else:
        # Redis: the key really carries a TTL (no sweep loop needed).
        ttl = s.r.ttl(shared_state.key("session", a))
        assert 0 < ttl <= TTL_SECONDS, ttl
        s.r.delete(shared_state.key("session", a))

    print(f"sessions.py self-test passed ({type(s).__name__}).")
    sys.exit(0)
