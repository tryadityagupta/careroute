"""
storage/sessions.py — conversation sessions for the multi-turn /chat endpoint.

A session ties together, under one id, what a conversation needs between HTTP
requests: the LangGraph thread (the session id IS the thread id), a stable
patient_id, and a turn counter.

THE TURN LOCK. Statelessness creates a new race: a double click or a retry can
run two turns of ONE conversation at once on two replicas; both read the same
checkpoint and one turn is lost. So a conversation runs one turn at a time:
acquire_turn_lock() is SET NX EX, released by compare-and-delete so a replica
can never release a lock that expired and was re-taken. The EX is the safety
net: a replica killed mid-turn can't wedge the conversation forever.
"""

from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod

from careroute.storage.redis_client import RedisConnection


class SessionStore(ABC):
    def __init__(self, ttl_s: int, turn_lock_s: int):
        self.ttl_s = ttl_s                  # idle sessions expire after this
        self.turn_lock_s = turn_lock_s      # must exceed the worst-case turn

    @abstractmethod
    def create_session(self, patient_id: str) -> str: ...

    @abstractmethod
    def get_session(self, session_id: str | None) -> dict | None:
        """{patient_id, created, turn} for a live session, else None. Touching
        a session slides its expiry."""

    @abstractmethod
    def next_turn(self, session_id: str) -> int | None: ...

    @abstractmethod
    def acquire_turn_lock(self, session_id: str) -> str | None:
        """A token if acquired, None if another turn holds the lock."""

    @abstractmethod
    def release_turn_lock(self, session_id: str, token: str) -> None: ...

    def sweep(self) -> list[str]:
        """Expire idle sessions; return their patient ids. No-op where the
        backend expires keys itself."""
        return []

    @abstractmethod
    def count(self) -> int: ...


class MemorySessionStore(SessionStore):
    """Single-replica backend: dev and offline tests."""

    def __init__(self, ttl_s: int = 1800, turn_lock_s: int = 180, max_sessions: int = 5000):
        super().__init__(ttl_s, turn_lock_s)
        # Hard ceiling so a flood can't grow memory without bound.
        self.max_sessions = max_sessions
        self._sessions: dict[str, dict] = {}
        self._locks: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def create_session(self, patient_id):
        sid = uuid.uuid4().hex
        now = time.monotonic()
        with self._lock:
            self._sessions[sid] = {"patient_id": patient_id, "created": now,
                                   "last_seen": now, "turn": 1}
        return sid

    def get_session(self, session_id):
        if not session_id:
            return None
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None or time.monotonic() - s["last_seen"] > self.ttl_s:
                return None                  # expired reads as gone, like Redis
            s["last_seen"] = time.monotonic()
            return dict(s)

    def next_turn(self, session_id):
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return None
            s["turn"] = s.get("turn", 1) + 1
            return s["turn"]

    def sweep(self):
        now = time.monotonic()
        removed: list[str] = []
        with self._lock:
            for sid in list(self._sessions):
                if now - self._sessions[sid]["last_seen"] > self.ttl_s:
                    removed.append(self._sessions.pop(sid)["patient_id"])
            overflow = len(self._sessions) - self.max_sessions
            if overflow > 0:
                oldest = sorted(self._sessions.items(), key=lambda kv: kv[1]["last_seen"])
                for sid, meta in oldest[:overflow]:
                    self._sessions.pop(sid, None)
                    removed.append(meta["patient_id"])
        return removed

    def count(self):
        with self._lock:
            return len(self._sessions)

    def acquire_turn_lock(self, session_id):
        token = uuid.uuid4().hex
        now = time.monotonic()
        with self._lock:
            held = self._locks.get(session_id)
            if held and held[1] > now:
                return None
            self._locks[session_id] = (token, now + self.turn_lock_s)
        return token

    def release_turn_lock(self, session_id, token):
        with self._lock:
            held = self._locks.get(session_id)
            if held and held[0] == token:
                self._locks.pop(session_id, None)


class RedisSessionStore(SessionStore):
    """Shared backend: any replica can serve any turn of any conversation."""

    # Compare-and-delete: only the holder of the token may release the lock.
    _RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""
    # Increment only a session that still exists: HINCRBY on a missing key
    # would CREATE it without a TTL. Atomic, and one round trip instead of two.
    _NEXT_TURN_LUA = """
if redis.call('EXISTS', KEYS[1]) == 1 then
  return redis.call('HINCRBY', KEYS[1], 'turn', 1)
end
return nil
"""

    def __init__(self, redis: RedisConnection, ttl_s: int = 1800, turn_lock_s: int = 180):
        super().__init__(ttl_s, turn_lock_s)
        self.redis = redis
        self._release = redis.client.register_script(self._RELEASE_LUA)
        self._next_turn = redis.client.register_script(self._NEXT_TURN_LUA)

    def _k(self, sid: str) -> str:
        return self.redis.key("session", sid)

    def create_session(self, patient_id):
        sid = uuid.uuid4().hex
        pipe = self.redis.client.pipeline()
        pipe.hset(self._k(sid), mapping={"patient_id": patient_id,
                                         "created": time.time(), "turn": 1})
        pipe.expire(self._k(sid), self.ttl_s)
        pipe.execute()
        return sid

    def get_session(self, session_id):
        if not session_id:
            return None
        pipe = self.redis.client.pipeline()
        pipe.hgetall(self._k(session_id))
        pipe.expire(self._k(session_id), self.ttl_s)    # sliding expiry
        data, _ = pipe.execute()
        if not data:
            return None
        return {"patient_id": data["patient_id"],
                "created": float(data.get("created", 0)),
                "turn": int(data.get("turn", 1))}

    def next_turn(self, session_id):
        n = self._next_turn(keys=[self._k(session_id)])
        return None if n is None else int(n)

    def count(self):
        return sum(1 for _ in self.redis.client.scan_iter(
            match=self.redis.key("session", "*"), count=1000))

    def acquire_turn_lock(self, session_id):
        token = uuid.uuid4().hex
        ok = self.redis.client.set(self.redis.key("turnlock", session_id), token,
                                   nx=True, ex=self.turn_lock_s)
        return token if ok else None

    def release_turn_lock(self, session_id, token):
        self._release(keys=[self.redis.key("turnlock", session_id)], args=[token])
