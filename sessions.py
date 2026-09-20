"""
sessions.py — in-memory conversation sessions for the multi-turn /chat endpoint.

A "session" ties together, under one id, the three things a running conversation
needs to persist between HTTP requests:
  * the LangGraph thread_id (the session id IS the thread id) — so the agent
    remembers the transcript;
  * a stable patient_id in tools._PATIENTS — so location and the accumulating
    medication/history list survive across turns instead of being rebuilt each
    request;
  * light bookkeeping (created / last_seen) so idle sessions can be swept.

WHY A SEPARATE MODULE: the same reason emergency.py, osm.py etc. are separate —
one concern, unit-testable on its own (see __main__), and it doesn't import
tools, so there's no import cycle. sweep() RETURNS the patient_ids to clean up
and lets the caller (server.py) delete them from tools._PATIENTS; sessions.py
stays ignorant of what a patient record is.

SCALING CAVEAT (same as the rate limiter and the checkpointer): this lives in
process memory, so sessions are per-replica and lost on restart. Move to Redis
or a DB when you run more than one replica. For a single instance it's correct
and dependency-free. Note that sweeping a session frees its patient record but
NOT its MemorySaver thread — an in-memory saver has no eviction, so a durable,
evicting checkpointer is the real fix at scale.
"""

import os
import threading
import time
import uuid

# Sessions idle longer than this (seconds) are swept. Default 30 minutes.
TTL_SECONDS = int(os.getenv("CAREROUTE_SESSION_TTL", "1800"))

# Hard ceiling so a flood of new sessions can't grow memory without bound.
MAX_SESSIONS = int(os.getenv("CAREROUTE_MAX_SESSIONS", "5000"))

# session_id -> {"patient_id", "created", "last_seen"}
_sessions: dict[str, dict] = {}
_lock = threading.Lock()


def create_session(patient_id: str) -> str:
    """Register a new conversation for patient_id and return its session id."""
    sid = uuid.uuid4().hex
    now = time.monotonic()
    with _lock:
        _sessions[sid] = {"patient_id": patient_id, "created": now,
                          "last_seen": now, "turn": 1}
    return sid


def next_turn(session_id: str) -> int | None:
    """Increment and return a session's turn counter (for logging / ordering)."""
    with _lock:
        s = _sessions.get(session_id)
        if s is None:
            return None
        s["turn"] = s.get("turn", 1) + 1
        return s["turn"]


def get_session(session_id: str | None) -> dict | None:
    """Return a COPY of the session (and refresh last_seen), or None if unknown."""
    if not session_id:
        return None
    with _lock:
        s = _sessions.get(session_id)
        if s is None:
            return None
        s["last_seen"] = time.monotonic()
        return dict(s)


def sweep() -> list[str]:
    """Remove expired (and, if over the cap, oldest) sessions.

    Returns the patient_ids of everything removed so the caller can delete the
    matching records from tools._PATIENTS. Cheap to call on every request.
    """
    now = time.monotonic()
    removed: list[str] = []
    with _lock:
        # 1) TTL expiry.
        for sid in list(_sessions.keys()):
            if now - _sessions[sid]["last_seen"] > TTL_SECONDS:
                removed.append(_sessions.pop(sid)["patient_id"])
        # 2) Hard cap: if still too many, drop the least-recently-seen.
        overflow = len(_sessions) - MAX_SESSIONS
        if overflow > 0:
            oldest = sorted(_sessions.items(),
                            key=lambda kv: kv[1]["last_seen"])
            for sid, meta in oldest[:overflow]:
                _sessions.pop(sid, None)
                removed.append(meta["patient_id"])
    return removed


def count() -> int:
    with _lock:
        return len(_sessions)


if __name__ == "__main__":
    # Standalone self-test — no server, no FastAPI needed.
    import sys

    # Force a tiny TTL and cap for the test by rebinding the module globals.
    TTL_SECONDS = 0            # everything is immediately "idle"
    MAX_SESSIONS = 2

    a = create_session("pA")
    b = create_session("pB")
    assert get_session(a)["patient_id"] == "pA"
    assert get_session("nope") is None
    assert get_session(None) is None

    # With TTL=0 both are expired, so sweep returns both patient ids.
    freed = sweep()
    print("Swept patient ids (expect pA, pB in some order):", sorted(freed))
    assert sorted(freed) == ["pA", "pB"]
    assert count() == 0

    # Cap test: TTL back to large, cap=2, create 3 -> sweep drops the oldest 1.
    TTL_SECONDS = 10_000
    c1 = create_session("p1")
    time.sleep(0.01)
    c2 = create_session("p2")
    time.sleep(0.01)
    c3 = create_session("p3")
    dropped = sweep()
    print("Over-cap drop (expect ['p1']):", dropped)
    assert dropped == ["p1"] and count() == 2

    print("sessions.py self-test passed.")
    sys.exit(0)
