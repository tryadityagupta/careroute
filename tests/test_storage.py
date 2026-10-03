"""
tests/test_storage.py — session store + patient store contracts.

Every test runs against the MEMORY backend, and again against REDIS when
REDIS_URL is set, so both implementations are held to one contract.
"""

import os
import time
import uuid

import pytest

from careroute.storage.patient_store import MemoryPatientStore, RedisPatientStore
from careroute.storage.redis_client import RedisConnection
from careroute.storage.sessions import MemorySessionStore, RedisSessionStore

REDIS_URL = os.getenv("REDIS_URL", "").strip()
BACKENDS = ["memory"] + (["redis"] if REDIS_URL else [])


def _redis():
    return RedisConnection(REDIS_URL, key_prefix=f"careroute-test-{uuid.uuid4().hex[:8]}")


@pytest.fixture(params=BACKENDS)
def sessions(request):
    if request.param == "memory":
        return MemorySessionStore(ttl_s=1800, turn_lock_s=180)
    return RedisSessionStore(_redis(), ttl_s=1800, turn_lock_s=180)


@pytest.fixture(params=BACKENDS)
def patients(request):
    if request.param == "memory":
        return MemoryPatientStore(ttl_s=60)
    return RedisPatientStore(_redis(), ttl_s=60)


def test_session_lifecycle(sessions):
    a = sessions.create_session("pA")
    assert sessions.get_session(a)["patient_id"] == "pA"
    assert sessions.get_session("nope") is None and sessions.get_session(None) is None
    assert sessions.next_turn(a) == 2 and sessions.get_session(a)["turn"] == 2
    assert sessions.next_turn("nope") is None


def test_turn_lock(sessions):
    a = sessions.create_session("pA")
    t1 = sessions.acquire_turn_lock(a)
    assert t1 and sessions.acquire_turn_lock(a) is None
    sessions.release_turn_lock(a, "not-the-owner")
    assert sessions.acquire_turn_lock(a) is None, "a foreign token released the lock"
    sessions.release_turn_lock(a, t1)
    t2 = sessions.acquire_turn_lock(a)
    assert t2, "lock not released by its owner"
    sessions.release_turn_lock(a, t2)


def test_memory_sessions_expire_and_are_capped():
    s = MemorySessionStore(ttl_s=0, turn_lock_s=180)
    a = s.create_session("pA")
    time.sleep(0.01)
    assert s.get_session(a) is None                # expired reads as gone
    assert "pA" in s.sweep() and s.count() == 0
    s = MemorySessionStore(ttl_s=10_000, turn_lock_s=180, max_sessions=2)
    for p in ("p1", "p2", "p3"):
        s.create_session(p)
        time.sleep(0.01)
    assert s.sweep() == ["p1"] and s.count() == 2


@pytest.mark.skipif(not REDIS_URL, reason="needs REDIS_URL")
def test_redis_session_has_native_ttl():
    s = RedisSessionStore(_redis(), ttl_s=1800, turn_lock_s=180)
    a = s.create_session("pA")
    ttl = s.redis.client.ttl(s.redis.key("session", a))
    assert 0 < ttl <= 1800
    assert s.next_turn("missing") is None
    assert not s.redis.client.exists(s.redis.key("session", "missing"))   # not created


def test_patient_store_is_copy_on_read(patients):
    patients.put("LIVE-test", {"name": "A", "current_medications": ["x"]})
    rec = patients.get("LIVE-test")
    rec["current_medications"].append("y")             # mutate the COPY...
    assert patients.get("LIVE-test")["current_medications"] == ["x"], "leaked mutation"
    patients.put("LIVE-test", rec)                     # ...nothing changes until put
    assert patients.get("LIVE-test")["current_medications"] == ["x", "y"]
    patients.delete("LIVE-test")
    assert patients.get("LIVE-test") is None
