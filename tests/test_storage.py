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

pytestmark = pytest.mark.anyio

REDIS_URL = os.getenv("REDIS_URL", "").strip()
BACKENDS = ["memory"] + (["redis"] if REDIS_URL else [])


def _redis():
    return RedisConnection(REDIS_URL, key_prefix=f"careroute-test-{uuid.uuid4().hex[:8]}")


@pytest.fixture(params=BACKENDS)
async def sessions(request):
    if request.param == "memory":
        yield MemorySessionStore(ttl_s=1800, turn_lock_s=180)
        return
    conn = _redis()
    yield RedisSessionStore(conn, ttl_s=1800, turn_lock_s=180)
    await conn.aclose()


@pytest.fixture(params=BACKENDS)
async def patients(request):
    if request.param == "memory":
        yield MemoryPatientStore(ttl_s=60)
        return
    conn = _redis()
    yield RedisPatientStore(conn, ttl_s=60)
    await conn.aclose()


async def test_session_lifecycle(sessions):
    a = await sessions.create_session("pA")
    assert (await sessions.get_session(a))["patient_id"] == "pA"
    assert (await sessions.get_session("nope")) is None and (await sessions.get_session(None)) is None
    assert (await sessions.next_turn(a)) == 2 and (await sessions.get_session(a))["turn"] == 2
    assert (await sessions.next_turn("nope")) is None


async def test_turn_lock(sessions):
    a = await sessions.create_session("pA")
    t1 = await sessions.acquire_turn_lock(a)
    assert t1 and (await sessions.acquire_turn_lock(a)) is None
    await sessions.release_turn_lock(a, "not-the-owner")
    assert (await sessions.acquire_turn_lock(a)) is None, "a foreign token released the lock"
    await sessions.release_turn_lock(a, t1)
    t2 = await sessions.acquire_turn_lock(a)
    assert t2, "lock not released by its owner"
    await sessions.release_turn_lock(a, t2)


async def test_memory_sessions_expire_and_are_capped():
    s = MemorySessionStore(ttl_s=0, turn_lock_s=180)
    a = await s.create_session("pA")
    time.sleep(0.01)
    assert (await s.get_session(a)) is None                # expired reads as gone
    assert "pA" in (await s.sweep()) and (await s.count()) == 0
    s = MemorySessionStore(ttl_s=10_000, turn_lock_s=180, max_sessions=2)
    for p in ("p1", "p2", "p3"):
        await s.create_session(p)
        time.sleep(0.01)
    assert (await s.sweep()) == ["p1"] and (await s.count()) == 2


@pytest.mark.skipif(not REDIS_URL, reason="needs REDIS_URL")
async def test_redis_session_has_native_ttl():
    s = RedisSessionStore(_redis(), ttl_s=1800, turn_lock_s=180)
    a = await s.create_session("pA")
    ttl = await s.redis.client.ttl(s.redis.key("session", a))
    assert 0 < ttl <= 1800
    assert (await s.next_turn("missing")) is None
    assert not await s.redis.client.exists(s.redis.key("session", "missing"))   # not created
    await s.redis.aclose()


async def test_patient_store_is_copy_on_read(patients):
    await patients.put("LIVE-test", {"name": "A", "current_medications": ["x"]})
    rec = await patients.get("LIVE-test")
    rec["current_medications"].append("y")             # mutate the COPY...
    assert (await patients.get("LIVE-test"))["current_medications"] == ["x"], "leaked mutation"
    await patients.put("LIVE-test", rec)              # ...nothing changes until put
    assert (await patients.get("LIVE-test"))["current_medications"] == ["x", "y"]
    await patients.delete("LIVE-test")
    assert (await patients.get("LIVE-test")) is None
