"""tests/test_security.py — token buckets and the daily cap, memory and Redis."""

import os
import time
import uuid

import pytest

from careroute.security.limits import (MemoryDailyCap, MemoryRateLimiter,
                                       RedisRateLimiter)
from careroute.storage.redis_client import RedisConnection

REDIS_URL = os.getenv("REDIS_URL", "").strip()


def test_burst_then_refuse_then_refill():
    rl = MemoryRateLimiter(rate_per_min=60, burst=5)      # 1 token/s, burst 5
    assert [rl.check("1.2.3.4")[0] for _ in range(6)] == [True] * 5 + [False]
    allowed, retry = rl.check("1.2.3.4")
    assert not allowed and 0 < retry <= 1.0
    time.sleep(1.1)
    assert rl.check("1.2.3.4")[0]
    assert rl.check("9.9.9.9")[0]                         # independent buckets


def test_daily_cap():
    assert [MemoryDailyCap(2).allow() for _ in range(1)] == [True]
    dc = MemoryDailyCap(cap=2)
    assert [dc.allow() for _ in range(3)] == [True, True, False]
    assert MemoryDailyCap(0).allow()                      # 0 = disabled


@pytest.mark.skipif(not REDIS_URL, reason="needs REDIS_URL")
def test_redis_bucket_is_shared_across_replicas():
    conn = RedisConnection(REDIS_URL, key_prefix=f"careroute-test-{uuid.uuid4().hex[:8]}")
    a = RedisRateLimiter(conn, rate_per_min=60, burst=5)  # two "replicas",
    b = RedisRateLimiter(conn, rate_per_min=60, burst=5)  # one Redis bucket
    got = [(a if i % 2 else b).check("ip")[0] for i in range(6)]
    assert got == [True] * 5 + [False]
    time.sleep(1.1)
    assert a.check("ip")[0]
