"""
security/limits.py — volume limits, framework-agnostic and unit-testable.

/care and /chat run the agent (several LLM calls + map lookups), so the real
exposure is VOLUME, not unit price. Two limits:

  RateLimiter  per-client token bucket: smooths bursts, stops one client looping.
  DailyCap     optional global cap per UTC day: backstop against distributed
               abuse. The hard money stop remains the billing-console limit.

In-memory versions are exact for ONE replica; with N replicas a client would
get N x the limit. The Redis versions keep one bucket for the whole fleet:
  * ATOMIC — read/refill/spend is one Lua script inside Redis, so two
    replicas can't both spend the last token.
  * ONE CLOCK — the script uses Redis's TIME, so replica clock skew can't
    mint or burn tokens.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod

from careroute.storage.redis_client import RedisConnection


class RateLimiter(ABC):
    def __init__(self, rate_per_min: int, burst: int):
        self.rate = rate_per_min / 60.0        # tokens per second
        self.burst = float(burst)

    @abstractmethod
    def check(self, key: str) -> tuple[bool, float]:
        """(allowed, retry_after_seconds); retry_after is 0 when allowed."""


class DailyCap(ABC):
    def __init__(self, cap: int):
        self.cap = cap                          # <= 0 disables the cap

    @abstractmethod
    def allow(self) -> bool: ...


class _Bucket:
    __slots__ = ("tokens", "last")

    def __init__(self, tokens: float, last: float):
        self.tokens = tokens
        self.last = last


class MemoryRateLimiter(RateLimiter):
    """Classic token bucket; new clients start full."""

    def __init__(self, rate_per_min: int, burst: int):
        super().__init__(rate_per_min, burst)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._last_prune = time.monotonic()

    def check(self, key):
        now = time.monotonic()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = self._buckets[key] = _Bucket(self.burst, now)
            b.tokens = min(self.burst, b.tokens + (now - b.last) * self.rate)
            b.last = now
            if b.tokens >= 1.0:
                b.tokens -= 1.0
                allowed, retry = True, 0.0
            else:
                allowed = False
                retry = (1.0 - b.tokens) / self.rate if self.rate > 0 else 60.0
            self._maybe_prune(now)
            return allowed, retry

    def _maybe_prune(self, now: float) -> None:
        """Drop full, idle buckets (under the lock) so memory stays bounded."""
        if now - self._last_prune < 60 or len(self._buckets) < 1024:
            return
        self._last_prune = now
        for k in [k for k, b in self._buckets.items()
                  if b.tokens >= self.burst and now - b.last > 300]:
            self._buckets.pop(k, None)


class MemoryDailyCap(DailyCap):
    def __init__(self, cap: int):
        super().__init__(cap)
        self._day = None
        self._count = 0
        self._lock = threading.Lock()

    def allow(self):
        if self.cap <= 0:
            return True
        today = time.gmtime().tm_yday          # flips at UTC midnight
        with self._lock:
            if today != self._day:
                self._day, self._count = today, 0
            if self._count >= self.cap:
                return False
            self._count += 1
            return True


class RedisRateLimiter(RateLimiter):
    _BUCKET_LUA = """
local rate  = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local t     = redis.call('TIME')
local now   = tonumber(t[1]) + tonumber(t[2]) / 1000000
local b      = redis.call('HMGET', KEYS[1], 'tokens', 'last')
local tokens = tonumber(b[1]) or burst
local last   = tonumber(b[2]) or now
tokens = math.min(burst, tokens + math.max(0, now - last) * rate)
local allowed, retry = 0, 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
elseif rate > 0 then
  retry = (1 - tokens) / rate
else
  retry = 60
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'last', tostring(now))
-- A bucket idle long enough to be full again carries no state: let it expire.
local ttl = 60
if rate > 0 then ttl = math.ceil(burst / rate) + 60 end
redis.call('EXPIRE', KEYS[1], ttl)
return {allowed, tostring(retry)}
"""

    def __init__(self, redis: RedisConnection, rate_per_min: int, burst: int):
        super().__init__(rate_per_min, burst)
        self.redis = redis
        self._script = redis.client.register_script(self._BUCKET_LUA)

    def check(self, key):
        allowed, retry = self._script(keys=[self.redis.key("ratelimit", key)],
                                      args=[self.rate, self.burst])
        return bool(int(allowed)), float(retry)


class RedisDailyCap(DailyCap):
    """One INCR per call on a key named after the UTC date; expires itself."""

    def __init__(self, redis: RedisConnection, cap: int):
        super().__init__(cap)
        self.redis = redis

    def allow(self):
        if self.cap <= 0:
            return True
        k = self.redis.key("daily", time.strftime("%Y-%m-%d", time.gmtime()))
        pipe = self.redis.client.pipeline()
        pipe.incr(k)
        pipe.expire(k, 2 * 86400)
        n, _ = pipe.execute()
        return n <= self.cap
