"""
storage/redis_client.py — one async Redis connection per process, plus key naming.

WHY SHARED STATE: a replica must be disposable — any request can land on any
replica, and any replica can die. So nothing a later request depends on may
live in process memory:

    state                     memory mode (dev/tests)   shared mode (replicas)
    ------------------------  ------------------------  ----------------------
    conversation transcript   LangGraph MemorySaver     AsyncPostgresSaver
    sessions + turn counter   MemorySessionStore        RedisSessionStore
    live patient records      MemoryPatientStore        RedisPatientStore
    rate-limit buckets        MemoryRateLimiter         RedisRateLimiter

Memory mode mimics the shared backends' semantics (copies on read, TTLs,
locks), so a bug that would only appear with Redis tends to show up offline.

ASYNC: redis.asyncio, so a request waiting on Redis yields the event loop
instead of holding a worker thread. The client binds to the event loop it is
first used on — one loop per process under uvicorn, one per test.
"""

from __future__ import annotations


class RedisConnection:
    def __init__(self, url: str, *, timeout_s: float = 1.0, key_prefix: str = "careroute"):
        self.url = url
        self.timeout_s = timeout_s
        # Namespacing lets dev/staging (and test runs) share one Redis safely.
        self.key_prefix = key_prefix
        self._client = None

    @property
    def client(self):
        """Async client owning a connection pool, created on first use.

        Creating it connects to nothing; the first command does. Short
        timeouts on purpose: Redis answers in well under a millisecond, so a
        slow Redis is a broken Redis — fail the request quickly.
        """
        if self._client is None:
            import redis.asyncio as aioredis
            self._client = aioredis.Redis.from_url(
                self.url, decode_responses=True,
                socket_timeout=self.timeout_s,
                socket_connect_timeout=self.timeout_s,
                health_check_interval=30)
        return self._client

    def key(self, *parts: str) -> str:
        """careroute:session:<id>, careroute:patient:<id>, ..."""
        return ":".join((self.key_prefix, *parts))

    async def ping(self) -> bool:
        return bool(await self.client.ping())

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
