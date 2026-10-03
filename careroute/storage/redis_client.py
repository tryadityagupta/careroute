"""
storage/redis_client.py — one Redis connection per process, plus key naming.

WHY SHARED STATE: a replica must be disposable — any request can land on any
replica, and any replica can die. So nothing a later request depends on may
live in process memory:

    state                     memory mode (dev/tests)   shared mode (replicas)
    ------------------------  ------------------------  ----------------------
    conversation transcript   LangGraph MemorySaver     Postgres checkpointer
    sessions + turn counter   MemorySessionStore        RedisSessionStore
    live patient records      MemoryPatientStore        RedisPatientStore
    rate-limit buckets        MemoryRateLimiter         RedisRateLimiter

Memory mode mimics the shared backends' semantics (copies on read, TTLs,
locks), so a bug that would only appear with Redis tends to show up offline.
"""

from __future__ import annotations

import threading


class RedisConnection:
    def __init__(self, url: str, *, timeout_s: float = 1.0, key_prefix: str = "careroute"):
        self.url = url
        self.timeout_s = timeout_s
        # Namespacing lets dev/staging (and test runs) share one Redis safely.
        self.key_prefix = key_prefix
        self._client = None
        self._lock = threading.Lock()

    @property
    def client(self):
        """Thread-safe client owning a connection pool, created on first use.

        Short timeouts on purpose: Redis answers in well under a millisecond,
        so a slow Redis is a broken Redis — fail the request quickly rather
        than hold a worker for seconds.
        """
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import redis
                    self._client = redis.Redis.from_url(
                        self.url, decode_responses=True,
                        socket_timeout=self.timeout_s,
                        socket_connect_timeout=self.timeout_s,
                        health_check_interval=30)
        return self._client

    def key(self, *parts: str) -> str:
        """careroute:session:<id>, careroute:patient:<id>, ..."""
        return ":".join((self.key_prefix, *parts))

    def ping(self) -> bool:
        return bool(self.client.ping())
