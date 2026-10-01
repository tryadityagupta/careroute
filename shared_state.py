"""
shared_state.py — where CareRoute keeps state that must outlive one request.

WHY THIS FILE EXISTS
--------------------
A replica must be disposable: any request can land on any replica, and a
replica can be killed at any moment (autoscale-in, deploy, crash). That only
works if NOTHING a later request depends on lives in process memory. CareRoute
has four such things:

    state                     before (per-process)      after (shared)
    ------------------------  ------------------------  ----------------------
    conversation transcript   LangGraph MemorySaver     Postgres checkpointer
    sessions + turn counter   dict in sessions.py       Redis hash, native TTL
    live patient records      dict in tools._PATIENTS   Redis JSON, native TTL
    rate-limit buckets        dict in security.py       Redis, atomic Lua

Everything else in memory is either read-only (the Synthea demo patients, the
provider JSON) or a cache that is merely slower when cold (the geocode and OSM
caches) — those are fine per-replica.

TWO MODES, ONE CODE PATH
------------------------
    REDIS_URL + DATABASE_URL set   -> shared mode: run as many replicas as you like
    neither set                    -> memory mode: single replica (dev, tests, CI)

Memory mode deliberately mimics the shared backends' semantics (copies on
read, TTLs, locks), so a bug that would only show up with Redis tends to show
up in the offline tests too.

FAIL FAST IN PRODUCTION: set CAREROUTE_REQUIRE_SHARED_STATE=1 on deployed
replicas. A replica that boots without REDIS_URL would otherwise run happily
in memory mode and silently split conversations across replicas — the worst
kind of bug, because it only appears under load. With the flag, it refuses to
start instead.
"""

import os

from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
REQUIRE_SHARED = os.getenv("CAREROUTE_REQUIRE_SHARED_STATE", "0").lower() in (
    "1", "true", "yes")
# Namespacing lets several environments (dev/staging) share one Redis safely.
KEY_PREFIX = os.getenv("CAREROUTE_KEY_PREFIX", "careroute")

USE_REDIS = bool(REDIS_URL)
USE_POSTGRES = bool(DATABASE_URL)

if REQUIRE_SHARED and not (USE_REDIS and USE_POSTGRES):
    missing = [n for n, v in (("REDIS_URL", USE_REDIS),
                              ("DATABASE_URL", USE_POSTGRES)) if not v]
    raise RuntimeError(
        "CAREROUTE_REQUIRE_SHARED_STATE=1 but " + " and ".join(missing) +
        " not set. Refusing to start in single-replica memory mode.")

print(
    f"[state] sessions/patients/rate-limits: "
    f"{'redis' if USE_REDIS else 'memory (single replica only)'} | "
    f"conversation checkpoints: "
    f"{'postgres' if USE_POSTGRES else 'memory (single replica only)'}"
)

_redis = None


def redis_client():
    """One shared, thread-safe Redis client per process (it owns a pool).

    Short timeouts on purpose: Redis answers in well under a millisecond, so a
    slow Redis is a broken Redis, and we would rather fail the request quickly
    than hold one of the threadpool's workers for seconds.
    """
    global _redis
    if _redis is None:
        import redis
        _redis = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_timeout=float(os.getenv("CAREROUTE_REDIS_TIMEOUT_S", "1.0")),
            socket_connect_timeout=float(
                os.getenv("CAREROUTE_REDIS_TIMEOUT_S", "1.0")),
            health_check_interval=30,
        )
    return _redis


def key(*parts: str) -> str:
    """careroute:session:<id>, careroute:patient:<id>, ..."""
    return ":".join((KEY_PREFIX, *parts))
