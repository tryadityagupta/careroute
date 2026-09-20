"""
security.py — the gate in front of the paid /care endpoint.

WHY THIS FILE EXISTS
--------------------
/care is expensive to serve: every call runs the LangGraph agent, which makes
up to MAX_LLM_CALLS model calls plus real map lookups. The endpoint used to be
wide open (CORS "*", no auth, no throttle), so anyone who found the URL could
loop it and spend our OpenAI and Google budget for us. The model itself is
already cheap (gpt-4o-mini, capped at 8 calls/request) — the real exposure is
VOLUME, not unit price. So this file caps volume.

THREE INDEPENDENT LAYERS (each can be turned on/off by env var):

  1. Rate limit (per client IP) — token bucket. Smooths out bursts and stops a
     single client from looping. ON by default with generous limits.
  2. API-key gate — optional. When CAREROUTE_API_KEYS is set, callers must send
     a matching X-API-Key header. Use it to lock the endpoint during a private
     beta. OFF by default (so local dev and the current demo keep working).
  3. Daily circuit breaker — optional global cap on total /care calls per UTC
     day. A backstop against distributed abuse (many IPs) or a runaway loop.
     OFF by default. This is belt-and-braces; the real hard stop is the monthly
     spend limit you set in the OpenAI and Google Cloud billing consoles.

DESIGN NOTE (good interview point): the limiter is pure stdlib and framework-
agnostic — the RateLimiter/DailyCounter classes know nothing about FastAPI, so
they're unit-testable on their own (see __main__). Only the two thin dependency
functions at the bottom touch FastAPI.

SCALING CAVEAT (say this out loud in an interview): the counters live in
process memory, so they are PER REPLICA. On one Container Apps replica that's
exact; if you scale to N replicas a client gets up to N× the limit. The moment
you run more than one replica, move the bucket store to Redis (same logic, keys
in Redis instead of a dict). For a single small instance this is correct and
has zero extra infrastructure.
"""

import hmac
import os
import threading
import time
from math import ceil

from dotenv import load_dotenv

load_dotenv()  # so these flags work from .env as well as real shell vars


# --- Configuration (all overridable by environment) ------------------------

# Per-IP allowance. 20/min with a burst of 5 is roomy for a human (who submits a
# handful of times) but shuts down a tight loop fast.
RATE_PER_MIN = int(os.getenv("CAREROUTE_RATE_PER_MIN", "20"))
BURST = int(os.getenv("CAREROUTE_BURST", "5"))

# Optional global daily cap on /care calls. 0 = disabled.
DAILY_CAP = int(os.getenv("CAREROUTE_DAILY_CALL_CAP", "0"))

# Optional API keys (comma-separated). Empty = auth disabled (endpoint open).
API_KEYS = [k.strip() for k in os.getenv("CAREROUTE_API_KEYS", "").split(",")
            if k.strip()]

# Behind Azure Container Apps' ingress the real client IP arrives in
# X-Forwarded-For (request.client.host would be the proxy). Trust it by default
# because the ONLY public path to the container is that ingress. If you ever
# expose the container directly, set CAREROUTE_TRUST_XFF=0 so a caller can't
# spoof the header to dodge the limit.
TRUST_XFF = os.getenv("CAREROUTE_TRUST_XFF",
                      "1").lower() not in ("0", "false", "")


# --- Layer 1: token-bucket rate limiter (framework-agnostic) ---------------

class _Bucket:
    """One client's bucket: a float count of tokens and the last refill time."""
    __slots__ = ("tokens", "last")

    def __init__(self, tokens: float, last: float):
        self.tokens = tokens
        self.last = last


class RateLimiter:
    """Classic token bucket.

    Each key (a client IP) has a bucket that refills at `rate` tokens/second up
    to `burst`. A request costs one token. If the bucket is empty the request is
    refused and we return how long until one token is back — that becomes the
    Retry-After header, so a well-behaved client knows exactly when to retry.
    """

    def __init__(self, rate_per_min: int, burst: int):
        self.rate = rate_per_min / 60.0   # tokens per second
        self.burst = float(burst)
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()     # FastAPI runs the sync route in a
        self._last_prune = time.monotonic()  # threadpool -> guard shared state

    def check(self, key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds). retry_after is 0 when allowed."""
        now = time.monotonic()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = _Bucket(self.burst, now)   # new clients start full
                self._buckets[key] = b

            # Refill for the time elapsed since we last saw this key, capped.
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
        """Drop full, idle buckets so memory can't grow without bound.

        Called under the lock. Cheap heuristic: only bothers once a minute and
        only once the dict is non-trivial. A bucket that is full again and
        hasn't been touched in 5 minutes carries no state worth keeping.
        """
        if now - self._last_prune < 60 or len(self._buckets) < 1024:
            return
        self._last_prune = now
        stale = [k for k, b in self._buckets.items()
                 if b.tokens >= self.burst and (now - b.last) > 300]
        for k in stale:
            self._buckets.pop(k, None)


# --- Layer 3: global daily circuit breaker ---------------------------------

class DailyCounter:
    """Count /care calls per UTC day and refuse past `cap`. cap<=0 disables it."""

    def __init__(self, cap: int):
        self.cap = cap
        self._day = None
        self._count = 0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        if self.cap <= 0:
            return True
        # tm_yday (day-of-year) flips at UTC midnight, which is all we need for a
        # daily reset. It repeats once a year — harmless for a rolling counter.
        today = time.gmtime().tm_yday
        with self._lock:
            if today != self._day:
                self._day, self._count = today, 0
            if self._count >= self.cap:
                return False
            self._count += 1
            return True


# Module-level singletons: one shared limiter + counter for the whole process.
_limiter = RateLimiter(RATE_PER_MIN, BURST)
_daily = DailyCounter(DAILY_CAP)

# Print the effective config at import time, matching tools.py's "[tools] ..."
# style. This makes a misconfigured .env obvious immediately: if you didn't mean
# to turn auth on, this line will still say "api-key auth: ON" and tell you so
# before you waste time debugging a 401.
print(
    f"[security] api-key auth: "
    f"{'ON (' + str(len(API_KEYS)) + ' key(s))' if API_KEYS else 'OFF'} | "
    f"rate limit: {RATE_PER_MIN}/min burst {BURST} | "
    f"daily cap: {DAILY_CAP if DAILY_CAP > 0 else 'off'} | "
    f"trust X-Forwarded-For: {TRUST_XFF}"
)


def _client_ip(request) -> str:
    """Best-effort real client IP (see TRUST_XFF note at the top)."""
    if TRUST_XFF:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            # Leftmost entry is the original client, set by the trusted ingress.
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# --- FastAPI dependencies (the only framework-aware part) ------------------

from fastapi import Header, HTTPException, Request  # noqa: E402  (kept local-ish
# to the dependencies so the classes above stay import-light and unit-testable)


def rate_limit(request: Request) -> None:
    """Dependency: global daily backstop, then per-IP token bucket.

    Wire it FIRST on the route so even unauthenticated floods are throttled
    before we bother checking a key.
    """
    if not _daily.allow():
        raise HTTPException(
            status_code=503,
            detail="Daily capacity reached. Please try again tomorrow.",
        )
    allowed, retry = _limiter.check(_client_ip(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down.",
            headers={"Retry-After": str(max(1, ceil(retry)))},
        )


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Dependency: enforce X-API-Key IFF CAREROUTE_API_KEYS is configured.

    hmac.compare_digest is a constant-time compare — it doesn't leak which
    character was wrong via timing. Overkill at this scale, but it's the correct
    habit and free to do.
    """
    if not API_KEYS:            # auth disabled -> endpoint is open
        return
    if x_api_key is None or not any(hmac.compare_digest(x_api_key, k)
                                    for k in API_KEYS):
        raise HTTPException(
            status_code=401, detail="Missing or invalid API key.")


# --- Standalone self-test (no server, no FastAPI needed to READ the logic) --

if __name__ == "__main__":
    # Exercise the pure limiter: a burst of 5 should pass, the 6th should fail.
    rl = RateLimiter(rate_per_min=60, burst=5)   # 1 token/sec, burst 5
    results = [rl.check("1.2.3.4")[0] for _ in range(6)]
    print("First 6 immediate requests (expect 5x True, then False):", results)
    assert results == [True, True, True, True, True, False], results

    # After ~1.1s one token has refilled, so the next call passes again.
    time.sleep(1.1)
    ok, _ = rl.check("1.2.3.4")
    print("After 1.1s refill (expect True):", ok)
    assert ok

    # A different IP has its own bucket — one client can't starve another.
    print("Fresh IP is independent (expect True):", rl.check("9.9.9.9")[0])

    # Daily counter: cap of 2 allows two, refuses the third.
    dc = DailyCounter(cap=2)
    print("Daily cap=2 over 3 calls (expect True, True, False):",
          [dc.allow() for _ in range(3)])

    print("security.py self-test passed.")
