"""
security/gate.py — the gate in front of the paid endpoints.

Three independent, env-toggled layers:
  1. per-IP token bucket            CAREROUTE_RATE_PER_MIN / CAREROUTE_BURST
  2. optional API key (X-API-Key)   CAREROUTE_API_KEYS (empty = open)
  3. optional global daily cap      CAREROUTE_DAILY_CALL_CAP (0 = off)

This is the ONLY framework-aware part of security/. Its two bound methods are
used directly as FastAPI dependencies:

    Depends(gate.rate_limit), Depends(gate.require_api_key)
"""

from __future__ import annotations

import hmac
from math import ceil

from fastapi import Header, HTTPException, Request

from careroute.security.limits import DailyCap, RateLimiter


class RequestGate:
    def __init__(self, limiter: RateLimiter, daily_cap: DailyCap, *,
                 api_keys: tuple[str, ...] = (), trust_xff: bool = True):
        self.limiter = limiter
        self.daily_cap = daily_cap
        self.api_keys = api_keys
        # Behind Azure Container Apps (or the Compose nginx) the client IP
        # arrives in X-Forwarded-For. Trust it ONLY when every public path goes
        # through that proxy; otherwise callers could spoof it to dodge limits.
        self.trust_xff = trust_xff

    def describe(self) -> str:
        auth = f"ON ({len(self.api_keys)} key(s))" if self.api_keys else "OFF"
        cap = self.daily_cap.cap if self.daily_cap.cap > 0 else "off"
        rpm = round(self.limiter.rate * 60)
        return (f"api-key auth: {auth} | rate limit: {rpm}/min burst "
                f"{int(self.limiter.burst)} | daily cap: {cap} | "
                f"trust X-Forwarded-For: {self.trust_xff}")

    def client_ip(self, request: Request) -> str:
        if self.trust_xff:
            xff = request.headers.get("x-forwarded-for")
            if xff:
                return xff.split(",")[0].strip()      # left-most = original client
        return request.client.host if request.client else "unknown"

    # --- FastAPI dependencies ------------------------------------------------
    async def rate_limit(self, request: Request) -> None:
        """Daily backstop, then per-IP bucket. Wire it FIRST so even
        unauthenticated floods are throttled before a key is checked."""
        try:
            daily_ok = await self.daily_cap.allow()
            allowed, retry = await self.limiter.check(self.client_ip(request))
        except Exception as e:     # Redis unreachable
            # FAIL OPEN on purpose: a Redis blip should briefly weaken abuse
            # protection, not take the service down. (Sessions fail CLOSED —
            # without them a conversation can't be served correctly.)
            print(f"[security] rate limiter unavailable, failing open: {e}")
            return
        if not daily_ok:
            raise HTTPException(503, "Daily capacity reached. Please try again tomorrow.")
        if not allowed:
            raise HTTPException(429, "Too many requests. Please slow down.",
                                headers={"Retry-After": str(max(1, ceil(retry)))})

    async def require_api_key(self, x_api_key: str | None = Header(default=None)) -> None:
        """Enforce X-API-Key iff keys are configured (constant-time compare)."""
        if not self.api_keys:
            return
        if x_api_key is None or not any(hmac.compare_digest(x_api_key, k)
                                        for k in self.api_keys):
            raise HTTPException(401, "Missing or invalid API key.")
