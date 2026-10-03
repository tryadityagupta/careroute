"""
observability/interaction_log.py — one JSON line per interaction.

Every /care request or /chat turn becomes one JSON object on stdout (and
optionally a file). On Azure Container Apps stdout ships to Log Analytics, so
"top symptoms this week" or "median latency by backend" is a KQL query.

PRIVACY: this app handles symptoms, medications and location. By default the
log keeps what analysis needs — the user's text, the answer, meds, a COARSE
location (2 dp ~ 1.1 km) — and omits direct identifiers (name, exact
coordinates). CAREROUTE_LOG_PII=1 adds them; then treat the log store as
sensitive (restricted access, retention limit).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time


class InteractionLogger:
    LOGGER_NAME = "careroute.interactions"

    def __init__(self, *, log_pii: bool = False, coord_dp: int = 2, log_file: str = "",
                 stream=None):
        self.log_pii = log_pii
        self.coord_dp = coord_dp
        self._logger = logging.getLogger(self.LOGGER_NAME)
        # propagate=False: uvicorn's root config can neither swallow nor
        # duplicate these lines. Handlers are (re)built per instance so a test
        # can capture output by passing a stream.
        self._logger.handlers.clear()
        fmt = logging.Formatter("%(message)s")         # the message is JSON
        h = logging.StreamHandler(stream or sys.stdout)
        h.setFormatter(fmt)
        self._logger.addHandler(h)
        if log_file:
            if os.path.dirname(log_file):
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            self._logger.addHandler(fh)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    def _coarse(self, v):
        return round(v, self.coord_dp) if isinstance(v, (int, float)) else None

    def log(self, *, endpoint, user_text, answer, session_id=None, turn=None,
            meds=None, name=None, lat=None, lng=None, backend=None,
            latency_ms=None, status="ok", error=None, trace=None) -> None:
        """Never raises: logging must not take down a request."""
        try:
            rec = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "endpoint": endpoint, "status": status,
                "session_id": session_id, "turn": turn, "backend": backend,
                "latency_ms": latency_ms,
                "user_text": user_text, "answer": answer,
                "meds": list(meds) if meds else [],
                "loc": {"lat": self._coarse(lat), "lng": self._coarse(lng)},
            }
            if trace is not None:
                # What the agent DID this turn (args already PII-trimmed).
                rec["llm_calls"] = trace.get("llm_calls")
                rec["tools"] = trace.get("tools", [])
            if error is not None:
                rec["error"] = str(error)[:500]
            if self.log_pii:
                rec["name"] = name or None
                rec["loc_exact"] = {"lat": lat, "lng": lng}
            self._logger.info(json.dumps(rec, ensure_ascii=False))
        except Exception:
            logging.getLogger("careroute").debug("failed to log interaction", exc_info=True)
