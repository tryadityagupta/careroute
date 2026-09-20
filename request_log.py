"""
request_log.py — structured, one-line-per-interaction logging.

GOAL: see what people ask and how the agent answers — without standing up a
database. Every interaction (a /care request or one /chat turn) is emitted as a
single JSON object on one line (JSONL) through Python's logging, so:
  * locally it prints to the console (redirect it to a file to analyse), and
  * in Azure Container Apps it goes to stdout, which the platform ships to Log
    Analytics automatically — query it with KQL, no extra infrastructure.

Structured (JSON), not free text, so you can later filter and aggregate:
"top symptoms this week", "every turn where the agent found no specialist",
"median latency by backend", etc.

PRIVACY (this matters — you're logging health data)
---------------------------------------------------
This app handles symptoms, medications and location: sensitive personal data.
The moment you persist it, the log store itself becomes sensitive. So by default
this logs the CONTENT you actually need for your stated goal — the user's text,
the agent's answer, the meds mentioned, and a COARSE location — and OMITS direct
identifiers (name, exact coordinates).

  * Coarse location = coordinates rounded (2 dp ≈ ~1.1 km): enough to see
    geographic demand patterns, not enough to pin a person to an address.
  * Set CAREROUTE_LOG_PII=1 to also log name + exact coordinates. If you do,
    treat the log store as sensitive: restrict access and set a retention limit.

Optional durable file: set CAREROUTE_LOG_FILE=/path/app.jsonl to ALSO append
each line to a file (handy for local pandas analysis). Gitignore that path.
For durable analytics in production, forward stdout to Log Analytics (default on
Container Apps) or append to Blob/a DB — a later upgrade, not needed to start.
"""

import json
import logging
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()  # self-sufficient: don't depend on another module loading .env first

_LOG_PII = os.getenv("CAREROUTE_LOG_PII", "0").lower() in ("1", "true", "yes")
# Decimal places to round coordinates to. 2 dp ≈ 1.1 km.
_COORD_DP = int(os.getenv("CAREROUTE_LOG_COORD_DP", "2"))
_LOG_FILE = os.getenv("CAREROUTE_LOG_FILE", "").strip()

# One dedicated logger. propagate=False so uvicorn's root config can neither
# swallow nor duplicate these lines. The handler guard keeps --reload from
# stacking duplicate handlers on re-import.
_logger = logging.getLogger("careroute.interactions")
if not _logger.handlers:
    _fmt = logging.Formatter("%(message)s")  # the message is already JSON
    _stdout = logging.StreamHandler(sys.stdout)
    _stdout.setFormatter(_fmt)
    _logger.addHandler(_stdout)
    if _LOG_FILE:
        # FileHandler won't create parent dirs — make them so a missing logs/
        # folder doesn't crash startup.
        _dir = os.path.dirname(_LOG_FILE)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        _fh.setFormatter(_fmt)
        _logger.addHandler(_fh)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def _coarse(v):
    return round(v, _COORD_DP) if isinstance(v, (int, float)) else None


def log_interaction(*, endpoint, user_text, answer, session_id=None, turn=None,
                    meds=None, name=None, lat=None, lng=None, backend=None,
                    latency_ms=None, status="ok", error=None):
    """Emit one JSON line describing a single request/response.

    Never raises: logging must not take down a request. Any failure to log is
    swallowed (and itself logged at debug level) so the user's answer still ships.
    """
    try:
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": endpoint,
            "status": status,
            "session_id": session_id,
            "turn": turn,
            "backend": backend,
            "latency_ms": latency_ms,
            "user_text": user_text,   # what the user actually asked
            "answer": answer,         # what the agent replied
            "meds": list(meds) if meds else [],
            "loc": {"lat": _coarse(lat), "lng": _coarse(lng)},
        }
        if error is not None:
            rec["error"] = str(error)[:500]
        if _LOG_PII:
            rec["name"] = (name or None)
            rec["loc_exact"] = {"lat": lat, "lng": lng}
        _logger.info(json.dumps(rec, ensure_ascii=False))
    except Exception:  # logging is best-effort; never break the response
        logging.getLogger("careroute").debug("failed to log interaction",
                                             exc_info=True)


if __name__ == "__main__":
    # Self-test: capture what we emit and assert on it. No server needed.
    import io

    buf = io.StringIO()
    for h in list(_logger.handlers):
        _logger.removeHandler(h)
    _logger.addHandler(logging.StreamHandler(buf))

    log_interaction(
        endpoint="/chat", user_text="chest pain and dizzy", answer="Call 112 now.",
        session_id="abc", turn=2, meds=["aspirin"], name="Aditya",
        lat=12.971900, lng=77.641200, backend="openstreetmap", latency_ms=812,
    )
    line = buf.getvalue().strip()
    rec = json.loads(line)  # must be valid JSON
    print(json.dumps(rec, indent=2, ensure_ascii=False))

    assert rec["user_text"] == "chest pain and dizzy"
    assert rec["answer"] == "Call 112 now."
    assert rec["loc"]["lat"] == 12.97 and rec["loc"]["lng"] == 77.64  # coarsened
    # PII omitted by default:
    assert "name" not in rec and "loc_exact" not in rec
    print("\nrequest_log.py self-test passed (PII omitted, coords coarsened).")
