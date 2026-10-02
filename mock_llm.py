"""
mock_llm.py — a stand-in for OpenAI's /v1/chat/completions. Free, offline.

Point the app at it and the WHOLE stack runs for real — LangGraph loop, tool
execution, PostGIS, OSRM, Nominatim, Redis, Postgres, answer guard — with only
the model faked:
    OPENAI_API_KEY=mock  OPENAI_BASE_URL=http://mock-llm:9100/v1

Two modes (MOCK_LLM_MODE):

  agent (default)  Behaves like a well-behaved tool-calling model. It reads the
                   transcript, decides the next step (emergency, pharmacy,
                   geocode a named place, specialty search, widen the radius
                   on a miss, retry once on a directory error) and returns
                   real OpenAI tool_calls; once the tools have answered, it
                   writes the final answer FROM the tool results, the way the
                   prompt asks. So a turn costs 2-4 model calls, like the real
                   thing, which is the baseline step 5 has to beat.
  echo             The original statelessness probe: replies
                   "turns_seen=N | meds=..." so test_stateless.py can tell
                   whether history survived a replica hop.

Like the real API it keeps NO conversation state: every decision is a pure
function of the request. Run one mock or ten; the answers are the same.

Latency: uniform in [MOCK_LLM_LATENCY_MIN_S, MOCK_LLM_LATENCY_MAX_S]
(default 0.8-2.0 s, roughly gpt-4o-mini with tools). MOCK_LLM_LATENCY_S pins
it to one value (test_stateless.py uses that).

Faults, switchable at RUNTIME so a load test can break the model mid-run:
    curl -XPOST localhost:9100/admin/fault -d '{"mode":"down"}'       # 503s
    curl -XPOST localhost:9100/admin/fault -d '{"mode":"ratelimit"}'  # 429s
    curl -XPOST localhost:9100/admin/fault -d '{"mode":"slow","slow_s":30}'
    curl -XPOST localhost:9100/admin/fault -d '{"mode":"flaky","error_rate":0.2}'
    curl -XPOST localhost:9100/admin/fault -d '{"mode":"ok"}'
Stats (calls, tokens, per-kind counts) for the cost model:
    curl localhost:9100/admin/stats      curl -XPOST localhost:9100/admin/reset

Run standalone:  uvicorn mock_llm:app --port 9100
"""

import asyncio
import json
import os
import random
import re
import time
import uuid
from collections import Counter

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

MODE = os.getenv("MOCK_LLM_MODE", "agent").strip().lower()
_FIXED = os.getenv("MOCK_LLM_LATENCY_S")
LAT_MIN = float(_FIXED or os.getenv("MOCK_LLM_LATENCY_MIN_S", "0.8"))
LAT_MAX = float(_FIXED or os.getenv("MOCK_LLM_LATENCY_MAX_S", "2.0"))

_FAULT = {"mode": "ok", "error_rate": 0.0, "slow_s": 30.0}
_STATS = Counter()


# ---------------------------------------------------------------------------
# Reading the transcript
# ---------------------------------------------------------------------------
def _content(m) -> str:
    c = m.get("content") or ""
    if isinstance(c, list):                        # content-parts format
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c


_PATIENT = re.compile(r"Patient (\S+)")
_LATLNG = re.compile(r"lat=(-?\d+(?:\.\d+)?), lng=(-?\d+(?:\.\d+)?)")
_AREA = re.compile(r"area=([^,\]]*)")
# The server wraps what the user typed; pull it back out.
_COMPLAINT = re.compile(
    r"(?:reports these symptoms:|now says:)\s*(.*?)\.\s*"
    r"(?:Find the nearest|Treat this on its own)", re.S)


def _turn(messages):
    """(last user text, [(tool_name, args, result), ...] since it)."""
    idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"),
              default=-1)
    user = _content(messages[idx]) if idx >= 0 else ""
    names = {}
    calls = []
    for m in messages[idx + 1:]:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            names[tc.get("id")] = (fn.get("name"), args)
        if m.get("role") == "tool":
            name, args = names.get(m.get("tool_call_id"), (None, {}))
            try:
                result = json.loads(_content(m))
            except ValueError:
                result = _content(m)              # e.g. ToolNode "Error: ..."
            calls.append((name, args, result))
    return user, calls


# ---------------------------------------------------------------------------
# The "model's judgement": keyword triage. Deliberately broader than any
# deterministic pre-pass we build in step 5, so the mock keeps exercising the
# LLM path for complaints a rule table would not cover.
# ---------------------------------------------------------------------------
_EMERGENCY = re.compile(
    r"chest pain|seizure|\bfits?\b|unconscious|fainted|passed out|"
    r"not breathing|can'?t breathe|trouble breathing|stroke|face droop|"
    r"slurred|heavy bleeding|severe bleeding|accident|overdose|suicid", re.I)
_PHARMACY = re.compile(
    r"pharmac|chemist|medicine|tablets?|painkiller|paracetamol|antacid|"
    r"\bors\b|cough syrup|\bbuy\b", re.I)
_SPECIALTIES = [
    (r"\b(child|kid|baby|infant|toddler|son|daughter)\b", "Pediatrics"),
    (r"pregnan|period|menstrua|pcos|gyn", "Gynecology"),
    (r"palpitation|heart|blood pressure|\bbp\b|cardi", "Cardiology"),
    (r"knee|back pain|joint pain|fracture|sprain|bone|ortho", "Orthopedics"),
    (r"skin|rash|acne|itch|eczema", "Dermatology"),
    (r"anxi|depress|panic|insomnia|can'?t sleep|stress", "Psychiatry"),
    (r"\bear\b|throat|sinus|tonsil|nose bleed|blocked nose", "ENT"),
    (r"\beye|vision|blurr", "Ophthalmology"),
    (r"tooth|teeth|gum|dental", "Dentistry"),
    (r"stomach|acidity|diarrh|vomit|constipat|abdomen|abdominal", "Gastroenterology"),
    (r"urin|kidney stone", "Urology"),
    (r"headache|migraine|numb|dizz|tingling", "Neurology"),
    (r"cough|asthma|wheez|breathless", "Pulmonology"),
    (r"diabet|sugar|thyroid", "Endocrinology"),
    (r"arthritis|stiff joints|lupus|autoimmune", "Rheumatology"),
]
_PLACE_EXPLICIT = re.compile(
    r"\b(?:i am|i'm|she is|he is|she's|he's|we are|we're|staying|currently)\s+"
    r"(?:in|at|near)\s+([A-Za-z][\w .'-]{2,40}?)(?=[,.!?]|\s+and\b|$)", re.I)
_PLACE_ANY = re.compile(
    r"\b(?:in|at|near)\s+([A-Z][\w .'-]{2,40}?)(?=[,.!?]|\s+and\b|$)")


def _specialty(text: str) -> str:
    for pat, spec in _SPECIALTIES:
        if re.search(pat, text, re.I):
            return spec
    return "General Medicine"


# ---------------------------------------------------------------------------
# Composing answers from tool results (never inventing names)
# ---------------------------------------------------------------------------
def _dist(f) -> str:
    d = f.get("distance_km")
    s = f"{d} km" if d is not None else "distance unknown"
    if f.get("distance_type") == "road":
        mins = f.get("drive_min_no_traffic")
        s += " by road" + \
            (f", ~{mins} min without traffic" if mins is not None else "")
    elif f.get("distance_type") == "straight_line":
        s += " (straight-line)"
    return s


def _numbered(items) -> str:
    return "\n".join(f"{i}. {f['name']} — {_dist(f)}"
                     for i, f in enumerate(items, 1) if isinstance(f, dict))


def _is_error(result) -> bool:
    return (isinstance(result, dict) and "error" in result) or (
        isinstance(result, str) and result.startswith("Error"))


# ---------------------------------------------------------------------------
# The decision function: transcript in, next step out. Pure.
# ---------------------------------------------------------------------------
def decide(messages, tool_names):
    """Return ("tools", [(name, args), ...]) or ("answer", text)."""
    user, calls = _turn(messages)
    m = _COMPLAINT.search(user)
    complaint = (m.group(1) if m else user).strip()
    pid = (_PATIENT.search(user) or [None, None])[1]
    ll = _LATLNG.search(user)
    lat, lng = (float(ll.group(1)), float(ll.group(2))) if ll else (None, None)
    done = [c[0] for c in calls]

    # 1. A place named in the message: geocode it, then search THERE.
    place_m = _PLACE_EXPLICIT.search(complaint) or (
        _PLACE_ANY.search(complaint) if lat is None else None)
    place = place_m.group(1).strip() if place_m else None
    pending_update = None
    if place and "geocode_place" in tool_names:
        if "geocode_place" not in done:
            return "tools", [("geocode_place", {"place": place})]
        geo = next(r for n, _, r in calls if n == "geocode_place")
        if isinstance(geo, dict) and "lat" in geo:
            lat, lng = geo["lat"], geo["lng"]
            if pid and "update_patient_record" not in done:
                pending_update = ("update_patient_record",
                                  {"patient_id": pid, "lat": lat, "lng": lng,
                                   "area": place})
        elif lat is None and _is_error(geo):
            # Lookup FAILED: that says nothing about whether the place exists.
            return "answer", ("I can't look up place names right now. Please "
                              "share your location, or try again shortly. If "
                              "this is urgent, call 112.")
        elif lat is None:
            return "answer", (f"I couldn't find \"{place}\" on the map. Could you "
                              "give a nearby area, landmark or city?")

    def search(name, args):
        """Batch the record update with the search, as models do."""
        return "tools", ([pending_update] if pending_update else []) + [(name, args)]

    # 2. No location at all: emergencies still get the number, else ask.
    if lat is None:
        if _EMERGENCY.search(complaint):
            return "answer", ("This may be a medical emergency. Call 112 now "
                              "(ambulance: 108). Then tell me where you are so I "
                              "can find the nearest hospital.")
        return "answer", ("Where are you right now? Share your location or "
                          "tell me the area, and I'll find the nearest options.")
    here = {"patient_lat": lat, "patient_lng": lng}

    # 3. Emergency: number first, hospitals second.
    if _EMERGENCY.search(complaint):
        res = [r for n, _, r in calls if n == "get_emergency_help"]
        if not res:
            return search("get_emergency_help", here)
        r = res[-1] if isinstance(res[-1], dict) else {}
        number = r.get("emergency_number", "112")
        text = (f"This may be a medical emergency. Call {number} now — an "
                "ambulance can start care on the way.")
        if r.get("nearest_hospitals"):
            text += ("\n\nNearest hospitals with emergency care:\n"
                     + _numbered(r["nearest_hospitals"]))
        return "answer", text

    # 4. Obtaining medicine: a pharmacy, never a clinic.
    if _PHARMACY.search(complaint):
        res = [r for n, _, r in calls if n == "find_pharmacies"]
        if not res:
            return search("find_pharmacies", here)
        r = res[-1]
        if _is_error(r):
            return "answer", ("The pharmacy directory is temporarily unreachable. "
                              "Please try again shortly.")
        if isinstance(r, dict) and r.get("pharmacies"):
            return "answer", ("Nearest pharmacies:\n" + _numbered(r["pharmacies"])
                              + "\n\nA pharmacist can advise on over-the-counter "
                              "options; I can't recommend a specific medicine.")
        return "answer", ("I couldn't find a pharmacy mapped near you. A wider "
                          "search or a local check may help.")

    # 5. Specialty search, widening on a miss (8 -> 16 -> 30 km), one retry
    #    on a directory error — exactly what the tool hints ask for.
    spec = _specialty(complaint)
    res = [(a, r) for n, a, r in calls if n == "find_providers"]
    if not res:
        return search("find_providers", {"specialty": spec, **here,
                                         "k": 3, "radius_m": 8000})
    args, r = res[-1]
    if _is_error(r):
        if sum(_is_error(x) for _, x in res) < 2:
            return search("find_providers", dict(args))
        return "answer", ("The provider directory is temporarily unreachable, so "
                          "I can't confirm who is nearby right now. Please try "
                          "again shortly. If symptoms are severe, call 112.")
    if isinstance(r, list) and r:
        return "answer", (f"Nearest {spec} options:\n" + _numbered(r)
                          + "\n\nDistances are without traffic.")
    radius = int(args.get("radius_m", 8000))
    if radius < 30000:
        return search("find_providers", {**args, "radius_m": min(radius * 2, 30000)})
    alts = r.get("general_alternatives") if isinstance(r, dict) else None
    if alts:
        return "answer", (f"I couldn't find a confirmed {spec} specialist within "
                          "30 km. Nearby general options (not specialists):\n"
                          + _numbered(alts))
    return "answer", (f"I couldn't find a {spec} specialist nearby. A general "
                      "physician can assess you and refer you.")


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
def _tokens(text: str) -> int:
    return max(1, len(text) // 4)                  # ~4 chars/token, English


def _echo(msgs) -> str:
    users = [m for m in msgs if m.get("role") == "user"]
    meds = re.search(r"current_medications=([^\]]*)\]",
                     _content(users[-1]) if users else "")
    return (f"turns_seen={len(users)} | "
            f"meds={meds.group(1).strip() if meds else '?'}")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    msgs = body.get("messages", [])
    tool_names = {t.get("function", {}).get("name")
                  for t in body.get("tools") or []}
    _STATS["requests"] += 1

    mode = _FAULT["mode"]
    if mode == "down" or (mode == "flaky" and random.random() < _FAULT["error_rate"]):
        _STATS["faults_503"] += 1
        return JSONResponse(status_code=503, content={"error": {
            "message": "mock: model unavailable", "type": "server_error"}})
    if mode == "ratelimit":
        _STATS["faults_429"] += 1
        return JSONResponse(status_code=429, headers={"retry-after": "1"},
                            content={"error": {"message": "mock: rate limited",
                                               "type": "rate_limit_error"}})
    await asyncio.sleep(_FAULT["slow_s"] if mode == "slow"
                        else random.uniform(LAT_MIN, LAT_MAX))

    if MODE == "echo":
        kind, out = "answer", _echo(msgs)
    else:
        kind, out = decide(msgs, tool_names)

    if kind == "tools":
        tool_calls = [{"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                       "function": {"name": n, "arguments": json.dumps(a)}}
                      for n, a in out]
        message = {"role": "assistant",
                   "content": None, "tool_calls": tool_calls}
        finish, completion_text = "tool_calls", json.dumps(tool_calls)
        for n, _ in out:
            _STATS["tool:" + n] += 1
    else:
        message = {"role": "assistant", "content": out}
        finish, completion_text = "stop", out
        _STATS["answers"] += 1

    prompt_text = json.dumps(msgs) + json.dumps(body.get("tools") or [])
    usage = {"prompt_tokens": _tokens(prompt_text),
             "completion_tokens": _tokens(completion_text)}
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    _STATS["prompt_tokens"] += usage["prompt_tokens"]
    _STATS["completion_tokens"] += usage["completion_tokens"]
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock"),
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": usage,
    }


@app.post("/admin/fault")
async def set_fault(request: Request):
    body = await request.json()
    if body.get("mode") not in ("ok", "down", "ratelimit", "slow", "flaky"):
        return JSONResponse(status_code=400, content={"error": "bad mode"})
    _FAULT.update({k: body[k]
                  for k in ("mode", "error_rate", "slow_s") if k in body})
    return dict(_FAULT)


@app.get("/admin/stats")
async def stats():
    return {"mode": MODE, "fault": dict(_FAULT),
            "latency_s": [LAT_MIN, LAT_MAX], **_STATS}


@app.post("/admin/reset")
async def reset():
    _STATS.clear()
    _FAULT.update({"mode": "ok", "error_rate": 0.0, "slow_s": 30.0})
    return {"ok": True}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
