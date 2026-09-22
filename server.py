"""
server.py — FastAPI backend that wraps the agent for the web demo.

You are the patient: describe symptoms, your browser shares your real location,
the agent finds real nearby providers. The only synthetic piece is "the patient"
(you), so the get_patient_record tool still demonstrates record retrieval.

Run it (PowerShell):
    $env:USE_REAL_PROVIDERS="osm"; uvicorn server:app --reload
Run it (bash/zsh):
    USE_REAL_PROVIDERS=osm uvicorn server:app --reload
Then open http://localhost:8000
"""

from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import tools
import os
import uuid
import time
import request_log
# Import ONLY the LangGraph engine. The previous version imported run_agent from
# BOTH agent and agent_langgraph; the second import silently shadowed the first,
# so agent.py was dead code that merely looked live. Now it's explicit.
# run_agent = single-shot (/care); continue_conversation = multi-turn (/chat).
from agent_langgraph import run_agent, continue_conversation
from security import require_api_key, rate_limit
import sessions
from dotenv import load_dotenv
load_dotenv()  # belt-and-braces; tools.py also loads .env before reading flags


app = FastAPI()

# Lock CORS to your real web origin(s) in production by setting
# CAREROUTE_ALLOWED_ORIGINS to a comma-separated list (e.g.
# "https://careroute.app"). Defaults to "*" so local dev is unchanged.
_origins = [o.strip() for o in os.getenv("CAREROUTE_ALLOWED_ORIGINS", "*").split(",")
            if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware, allow_origins=_origins, allow_methods=["*"], allow_headers=["*"]
)


class CareRequest(BaseModel):
    symptoms: str
    lat: float
    lng: float
    name: str | None = None
    meds: str | None = None


class ChatRequest(BaseModel):
    # message = this turn's text (symptoms on turn 1, a follow-up afterwards).
    # session_id is None on the first turn; the response returns one to send back
    # on every follow-up. lat/lng are required only to START a conversation.
    message: str
    lat: float | None = None
    lng: float | None = None
    name: str | None = None
    meds: str | None = None
    session_id: str | None = None


def _parse_meds(meds: str | None) -> list[str]:
    """Split a free-text meds field (commas or newlines) into a clean list."""
    return [m.strip() for m in (meds or "").replace("\n", ",").split(",") if m.strip()]


@app.get("/")
def home():
    return FileResponse("index.html")


# NOTE: this is a SYNC def, not async. run_agent makes blocking OpenAI calls,
# so FastAPI runs this in a threadpool — that keeps the blocking call off the
# event loop.
@app.post("/care")
def care(
    req: CareRequest,
    # Two gates run before the handler body. Order matters: rate_limit first so
    # even unauthenticated floods are throttled; then the (optional) API-key
    # check. Both are no-ops unless configured — see security.py.
    _rl: None = Depends(rate_limit),
    _auth: None = Depends(require_api_key),
):
    # Register the live user as a synthetic patient so get_patient_record works
    # unchanged. Each request now gets a UNIQUE id. The old hardcoded "LIVE" key
    # was a single shared slot in a module-level dict: two concurrent users
    # overwrote each other (last-writer-wins), so the agent could read the wrong
    # person's location. A per-request uuid isolates them; the finally-block
    # deletes it so the dict can't grow without bound. Name and medications are
    # OPTIONAL — empty fields fall back to the anonymous behaviour.
    patient_id = "LIVE-" + uuid.uuid4().hex[:12]
    meds_list = _parse_meds(req.meds)
    tools._PATIENTS[patient_id] = {
        "patient_id": patient_id,
        "name": (req.name or "").strip() or "Live user",
        "area": "Current location",
        "lat": req.lat,
        "lng": req.lng,
        "history": [],
        "current_medications": meds_list,
    }

    t0 = time.perf_counter()
    try:
        user_request = (
            f"Patient {patient_id} reports these symptoms: {req.symptoms}. "
            f"Find the nearest appropriate specialists."
        )
        answer = run_agent(user_request)
        # may have been updated by the agent
        rec = tools._PATIENTS.get(patient_id, {})
        request_log.log_interaction(
            endpoint="/care", user_text=req.symptoms, answer=answer,
            meds=rec.get("current_medications", meds_list),
            name=rec.get("name", req.name),
            lat=rec.get("lat", req.lat), lng=rec.get("lng", req.lng),
            backend=tools._BACKEND,
            latency_ms=round((time.perf_counter() - t0) * 1000),
        )
        return {"answer": answer}
    except Exception as exc:
        rec = tools._PATIENTS.get(patient_id, {})
        request_log.log_interaction(
            endpoint="/care", user_text=req.symptoms, answer=None,
            meds=rec.get("current_medications", meds_list),
            name=rec.get("name", req.name),
            lat=rec.get("lat", req.lat), lng=rec.get("lng", req.lng),
            backend=tools._BACKEND,
            latency_ms=round((time.perf_counter() - t0) * 1000),
            status="error", error=exc,
        )
        raise
    finally:
        # Always clean up, even if run_agent raised — no leaked patient records.
        tools._PATIENTS.pop(patient_id, None)


@app.post("/chat")
def chat(
    req: ChatRequest,
    _rl: None = Depends(rate_limit),
    _auth: None = Depends(require_api_key),
):
    """Multi-turn sibling of /care.

    The FIRST call omits session_id and starts a conversation; the response
    returns a session_id the client sends back on every follow-up. The agent
    then remembers the thread (via the LangGraph checkpointer) and the SAME
    patient record persists, so medications/history added in later turns refine
    the routing instead of starting from scratch.

    Unlike /care, the patient record is NOT deleted after the request — it lives
    for the length of the session and is swept when the session expires.
    """
    # Opportunistic cleanup: expire idle sessions and free their patient records.
    for pid in sessions.sweep():
        tools._PATIENTS.pop(pid, None)

    if not req.message.strip():
        raise HTTPException(
            status_code=400, detail="message must not be empty.")

    sess = sessions.get_session(req.session_id)

    if sess is None:
        # --- New conversation (turn 1) ---------------------------------------
        if req.lat is None or req.lng is None:
            raise HTTPException(
                status_code=400,
                detail="lat and lng are required to start a conversation.",
            )
        patient_id = "LIVE-" + uuid.uuid4().hex[:12]
        session_id = sessions.create_session(patient_id)
        turn = 1
        tools._PATIENTS[patient_id] = {
            "patient_id": patient_id,
            "name": (req.name or "").strip() or "Live user",
            "area": "Current location",
            "lat": req.lat,
            "lng": req.lng,
            "history": [],
            "current_medications": _parse_meds(req.meds),
        }
        user_message = (
            f"Patient {patient_id} reports these symptoms: {req.message}. "
            f"Find the nearest appropriate specialists."
        )
    else:
        # --- Follow-up turn ---------------------------------------------------
        session_id = req.session_id
        patient_id = sess["patient_id"]
        turn = sessions.next_turn(session_id)
        rec = tools._PATIENTS.get(patient_id)
        if rec is None:
            # The record was swept while the client still held the id.
            raise HTTPException(
                status_code=409,
                detail="Session expired. Please start a new conversation.",
            )
        # Merge any newly supplied details so the agent can refine its routing.
        for med in _parse_meds(req.meds):
            if med not in rec["current_medications"]:
                rec["current_medications"].append(med)
        if (req.name or "").strip():
            rec["name"] = req.name.strip()
        if req.lat is not None and req.lng is not None:
            rec["lat"], rec["lng"] = req.lat, req.lng
        user_message = (
            f"Patient {patient_id} (same conversation) now says: {req.message}. "
            f"Treat this on its own merits — it may add detail to the earlier "
            f"complaint or raise a NEW need (e.g. wanting painkillers, which "
            f"means a pharmacy). Run the search that fits THIS message. Their "
            f"record may have been updated, so re-check it with "
            f"get_patient_record if relevant."
        )

    t0 = time.perf_counter()
    try:
        answer = continue_conversation(user_message, thread_id=session_id)
        rec = tools._PATIENTS.get(patient_id, {})  # reflects agent updates
        request_log.log_interaction(
            endpoint="/chat", user_text=req.message, answer=answer,
            session_id=session_id, turn=turn,
            meds=rec.get("current_medications", _parse_meds(req.meds)),
            name=rec.get("name", req.name),
            lat=rec.get("lat", req.lat), lng=rec.get("lng", req.lng),
            backend=tools._BACKEND,
            latency_ms=round((time.perf_counter() - t0) * 1000),
        )
        return {"session_id": session_id, "answer": answer}
    except Exception as exc:
        rec = tools._PATIENTS.get(patient_id, {})
        request_log.log_interaction(
            endpoint="/chat", user_text=req.message, answer=None,
            session_id=session_id, turn=turn,
            meds=rec.get("current_medications", _parse_meds(req.meds)),
            name=rec.get("name", req.name),
            lat=rec.get("lat", req.lat), lng=rec.get("lng", req.lng),
            backend=tools._BACKEND,
            latency_ms=round((time.perf_counter() - t0) * 1000),
            status="error", error=exc,
        )
        raise


CODE_VERSION = "2026-08-18-langgraph-port"
GIT_SHA = os.environ.get("GIT_SHA", "local-dev")


@app.get("/version")
def version():
    return {"version": CODE_VERSION, "commit": GIT_SHA, "backend": tools._BACKEND}
