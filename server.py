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
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import tools
import os
import uuid
# Import ONLY the LangGraph engine. The previous version imported run_agent from
# BOTH agent and agent_langgraph; the second import silently shadowed the first,
# so agent.py was dead code that merely looked live. Now it's explicit.
from agent_langgraph import run_agent
from security import require_api_key, rate_limit
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
    meds_list = [m.strip() for m in (req.meds or "").replace("\n", ",").split(",")
                 if m.strip()]
    tools._PATIENTS[patient_id] = {
        "patient_id": patient_id,
        "name": (req.name or "").strip() or "Live user",
        "area": "Current location",
        "lat": req.lat,
        "lng": req.lng,
        "history": [],
        "current_medications": meds_list,
    }

    try:
        user_request = (
            f"Patient {patient_id} reports these symptoms: {req.symptoms}. "
            f"Find the nearest appropriate specialists."
        )
        answer = run_agent(user_request)
        return {"answer": answer}
    finally:
        # Always clean up, even if run_agent raised — no leaked patient records.
        tools._PATIENTS.pop(patient_id, None)


CODE_VERSION = "2026-08-18-langgraph-port"
GIT_SHA = os.environ.get("GIT_SHA", "local-dev")


@app.get("/version")
def version():
    return {"version": CODE_VERSION, "commit": GIT_SHA, "backend": tools._BACKEND}
