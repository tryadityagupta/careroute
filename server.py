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
from fastapi import FastAPI
from pydantic import BaseModel
import tools
import os
from agent import run_agent
from agent_langgraph import run_agent
from dotenv import load_dotenv
load_dotenv()  # belt-and-braces; tools.py also loads .env before reading flags


app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
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
def care(req: CareRequest):
    # Register the live user as a synthetic patient so get_patient_record works
    # unchanged. "LIVE" is fine for a single-user demo; multi-user would need a
    # unique id per request (last-writer-wins otherwise). Name and medications
    # are OPTIONAL — empty fields fall back to the anonymous behaviour.
    meds_list = [m.strip() for m in (req.meds or "").replace("\n", ",").split(",")
                 if m.strip()]
    tools._PATIENTS["LIVE"] = {
        "patient_id": "LIVE",
        "name": (req.name or "").strip() or "Live user",
        "area": "Current location",
        "lat": req.lat,
        "lng": req.lng,
        "history": [],
        "current_medications": meds_list,
    }

    user_request = (
        f"Patient LIVE reports these symptoms: {req.symptoms}. "
        f"Find the nearest appropriate specialists."
    )
    answer = run_agent(user_request)
    return {"answer": answer}


CODE_VERSION = "2026-08-18-langgraph-port"
GIT_SHA = os.environ.get("GIT_SHA", "local-dev")


@app.get("/version")
def version():
    return {"version": CODE_VERSION, "commit": GIT_SHA, "backend": tools._BACKEND}
