"""
server.py — FastAPI backend that wraps the agent for the web demo.

You are the patient: describe symptoms, your browser shares your real location,
the agent finds real nearby providers. The only synthetic piece is "the patient"
(you), so the get_patient_record tool still demonstrates record retrieval.

Run it:
    USE_REAL_PROVIDERS=osm uvicorn server:app --reload
Then open http://localhost:8000
"""


from agent import run_agent
import tools
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi import FastAPI
import os
from dotenv import load_dotenv

# Load OPENAI_API_KEY from .env BEFORE importing agent — agent.py builds the
# OpenAI client at import time, so the key must be in the environment first.
load_dotenv()


app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class CareRequest(BaseModel):
    symptoms: str
    lat: float
    lng: float


@app.get("/")
def home():
    return FileResponse("index.html")


# NOTE: this is a SYNC def, not async. run_agent makes blocking OpenAI calls,
# so FastAPI runs this in a threadpool — that keeps the blocking call off the
# event loop. (Same async lesson from your portfolio, applied the other way.)
@app.post("/care")
def care(req: CareRequest):
    # Register the live user as a synthetic patient so get_patient_record works
    # unchanged. "LIVE" is fine for a single-user demo; multi-user would need a
    # unique id per request (last-writer-wins otherwise).
    tools._PATIENTS["LIVE"] = {
        "patient_id": "LIVE",
        "name": "Live user",
        "area": "Current location",
        "lat": req.lat,
        "lng": req.lng,
        "history": [],
        "current_medications": [],
    }

    user_request = (
        f"Patient LIVE reports these symptoms: {req.symptoms}. "
        f"Find the nearest appropriate specialists."
    )
    answer = run_agent(user_request)
    return {"answer": answer}
