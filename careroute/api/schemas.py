"""api/schemas.py — request bodies."""

from __future__ import annotations

from pydantic import BaseModel


class CareRequest(BaseModel):
    symptoms: str
    # Either the browser's coordinates OR a place the user typed.
    lat: float | None = None
    lng: float | None = None
    location_text: str | None = None
    name: str | None = None
    meds: str | None = None


class ChatRequest(BaseModel):
    # message: this turn's text. session_id is None on turn 1; the response
    # returns one to send back on every follow-up.
    message: str
    lat: float | None = None
    lng: float | None = None
    # A place the user typed. Wins over lat/lng: the user said it on purpose.
    location_text: str | None = None
    name: str | None = None
    meds: str | None = None
    session_id: str | None = None
