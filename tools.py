"""
tools.py - the function the agent is allowed to call.

KEY IDEA: each function here is a "tool". The LLM never touches your data
directly. Instead it *decides* which of these functions to call and with what
arguments; your code runs the function and hands the result back. That hand-off
is the whole mechanic of "tool calling" / "function calling".

The docstrings are written for a reader who is the model - they explain WHEN to
use each tool. We feed these descriptions to the model so it can choose.
"""

import json
import os
from math import radians, sin, cos, sqrt, atan2

# Load the mock data once at import time (small files, fine to keep in memory).
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

with open(os.path.join(_DATA_DIR, "providers.json"), encoding="utf-8") as f:
    _PROVIDERS = json.load(f)

with open(os.path.join(_DATA_DIR, "patients.json"), encoding="utf-8") as f:
    _PATIENTS = json.load(f)


def _haversine_km(lat1, lng1, lat2, lng2):
    """
    Great-circle distance between two lat/lng poitns, in kilometres.

    The standard formula for distance over the Earth's surface - it accounts
    for the planet's curvature, not straight-line on a flat map. Pure math, 
    no external API needed. This is our 'proximity' calculation.
    """

    R = 6371  # Earth's radius in km
    d_lat = radians(lat2 - lat1)
    d_lng = radians(lng2 - lng1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(d_lng / 2) ** 2
    return round(R * 2 * atan2(sqrt(a), sqrt(1-a)), 2)


def get_patient_record(patient_id: str) -> dict:
    """
    Retrieve a patient's clinical record by their patient ID.

    Use this FIRST when you need the patient's location or medical history.
    Returns demographics, location (lat/lng), conditions, and medications.
    """

    record = _PATIENTS.get(patient_id)
    if record is None:
        return {"error": f"No patient found with id {patient_id}"}
    return record


def find_providers(specialty: str, patient_lat: float, patient_lng: float, k: int = 3) -> list:
    """
    Find the k nearest healtcare providers of a given specialty.

    Use this AFTER you know the patient's location and have decided which 
    medical specialty the condition requires (e.g 'Cardiology' for chest pain).
    Returns providers of that specialty sorted nearest-first, each with a
    distance_km field.
    """

    results = []
    for p in _PROVIDERS:
        if p["specialty"].lower() == specialty.lower():
            # Build a NEW dict (a copy) so we never mutate the shared cache.
            results.append({
                **p,
                "distance_km": _haversine_km(patient_lat, patient_lng, p["lat"], p["lng"]),
            })

    results.sort(key=lambda p: p["distance_km"])
    return results[:k]


# A registry so the agent loop can look up a tool by name and call it.
TOOL_REGISTRY = {
    "get_patient_record": get_patient_record,
    "find_providers": find_providers,
}

if __name__ == "__main__":
    # Quick self-test of the deterministic tools (no LLM involved).
    rec = get_patient_record("P001")
    print("Patient:", rec["name"], "| location:", rec["area"])
    near = find_providers("Cardiology", rec["lat"], rec["lng"], k=3)
    print("Nearest cardiologists:")

    for p in near:
        print(f" {p['name']:18} {p['facility']:28} {p['distance_km']} km")
