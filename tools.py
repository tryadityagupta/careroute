"""
tools.py — the functions the agent is allowed to call.

KEY IDEA: each function here is a "tool". The LLM never touches your data
directly. Instead it *decides* which of these functions to call and with what
arguments; your code runs the function and hands the result back. That hand-off
is the whole mechanic of "tool calling" / "function calling".

The docstrings are written for a reader who is the model — they explain WHEN to
use each tool. We feed these descriptions to the model so it can choose.
"""

from emergency import get_emergency_help  # always available, all backends
import json
import os
import requests
from math import radians, sin, cos, sqrt, atan2

from dotenv import load_dotenv

from data_source import load_patients, load_providers

# Load .env BEFORE reading any flags below. Previously USE_REAL_PROVIDERS was
# read here at import time, BEFORE anything had called load_dotenv — so it only
# worked as a real shell variable. Loading .env first means both work now.
load_dotenv()

# Load the mock data once at import time (small files, fine to keep in memory).
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# with open(os.path.join(_DATA_DIR, "providers.json"), encoding="utf-8") as f:
#     _PROVIDERS = json.load(f)

# with open(os.path.join(_DATA_DIR, "patients.json"), encoding="utf-8") as f:
#     _PATIENTS = json.load(f)

_PROVIDERS = load_providers()
_PATIENTS = load_patients()


def _haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance between two lat/lng points, in kilometres.

    The standard formula for distance over the Earth's surface — it accounts
    for the planet's curvature, not straight-line on a flat map. Pure math,
    no external API needed. This is our 'proximity' calculation.
    """
    R = 6371  # Earth's radius in km
    d_lat = radians(lat2 - lat1)
    d_lng = radians(lng2 - lng1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(d_lng / 2) ** 2
    return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 2)


def get_patient_record(patient_id: str) -> dict:
    """Retrieve a patient's clinical record by their patient ID.

    Use this FIRST when you need the patient's location or medical history.
    Returns demographics, location (lat/lng), conditions, and medications.
    """
    record = _PATIENTS.get(patient_id)
    if record is None:
        return {"error": f"No patient found with id {patient_id}"}
    return record


# A named place resolves to the same point every time, and Nominatim's usage
# policy asks callers to be gentle — so cache and never look one up twice.
_GEOCODE_CACHE: dict = {}
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def geocode_place(place: str) -> dict:
    """Resolve a place NAME (city, area, address) to coordinates.

    Call this whenever the user gives a location by name — e.g. "she is in
    Guwahati" — instead of trusting the patient's stored coordinates. Feed the
    returned lat/lng into find_providers / find_general_facilities /
    find_pharmacies so the search actually happens THERE. Returns
    {lat, lng, display_name}, or an error. Never guess coordinates yourself, and
    never claim a result is in a city you did not resolve here.
    """
    key = (place or "").strip().lower()
    if not key:
        return {"error": "empty place"}
    if key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[key]
    try:
        resp = requests.get(
            _NOMINATIM_URL,
            params={"q": place, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": "CareRoute/1.0 (care-routing demo)"},
            timeout=10,
        )
        resp.raise_for_status()
        hits = resp.json()
    except requests.RequestException as e:
        return {"error": f"Geocoding failed: {e}"}
    if not hits:
        return {"match_found": False,
                "reason": f"Could not find a place named '{place}'."}
    top = hits[0]
    out = {"lat": float(top["lat"]), "lng": float(top["lon"]),
           "display_name": top.get("display_name", place)}
    _GEOCODE_CACHE[key] = out
    return out


def update_patient_record(patient_id: str, name: str = None,
                          medications=None, lat: float = None,
                          lng: float = None, area: str = None) -> dict:
    """Save details the user states in conversation onto the patient's record.

    Use when the user gives — in their message — the patient's NAME, their
    MEDICATIONS, or a corrected LOCATION. For a location given by name, call
    geocode_place first, then pass its lat/lng here (plus area=<the place>).
    Only the fields you pass are changed. Returns the updated record.
    """
    rec = _PATIENTS.get(patient_id)
    if rec is None:
        return {"error": f"No patient found with id {patient_id}"}
    if name:
        rec["name"] = name.strip()
    if medications is not None:
        meds = ([m.strip() for m in medications.split(",")]
                if isinstance(medications, str) else list(medications))
        existing = rec.setdefault("current_medications", [])
        for m in meds:
            if m and m not in existing:
                existing.append(m)
    if lat is not None and lng is not None:
        rec["lat"], rec["lng"] = float(lat), float(lng)
    if area:
        rec["area"] = area.strip()
    return rec


def find_providers(specialty: str, patient_lat: float, patient_lng: float,
                   k: int = 3, radius_m: int = 8000) -> list | dict:
    """Find the k nearest healthcare providers of a given specialty.

    Use this AFTER you know the patient's location and have decided which
    medical specialty the condition requires (e.g. 'Cardiology' for chest
    pain). Returns providers sorted nearest-first, each with a distance_km
    field. On a miss, returns match_found=False with a reason and a hint about
    how to recover (a different specialty, or a larger radius_m).
    """
    spec_matches = []
    for p in _PROVIDERS:
        if p["specialty"].lower() == specialty.lower():
            # Build a NEW dict (a copy) so we never mutate the shared cache.
            spec_matches.append({
                **p,
                "distance_km": _haversine_km(patient_lat, patient_lng, p["lat"], p["lng"]),
            })

    # Miss type 1: the specialty doesn't exist in this directory at all.
    # A bigger radius can't fix that — so tell the model what IS available and
    # let it re-map (e.g. Dentistry -> General Medicine). This is a runtime
    # version of an enum constraint, and it works across swappable backends.
    if not spec_matches:
        return {
            "match_found": False,
            "reason": f"No providers with specialty '{specialty}' exist in the directory.",
            "available_specialties": sorted({p["specialty"] for p in _PROVIDERS}),
            "hint": ("Pick the most clinically appropriate specialty from "
                     "available_specialties and call find_providers again."),
        }

    in_radius = [p for p in spec_matches if p["distance_km"]
                 * 1000 <= radius_m]

    # Miss type 2: the specialty exists, just not this close.
    # A bigger radius CAN fix this one — say so explicitly.
    if not in_radius:
        nearest = min(spec_matches, key=lambda p: p["distance_km"])
        return {
            "match_found": False,
            "reason": (f"No {specialty} providers within {radius_m / 1000:.1f} km "
                       f"(nearest is {nearest['distance_km']} km away)."),
            "hint": "Call find_providers again with a larger radius_m.",
        }

    in_radius.sort(key=lambda p: p["distance_km"])
    return in_radius[:k]


def find_general_facilities(patient_lat: float, patient_lng: float,
                            k: int = 3, radius_m: int = 8000) -> dict:
    """Nearest providers of ANY specialty — explicitly NOT specialist matches.

    Only call this after find_providers has reported match_found=false and you
    have told the user no specialist was found. Every item is labelled
    is_specialist_match=false, and the payload carries a disclaimer, because a
    prompt instruction alone was not enough to stop the model presenting these
    as specialists.
    """
    scored = [
        {**p, "distance_km": _haversine_km(patient_lat, patient_lng, p["lat"], p["lng"]),
         "is_specialist_match": False}
        for p in _PROVIDERS
    ]
    scored = [p for p in scored if p["distance_km"] * 1000 <= radius_m]
    if not scored:
        return {"match_found": False,
                "reason": f"No facilities of any kind within {radius_m / 1000:.1f} km."}
    scored.sort(key=lambda p: p["distance_km"])
    return {
        "disclaimer": ("These are general healthcare providers, NOT matches for "
                       "the requested specialty. Present them only as general "
                       "options, never as specialists."),
        "facilities": scored[:k],
    }


def find_pharmacies(patient_lat: float, patient_lng: float,
                    k: int = 3, radius_m: int = 8000) -> dict:
    """Nearest pharmacies — the dummy backend has no pharmacy data.

    The demo JSON is a provider directory, not a pharmacy directory, so this is
    honest about having nothing rather than pointing at a clinic. Real pharmacy
    lookup lives in the osm/google backends (USE_REAL_PROVIDERS).
    """
    return {
        "match_found": False,
        "reason": "Pharmacy lookup needs a real backend (USE_REAL_PROVIDERS=osm or google).",
    }


# Swap to a REAL provider lookup by setting USE_REAL_PROVIDERS. This rebinds
# find_providers to the real implementation (identical signature), so agent.py
# and the registry below DON'T change — that's the interface lesson: the agent
# can't tell the data source changed.
_provider_mode = os.getenv("USE_REAL_PROVIDERS", "").lower()
if _provider_mode in ("google", "1"):
    from places import find_providers, find_general_facilities, find_pharmacies  # noqa: F811
    _BACKEND = "google-places"
elif _provider_mode == "osm":
    from osm import find_providers, find_general_facilities, find_pharmacies      # noqa: F811
    _BACKEND = "openstreetmap"
else:
    _BACKEND = "dummy-json"
print(f"[tools] find_providers backend: {_BACKEND}")

# A registry so the agent loop can look up a tool by name and call it.

TOOL_REGISTRY = {
    "get_patient_record": get_patient_record,
    "update_patient_record": update_patient_record,
    "geocode_place": geocode_place,
    "find_providers": find_providers,
    "find_general_facilities": find_general_facilities,
    "find_pharmacies": find_pharmacies,
    "get_emergency_help": get_emergency_help,
}


if __name__ == "__main__":
    # Quick self-test of the deterministic tools (no LLM involved).
    # Run with USE_REAL_PROVIDERS unset to test the dummy paths below.
    rec = get_patient_record("P001")
    print("Patient:", rec["name"], "| location:", rec["area"])
    near = find_providers("Cardiology", rec["lat"], rec["lng"], k=3)
    print("Nearest cardiologists:")
    for p in near:
        print(f"  {p['name']:18} {p['facility']:28} {p['distance_km']} km")
    # Exercise both structured-miss paths (the agent's recovery signals):
    print("Unknown specialty ->",
          find_providers("Dentistry", rec["lat"], rec["lng"]))
    print("Radius too small  ->",
          find_providers("Neurology", rec["lat"], rec["lng"], radius_m=2000))
    print("Explicit fallback ->",
          find_general_facilities(rec["lat"], rec["lng"], k=2))
