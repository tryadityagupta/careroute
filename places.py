"""
places.py — REAL provider lookup via Google Places API (Text Search).

This is a drop-in replacement for the dummy find_providers in tools.py.
Same function signature, so agent.py does NOT change — you only swap the
implementation behind the interface.

Setup:
  1. Google Cloud console -> enable "Places API (New)" -> enable billing.
  2. Create an API key, restrict it to the Places API.
  3. export GOOGLE_MAPS_API_KEY=...
Cost: Text Search is a Pro SKU — 35,000 free calls/month in India. A demo
uses a handful, so effectively free. (Set a budget alert anyway.)
"""

import os
import requests
from math import radians, sin, cos, sqrt, atan2

GOOGLE_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"


def _haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km (same proximity math as the dummy version)."""
    R = 6371
    d_lat = radians(lat2 - lat1)
    d_lng = radians(lng2 - lng1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(d_lng / 2) ** 2
    return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 2)


def find_providers(specialty: str, patient_lat: float, patient_lng: float, k: int = 3) -> list:
    """Find the k nearest REAL providers of a specialty via Google Places.

    Identical signature to the dummy version — that's the whole point. The agent
    can't tell the difference; only the data source changed.
    """
    if not GOOGLE_KEY:
        return [{"error": "GOOGLE_MAPS_API_KEY not set"}]

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_KEY,
        # FieldMask = pay only for the fields you ask for. Always set this.
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location",
    }
    body = {
        "textQuery": f"{specialty} doctor",
        # locationBias nudges results toward the patient; radius in metres.
        "locationBias": {
            "circle": {
                "center": {"latitude": patient_lat, "longitude": patient_lng},
                "radius": 10000.0,
            }
        },
        "maxResultCount": 10,
    }

    try:
        resp = requests.post(
            _TEXT_SEARCH_URL, headers=headers, json=body, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return [{"error": f"Places API call failed: {e}"}]

    places = resp.json().get("places", [])
    results = []
    for p in places:
        loc = p.get("location", {})
        lat, lng = loc.get("latitude"), loc.get("longitude")
        if lat is None or lng is None:
            continue
        results.append({
            "name": p.get("displayName", {}).get("text", "Unknown"),
            "facility": p.get("formattedAddress", ""),
            "lat": lat,
            "lng": lng,
            "distance_km": _haversine_km(patient_lat, patient_lng, lat, lng),
        })

    # Places returns by its own relevance; WE re-rank strictly by distance.
    results.sort(key=lambda x: x["distance_km"])
    return results[:k]


if __name__ == "__main__":
    # Patient P001 is in Koramangala. Needs a real key + billing to run.
    out = find_providers("Cardiology", 12.9352, 77.6245, k=3)
    for r in out:
        print(r)
