"""
osm.py — FREE provider lookup via OpenStreetMap's Overpass API.

No API key. No signup. No billing. No credit card. You just POST a query.
Drop-in replacement for find_providers — identical signature, so agent.py and
the registry don't change.

Honest caveat (and a GREAT interview talking point):
OSM is community-contributed, so coverage varies and specialty tagging is
sparse — most facilities are tagged amenity=hospital/clinic/doctors but few
carry healthcare:speciality=cardiology. So we fetch nearby healthcare
facilities and RANK specialty matches first, then fall back to nearest. This
is a real data-quality tradeoff you can speak to: free + open vs. curated.
"""

import os
import requests
from math import radians, sin, cos, sqrt, atan2

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km — same proximity math as the other versions."""
    R = 6371
    d_lat = radians(lat2 - lat1)
    d_lng = radians(lng2 - lng1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(d_lng / 2) ** 2
    return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 2)


def find_providers(specialty: str, patient_lat: float, patient_lng: float,
                   k: int = 3, radius_m: int = 8000) -> list:
    """Find the k nearest REAL healthcare facilities via OpenStreetMap.

    Same signature as the dummy and Google versions. Queries hospitals, clinics,
    and doctors within radius_m of the patient, then ranks specialty matches
    first and nearest-first within that.
    """
    # Overpass query: nodes AND ways tagged as healthcare facilities near the point.
    # 'out center tags' gives ways a single center lat/lon so we can treat them
    # like points.
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"~"hospital|clinic|doctors"](around:{radius_m},{patient_lat},{patient_lng});
      way["amenity"~"hospital|clinic|doctors"](around:{radius_m},{patient_lat},{patient_lng});
    );
    out center tags;
    """

    headers = {"User-Agent": "CareRoute-demo/1.0 (learning project)"}
    try:
        resp = requests.post(_OVERPASS_URL, data={
                             "data": query}, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        return [{"error": f"Overpass API call failed: {e}"}]

    spec = specialty.lower()
    results = []
    for el in resp.json().get("elements", []):
        tags = el.get("tags", {})
        # node has lat/lon directly; way has it under 'center'.
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lng = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lng is None:
            continue
        name = tags.get("name", "Unnamed facility")
        speciality_tag = tags.get("healthcare:speciality", "").lower()
        # Best-effort specialty match: in the name or the speciality tag.
        matches_specialty = spec in name.lower() or spec in speciality_tag
        results.append({
            "name": name,
            "facility": tags.get("amenity", "healthcare"),
            "lat": lat,
            "lng": lng,
            "distance_km": _haversine_km(patient_lat, patient_lng, lat, lng),
            "specialty_match": matches_specialty,
        })

    # Specialty matches first, then nearest. (-True sorts before -False.)
    results.sort(key=lambda r: (not r["specialty_match"], r["distance_km"]))
    return results[:k]


if __name__ == "__main__":
    # Patient P001 is in Koramangala, Bengaluru. No key needed to run this.
    for r in find_providers("Cardiology", 12.9352, 77.6245, k=5):
        print(r)
