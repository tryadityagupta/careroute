"""
osm.py — FREE provider lookup via OpenStreetMap's Overpass API.

No API key. No signup. No billing. No credit card. You just POST a query.
Drop-in replacement for find_providers — identical signature, so agent.py and
the registry don't change.

Data-quality reality (and a GREAT interview talking point):
OSM is community-contributed, so specialty tagging is sparse — most facilities
are tagged amenity=hospital/clinic/doctors; few carry
healthcare:speciality=cardiology. The original version handled this by only
RANKING specialty matches first, silently falling back to nearest-anything —
which is how a dental clinic got recommended for anxiety attacks. Now we
FILTER: a miss returns a structured {"match_found": false} result with a hint,
so the agent can retry with a larger radius or fall back honestly.
"""

import requests
from math import radians, sin, cos, sqrt, atan2

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Guardrail: never trust model-supplied arguments blindly. Clamp the radius so
# a hallucinated radius_m=9999999 can't hammer Overpass or blow the timeout.
_MIN_RADIUS_M = 500
_MAX_RADIUS_M = 30000


def _haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km — same proximity math as the other versions."""
    R = 6371
    d_lat = radians(lat2 - lat1)
    d_lng = radians(lng2 - lng1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(d_lng / 2) ** 2
    return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 2)


def find_providers(specialty: str, patient_lat: float, patient_lng: float,
                   k: int = 3, radius_m: int = 8000) -> list | dict:
    """Find the k nearest REAL facilities MATCHING a specialty via OpenStreetMap.

    Same signature as the dummy and Google versions (all three now accept
    radius_m). Returns a list of matches on success. If nothing within
    radius_m matches the specialty, returns a dict with match_found=False, the
    radius searched, a retry hint, and the nearest general facilities — so the
    agent can escalate the radius or fall back HONESTLY, instead of dressing
    up "nearest building" as "right specialist".
    """
    radius_m = max(_MIN_RADIUS_M, min(int(radius_m), _MAX_RADIUS_M))

    # Overpass query: nodes AND ways tagged as healthcare facilities near the
    # point. 'out center tags' gives ways a single center lat/lon so we can
    # treat them like points.
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
        resp = requests.post(_OVERPASS_URL, data={"data": query},
                             headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"Overpass API call failed: {e}"}

    spec = specialty.lower()
    # Stem the specialty so word-form differences still match:
    # 'psychiatry' -> 'psychiatr' matches 'Psychiatric' / 'psychiatrist';
    # 'cardiology' -> 'cardiolog' matches 'Cardiologist'. (rstrip strips any
    # trailing 's'/'y' chars — a cheap heuristic, not real lemmatization.)
    stem = spec.rstrip("sy")

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
        matches_specialty = stem in name.lower() or stem in speciality_tag
        results.append({
            "name": name,
            "facility": tags.get("amenity", "healthcare"),
            "lat": lat,
            "lng": lng,
            "distance_km": _haversine_km(patient_lat, patient_lng, lat, lng),
            "specialty_match": matches_specialty,
        })

    results.sort(key=lambda r: r["distance_km"])
    matches = [r for r in results if r["specialty_match"]]

    if matches:
        return matches[:k]

    # FILTER, don't rank. A miss must be LOUD: return a structured result the
    # model can act on — never a silent nearest-anything list.
    return {
        "match_found": False,
        "reason": f"No facility matching '{specialty}' within {radius_m / 1000:.1f} km.",
        "radius_searched_m": radius_m,
        "hint": ("Retry with a larger radius_m (double it, up to 30000). If it "
                 "still misses, tell the user honestly that no matching "
                 "specialist was found nearby and offer "
                 "nearest_general_facilities as general options instead."),
        "nearest_general_facilities": results[:k],
    }


if __name__ == "__main__":
    # Patient P001 is in Koramangala, Bengaluru. No key needed to run this.
    print("HIT CASE:")
    print(find_providers("Cardiology", 12.9352, 77.6245, k=5))
    print("\nMISS CASE (structured, so the agent can recover):")
    print(find_providers("Rheumatology", 12.9352, 77.6245, k=3, radius_m=1000))
