"""
osm.py — FREE provider lookup via OpenStreetMap's Overpass API.

No API key. No signup. No billing. No credit card. You just POST a query.
Drop-in replacement for find_providers — identical signature, so agent.py and
the registry don't change.

Data-quality reality (and a GREAT interview talking point):
OSM is community-contributed, so specialty tagging is sparse — most facilities
are tagged amenity=hospital/clinic/doctors; few carry
healthcare:speciality=cardiology. Two bugs came out of that:

  1. The original version only RANKED specialty matches first, silently falling
     back to nearest-anything — which is how a dental clinic got recommended
     for anxiety attacks. Fixed by FILTERING.

  2. The first fix returned the nearest general facilities INSIDE the miss
     payload, "for context". The model listed them as specialists anyway,
     despite a prompt instruction not to. Fixed structurally: a miss now
     carries a COUNT, never names. If the model wants general facilities it
     must deliberately call find_general_facilities, whose results are
     labelled as non-specialist at the item level.

Lesson: a prompt instruction is advisory; a data structure is enforced.
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


def _clamp_radius(radius_m):
    return max(_MIN_RADIUS_M, min(int(radius_m), _MAX_RADIUS_M))


def _stem(specialty: str) -> str:
    """Crude stem so word forms match: 'psychiatry' -> 'psychiatr' matches
    'Psychiatric'/'psychiatrist'; 'cardiology' -> 'cardiolog' matches
    'Cardiologist'. A heuristic, not real lemmatization — say so if asked."""
    return specialty.lower().rstrip("sy")


def _fetch_nearby(patient_lat, patient_lng, radius_m, specialty=None):
    """Shared Overpass fetch. Returns (facilities, error_or_None).

    Both find_providers and find_general_facilities use this, so the query and
    the distance math live in exactly one place.
    """
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
        return None, {"error": f"Overpass API call failed: {e}"}

    stem = _stem(specialty) if specialty else None
    out = []
    for el in resp.json().get("elements", []):
        tags = el.get("tags", {})
        # node has lat/lon directly; way has it under 'center'.
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lng = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lng is None:
            continue
        name = tags.get("name", "Unnamed facility")
        item = {
            "name": name,
            "facility": tags.get("amenity", "healthcare"),
            "lat": lat,
            "lng": lng,
            "distance_km": _haversine_km(patient_lat, patient_lng, lat, lng),
        }
        if stem:
            # Bengaluru OSM facilities often carry healthcare:speciality as a
            # long semicolon list, sometimes batch-pasted across many POIs.
            # Substring-matching the whole blob confirmed a DENTAL clinic for
            # Psychiatry. So: match token-by-token, and RECORD THE EVIDENCE —
            # matched_via + the raw tag — so no match is ever opaque again.
            raw_tag = tags.get("healthcare:speciality", "")
            tokens = [t.strip().lower()
                      for t in raw_tag.split(";") if t.strip()]
            if stem in name.lower():
                item["specialty_match"] = True
                item["matched_via"] = "name"
            elif any(stem in t for t in tokens):
                item["specialty_match"] = True
                item["matched_via"] = "speciality_tag"
                item["speciality_tag"] = raw_tag       # the evidence, verbatim
                item["_tag_tokens"] = len(tokens)      # mega-list detector
            else:
                item["specialty_match"] = False
        out.append(item)

    out.sort(key=lambda r: r["distance_km"])
    return out, None


def _dedupe(items):
    """OSM stores big facilities as a node AND a way; the same name then eats
    two result slots (NIMHANS showed up twice). Keep the nearest instance.
    Unnamed facilities are left alone — collapsing distinct ones would lie."""
    seen, out = set(), []
    for f in items:                       # callers pass distance-sorted lists
        key = f["name"].lower()
        if key != "unnamed facility" and key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def find_providers(specialty: str, patient_lat: float, patient_lng: float,
                   k: int = 3, radius_m: int = 8000) -> list | dict:
    """Find the k nearest REAL facilities MATCHING a specialty via OpenStreetMap.

    Returns a list of confirmed specialty matches on success. On a miss returns
    a dict with match_found=False and a COUNT of nearby general facilities —
    deliberately NOT their names, so they cannot be passed off as specialists.
    """
    radius_m = _clamp_radius(radius_m)
    facilities, err = _fetch_nearby(
        patient_lat, patient_lng, radius_m, specialty)
    if err:
        return err

    matches = _dedupe([f for f in facilities if f["specialty_match"]])
    if matches:
        # Rank by evidence quality before distance: a specialty in the NAME
        # beats a speciality tag, and a dedicated tag (few entries) beats a
        # 20-specialty mega-list — which is how a dental clinic can "offer"
        # psychiatry. Distance breaks ties.
        matches.sort(key=lambda f: (f.get("matched_via") != "name",
                                    f.get("_tag_tokens", 0),
                                    f["distance_km"]))
        top = matches[:k]
        for f in top:
            f.pop("_tag_tokens", None)
        return top

    return {
        "match_found": False,
        "reason": f"No facility matching '{specialty}' within {radius_m / 1000:.1f} km.",
        "radius_searched_m": radius_m,
        "general_facilities_nearby": len(facilities),  # a COUNT, never names
        "hint": ("Retry with a larger radius_m (double it, up to 30000). If it "
                 "still misses, tell the user plainly that no matching "
                 "specialist was found. You may call find_general_facilities "
                 "to offer non-specialist options, but you must not describe "
                 "anything it returns as a specialist."),
    }


def find_general_facilities(patient_lat: float, patient_lng: float,
                            k: int = 3, radius_m: int = 8000) -> list | dict:
    """Nearest healthcare facilities of ANY type — explicitly NOT specialists.

    Only call this after find_providers has reported match_found=false and you
    have told the user no specialist was found. Every item is labelled
    is_specialist_match=false.
    """
    radius_m = _clamp_radius(radius_m)
    facilities, err = _fetch_nearby(patient_lat, patient_lng, radius_m)
    if err:
        return err
    if not facilities:
        return {"match_found": False,
                "reason": f"No healthcare facilities at all within {radius_m / 1000:.1f} km."}

    facilities = _dedupe(facilities)
    for f in facilities[:k]:
        f["is_specialist_match"] = False
    return {
        "disclaimer": ("These are general healthcare facilities, NOT verified "
                       "specialists. Present them only as general options."),
        "facilities": facilities[:k],
    }


if __name__ == "__main__":
    # Patient P001 is in Koramangala, Bengaluru. No key needed to run this.
    print("HIT CASE:")
    print(find_providers("Cardiology", 12.9352, 77.6245, k=5))
    print("\nMISS CASE (count only — no names to misuse):")
    print(find_providers("Rheumatology", 12.9352, 77.6245, k=3, radius_m=1000))
    print("\nEXPLICIT FALLBACK (labelled non-specialist):")
    print(find_general_facilities(12.9352, 77.6245, k=3))
