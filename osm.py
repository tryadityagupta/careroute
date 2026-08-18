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

import json
import os
import time
from math import radians, sin, cos, sqrt, atan2

import requests

# Free public instances with GLOBAL coverage, from the OSM wiki's instance
# table. NOTE: overpass.kumi.systems is just the OLD NAME of the
# private.coffee instance — listing both meant two of our three "mirrors"
# were the same servers, which is why both timed out together. VK Maps
# (mail.ru) is a genuinely independent operator with no request limits, so
# this list is now three separate organisations, not two.
_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# Guardrail: never trust model-supplied arguments blindly. Clamp the radius so
# a hallucinated radius_m=9999999 can't hammer Overpass or blow the timeout.
_MIN_RADIUS_M = 500
_MAX_RADIUS_M = 30000

# ---------------------------------------------------------------------------
# LOCAL FETCH CACHE — because every public Overpass instance is a shared,
# best-effort service that the wiki itself says "can often become overloaded",
# and hospitals don't move week to week.
#
# The Overpass query below never mentions the specialty — filtering happens
# client-side after the fetch — so ONE cached fetch per (~1 km location cell,
# radius) serves EVERY specialty and every nearby user. distance_km is always
# recomputed from the caller's exact coordinates, so ranking stays per-user
# even on a cache hit. On a total mirror outage we serve an EXPIRED entry
# rather than nothing ("stale-if-error"): week-old facility data beats telling
# a patient the directory is unreachable. Correctness never depends on the
# cache — only speed and availability do.
# ---------------------------------------------------------------------------
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "osm_cache.json")
_CACHE_TTL_S = 7 * 24 * 3600


def _cache_load() -> dict:
    try:
        with open(_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _cache_save() -> None:
    # Last-writer-wins under concurrency — fine for a single-user demo, the
    # same caveat as the LIVE patient in server.py.
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_CACHE, f)
    except OSError as e:
        print(f"[osm] cache write failed (non-fatal): {e}")


def _cache_key(lat: float, lng: float, radius_m: int) -> str:
    # 2 decimal places ~= 1.1 km grid cells: nearby users share a fetch. A
    # user just across a cell boundary merely triggers one extra fetch —
    # never a wrong answer, because distances are recomputed per caller.
    return f"{lat:.2f},{lng:.2f},r{int(radius_m)}"


_CACHE = _cache_load()


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

    Both find_providers and find_general_facilities use this, so the query,
    the cache, and the distance math live in exactly one place.
    """
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"~"hospital|clinic|doctors"](around:{radius_m},{patient_lat},{patient_lng});
      way["amenity"~"hospital|clinic|doctors"](around:{radius_m},{patient_lat},{patient_lng});
    );
    out center tags;
    """
    key = _cache_key(patient_lat, patient_lng, radius_m)
    hit = _CACHE.get(key)
    now = time.time()
    elements = None

    if hit and now - hit["fetched_at"] < _CACHE_TTL_S:
        elements = hit["elements"]                 # fresh cache: zero network
    else:
        headers = {"User-Agent": "CareRoute-demo/1.0 (learning project)"}
        last_error = None
        for url in _OVERPASS_ENDPOINTS:
            try:
                resp = requests.post(url, data={"data": query},
                                     headers=headers, timeout=30)
                resp.raise_for_status()
                # .json() stays inside the try: a mirror answering 200 with an
                # HTML error page falls through to the next mirror instead of
                # crashing the tool.
                elements = resp.json().get("elements", [])
                _CACHE[key] = {"fetched_at": now, "elements": elements}
                _cache_save()
                break
            except requests.RequestException as e:
                last_error = e
                # str(e) shows the actual reason — "429 Client Error" (rate
                # limited, back off 30 s per the usage policy) vs "504 Server
                # Error" (overloaded) vs "Read timed out" — so an outage is
                # diagnosable from the log alone.
                print(f"[osm] {url} failed: {str(e)[:90]}")

        if elements is None and hit:
            # stale-if-error: every mirror is down, but we have seen this
            # area before. Serve the expired data and say so.
            age_h = (now - hit["fetched_at"]) / 3600
            print(
                f"[osm] all mirrors down — serving cached data ({age_h:.0f} h old)")
            elements = hit["elements"]

        if elements is None:
            # Every mirror is down AND this area was never fetched before.
            # This is a TRANSPORT failure, and it is not the same fact as "no
            # specialists nearby" — the model conflated the two and told a
            # cardiac patient no cardiologist was found while one sat 3.3 km
            # away. The payload says so in the data, because a prompt
            # instruction is advisory and a data structure is enforced.
            return None, {
                "error": f"Provider directory unreachable: {last_error}",
                "error_type": "upstream_unavailable",
                "hint": ("The directory could not be reached, so nothing is known "
                         "about which providers exist. Do NOT retry with a different "
                         "radius_m — the radius is not the problem. Do NOT say that "
                         "no specialists were found. Tell the user the provider "
                         "directory is temporarily unreachable and to try again "
                         "shortly; for urgent symptoms, direct them to emergency care."),
            }

    stem = _stem(specialty) if specialty else None
    out = []
    for el in elements:
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
    # Running this once while a mirror is up also PRE-WARMS the cache, which
    # makes the web demo outage-proof for this area for a week.
    print("HIT CASE:")
    print(find_providers("Cardiology", 12.9352, 77.6245, k=5))
    print("\nMISS CASE (count only — no names to misuse):")
    print(find_providers("Rheumatology", 12.9352, 77.6245, k=3, radius_m=1000))
    print("\nEXPLICIT FALLBACK (labelled non-specialist):")
    print(find_general_facilities(12.9352, 77.6245, k=3))
