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

  2. Returning the nearest general facilities INSIDE the miss payload "for
     context" once caused the model to list them as specialists. The current
     design re-surfaces them (users want nearby options immediately) but makes
     the labelling ENFORCED, not advisory: each carries is_specialist_match=
     false under 'general_alternatives', and the answer guard harvests those
     names into 'offered' (unverified), so a bare list can never be presented
     as confirmed specialists.

Lesson: a prompt instruction is advisory; a data structure — and a code guard —
is enforced.
"""

import json
import os
import time
from math import radians, sin, cos, sqrt, atan2

import requests

from routing import annotate_road_distance

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

# Fix C: on a transient outage, retry the SAME radius a couple of times with a
# short backoff rather than escalating — a bigger query only makes an overloaded
# server likelier to fail. Correctness never depends on this; only availability.
_FETCH_ROUNDS = 2
_RETRY_BACKOFF_S = 2
_HTTP_TIMEOUT_S = 20

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
        # Fix C: one pass over the mirrors, and if all fail transiently, back off
        # briefly and try the SAME query again before declaring an outage — never
        # a larger radius.
        for round_i in range(_FETCH_ROUNDS):
            if round_i:
                time.sleep(_RETRY_BACKOFF_S)
                print(
                    f"[osm] retry round {round_i + 1} (same radius {radius_m} m)")
            for url in _OVERPASS_ENDPOINTS:
                try:
                    resp = requests.post(url, data={"data": query},
                                         headers=headers, timeout=_HTTP_TIMEOUT_S)
                    resp.raise_for_status()
                    # .json() stays inside the try: a mirror answering 200 with
                    # an HTML error page falls through to the next mirror instead
                    # of crashing the tool.
                    elements = resp.json().get("elements", [])
                    _CACHE[key] = {"fetched_at": now, "elements": elements}
                    _cache_save()
                    break
                except requests.RequestException as e:
                    last_error = e
                    # str(e) shows the actual reason — "429 Client Error" (rate
                    # limited) vs "504 Server Error" (overloaded) vs "Read timed
                    # out" — so an outage is diagnosable from the log alone.
                    print(f"[osm] {url} failed: {str(e)[:90]}")
            if elements is not None:
                break

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
                         "about which providers exist. Do NOT change radius_m — the "
                         "radius is not the problem. You may retry the SAME call "
                         "once. Do NOT say that no specialists were found. Tell the "
                         "user the provider directory is temporarily unreachable and "
                         "to try again shortly; for urgent symptoms, direct them to "
                         "emergency care."),
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
    a dict with match_found=False plus general_alternatives: the nearest general
    facilities, each labelled is_specialist_match=false, so the user gets nearby
    options immediately without them being passed off as specialists.
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
        # Take a small shortlist on the cheap straight-line sort, then upgrade
        # JUST those to real road distance + ETA (one OSRM call). Re-rank with
        # the same key so evidence quality still leads and road distance is the
        # tie-break — and the distance we DISPLAY is now what a user will drive.
        shortlist = matches[:max(k + 4, 8)]
        annotate_road_distance(patient_lat, patient_lng, shortlist)
        shortlist.sort(key=lambda f: (f.get("matched_via") != "name",
                                      f.get("_tag_tokens", 0),
                                      f["distance_km"]))
        top = shortlist[:k]
        for f in top:
            f.pop("_tag_tokens", None)
        return top

    # No specialty match at this radius. Instead of a bare count, hand back the
    # nearest GENERAL facilities right now (labelled non-specialist, with road
    # distance) so a minor complaint isn't forced to chase a far specialist —
    # the agent can still widen the radius for the real specialist and let the
    # user choose. Names live under 'general_alternatives' and carry
    # is_specialist_match=false, so the answer guard counts them as OFFERED
    # (unverified), never confirmed specialists.
    alternatives = _dedupe(facilities)[:3]
    annotate_road_distance(patient_lat, patient_lng, alternatives)
    alternatives = [{"name": f["name"], "facility": f["facility"],
                     "distance_km": f["distance_km"],
                     "duration_min": f.get("duration_min"),
                     "distance_type": f.get("distance_type"),
                     "is_specialist_match": False}
                    for f in alternatives]
    return {
        "match_found": False,
        "specialty_requested": specialty,
        "reason": f"No facility matching '{specialty}' within {radius_m / 1000:.1f} km.",
        "radius_searched_m": radius_m,
        "general_facilities_nearby": len(facilities),
        "general_alternatives": alternatives,
        "general_alternatives_disclaimer": (
            f"These are the nearest GENERAL facilities, NOT {specialty} "
            "specialists. Offer them as convenient nearby options — the user may "
            "prefer one over travelling far for a specialist."),
        "hint": ("Do BOTH: (1) show general_alternatives to the user as nearby "
                 "general options with their drive distance/time; (2) you MAY "
                 "call find_providers again with a larger radius_m (double it, up "
                 "to 30000, at most twice) to look for an actual specialist "
                 "further out, then let the user choose. Never describe "
                 "general_alternatives as specialists."),
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
    # Shortlist on straight-line, upgrade to road distance, then re-rank.
    shortlist = facilities[:max(k + 4, 8)]
    annotate_road_distance(patient_lat, patient_lng, shortlist)
    shortlist.sort(key=lambda f: f["distance_km"])
    top = shortlist[:k]
    for f in top:
        f["is_specialist_match"] = False
    return {
        "disclaimer": ("These are general healthcare facilities, NOT verified "
                       "specialists. Present them only as general options."),
        "facilities": top,
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
