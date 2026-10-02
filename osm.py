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
import re
import time
from math import radians, sin, cos, sqrt, atan2

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

from routing import annotate_road_distance

load_dotenv()  # self-sufficient: OSM_SOURCE is read at import time

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

# SPEED FIX: the mirrors used to be tried ONE AFTER ANOTHER, 20 s each, for two
# rounds — up to ~122 s before giving up. Now all mirrors are queried IN
# PARALLEL and the first good answer wins.
# 12 s proved too tight: a dense-city 8 km query routinely takes ~12 s, so a
# busy moment made EVERY mirror time out. 25 s matches the query's own
# [timeout:25]; since the mirrors race in parallel, the worst case stays ~25 s.
_HTTP_TIMEOUT_S = 25
# If every mirror fails FAST (e.g. instant 429/504), one quick second race is
# worth it. Skipped if the first race already used up most of the budget.
_RETRY_IF_FAILED_WITHIN_S = 8
_RETRY_BACKOFF_S = 1.5

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


_CACHE_LOCK = threading.Lock()


def _cache_save() -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with _CACHE_LOCK, open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_CACHE, f)
    except (OSError, RuntimeError) as e:   # RuntimeError: dict changed mid-dump
        print(f"[osm] cache write failed (non-fatal): {e}")


# Bump this whenever the Overpass QUERY below changes shape. It is part of the
# cache key, so a query change auto-invalidates every stale entry (fetched with
# the old, narrower query) instead of serving week-old results.
_QUERY_VERSION = 3


def _cache_key(lat: float, lng: float, radius_m: int) -> str:
    # 2 decimal places ~= 1.1 km grid cells: nearby users share a fetch. A
    # user just across a cell boundary merely triggers one extra fetch —
    # never a wrong answer, because distances are recomputed per caller.
    return f"v{_QUERY_VERSION}|{lat:.2f},{lng:.2f},r{int(radius_m)}"


_CACHE = _cache_load()


def _find_cached(lat: float, lng: float, radius_m: int, now: float, fresh=True):
    """Return a cache entry for this ~1 km cell whose radius is AT LEAST the one
    requested. A 30 km fetch already contains everything within 8 km, so a
    smaller-radius search never needs a new Overpass call. Results are filtered
    back down to radius_m in _fetch_nearby, so the answer is unchanged.
    fresh=False also accepts expired entries (for stale-if-error)."""
    prefix = f"v{_QUERY_VERSION}|{lat:.2f},{lng:.2f},r"
    best = None
    for key, hit in list(_CACHE.items()):
        if not key.startswith(prefix):
            continue
        try:
            r = int(key[len(prefix):])
        except ValueError:
            continue
        if r < radius_m:
            continue
        if fresh and now - hit["fetched_at"] >= _CACHE_TTL_S:
            continue
        if best is None or r < best[0]:
            # smallest sufficient radius = least filtering
            best = (r, hit)
    return best[1] if best else None


def _post_overpass(url: str, query: str, headers: dict):
    resp = requests.post(url, data={"data": query},
                         headers=headers, timeout=_HTTP_TIMEOUT_S)
    resp.raise_for_status()
    # .json() inside the worker: a mirror answering 200 with an HTML error page
    # counts as a failure and the race continues with the other mirrors.
    return resp.json().get("elements", [])


def _race_mirrors(query: str, headers: dict):
    """Query every mirror at once; return (elements, None) from the FIRST one
    that succeeds, or (None, last_error) if all fail."""
    ex = ThreadPoolExecutor(max_workers=len(_OVERPASS_ENDPOINTS))
    futures = {ex.submit(_post_overpass, u, query, headers): u
               for u in _OVERPASS_ENDPOINTS}
    last_error = None
    try:
        for fut in as_completed(futures):
            try:
                return fut.result(), None
            except (requests.RequestException, ValueError) as e:
                last_error = e
                print(f"[osm] {futures[fut]} failed: {str(e)[:90]}")
        return None, last_error
    finally:
        # Don't wait for the slower mirrors once we have an answer.
        ex.shutdown(wait=False, cancel_futures=True)


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


# ---------------------------------------------------------------------------
# SPECIALTY MATCHING — why "Orthopedics" found nothing in Bengaluru.
#
# The crude stem turned "Orthopedics" into "orthopedic", which never matches
# OSM's healthcare:speciality value "orthopaedics" (OSM uses British spelling),
# nor Indian clinic names like "Sparsh Ortho Care" or "Bone & Joint Hospital".
# So each common specialty gets explicit patterns: spelling variants, the
# short forms clinics actually use, and — where a short form is ambiguous —
# an EXCLUDE pattern ("ortho" must not match an orthodontic dental clinic).
# Unknown specialties still fall back to the old stem.
# ---------------------------------------------------------------------------
_SPECIALTY_PATTERNS = [
    # (keys: regexes matched at a WORD START of the requested specialty,
    #  include regex, exclude regex). Word-start matters: a bare "ent" key
    #  would otherwise fire for "Gastroenterology" and "Dentistry".
    (("ortho",), r"orthop(a)?ed|\bortho\b|bone\s*(and|&)\s*joint|joint\s*replacement|fracture",
     r"orthodont|dental|dentist|\bteeth\b"),
    (("pediatr", "paediatr", "child"),
     r"pa?ediatr|children'?s\s*(hospital|clinic)|\bchild\s*care\b", None),
    (("gyn", "obstet", "women"),
     r"gyn(a)?ecolog|obstetric|maternity|women'?s\s*(hospital|clinic)|\bivf\b", None),
    (("ent\\b", "otolaryng", "ear\\b"),
     r"\bent\b|otolaryng|otorhinolaryng|ear,?\s*nose", None),
    (("cardio", "heart"), r"cardi|\bheart\b", None),
    (("derma", "skin"), r"dermatolog|\bskin\b", None),
    (("neuro",), r"neurolog|neuro\s*(care|clinic|hospital)|\bneuro\b", r"neurosurg(?!ery)"),
    (("psychiat", "mental"), r"psychiatr|mental\s*health|de-?addiction", None),
    (("ophthalm", "eye"), r"ophthalm|\beye\b", None),
    (("gastro",), r"gastro", None),
    (("uro",), r"urolog|\buro\b", None),
    (("oncol", "cancer"), r"oncolog|cancer", None),
    (("pulmon", "chest", "respir"), r"pulmon|\bchest\b|respirat", None),
    (("nephro", "kidney"), r"nephrolog|kidney|dialysis", None),
    (("endocrin", "diabet"), r"endocrin|diabet", None),
    (("dent",), r"dent|orthodont", None),
    (("general", "family"),
     r"general\s*(medicine|practice|physician)|family\s*(medicine|doctor)", None),
]


def _specialty_matcher(specialty: str):
    """Return a function(text) -> bool for this specialty."""
    spec = specialty.lower()
    for keys, inc, exc in _SPECIALTY_PATTERNS:
        if any(re.search(r"\b" + k, spec) for k in keys):
            inc_re = re.compile(inc, re.I)
            exc_re = re.compile(exc, re.I) if exc else None
            return lambda text: bool(inc_re.search(text)) and not (
                exc_re and exc_re.search(text))
    stem = _stem(specialty)
    return lambda text: stem in text.lower()


def _unreachable(last_error):
    """Payload for 'the directory could not be queried'.

    This is a TRANSPORT failure, and it is not the same fact as "no
    specialists nearby" — the model conflated the two and told a cardiac
    patient no cardiologist was found while one sat 3.3 km away. The payload
    says so in the data, because a prompt instruction is advisory and a data
    structure is enforced. Same shape for both sources (Overpass, PostGIS).
    """
    return {
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


def _elements_from_postgis(patient_lat, patient_lng, radius_m):
    """Self-hosted source: one indexed spatial query, no cache needed."""
    import geo_db
    t0 = time.perf_counter()
    try:
        elements = geo_db.nearby_elements(patient_lat, patient_lng, radius_m)
    except geo_db.GeoUnavailable as e:
        print(f"[osm] postgis lookup failed: {e}")
        return None, _unreachable(e)
    print(f"[osm] postgis r={radius_m} rows={len(elements)} "
          f"took {(time.perf_counter() - t0) * 1000:.0f}ms")
    return elements, None


def _elements_from_overpass(patient_lat, patient_lng, radius_m):
    """Public Overpass mirrors, with the on-disk cache and stale-if-error."""
    # Two tagging schemes, because OSM uses both and small Indian clinics and
    # chemists are split across them:
    #   * amenity=hospital|clinic|doctors|pharmacy — the classic scheme.
    #   * healthcare=*  — the newer scheme (healthcare=clinic|doctor|hospital|
    #     pharmacy|dentist|...). A place tagged ONLY healthcare=clinic is invisible
    #     to an amenity-only query even though it sits in OSM.
    # pharmacy is included so "I need medicines now" can surface a chemist.
    # Overpass dedupes the union by element id, so a POI carrying both keys is
    # returned once. geo/healthcare.lua mirrors this filter for PostGIS.
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"~"hospital|clinic|doctors|pharmacy"](around:{radius_m},{patient_lat},{patient_lng});
      way["amenity"~"hospital|clinic|doctors|pharmacy"](around:{radius_m},{patient_lat},{patient_lng});
      node["healthcare"](around:{radius_m},{patient_lat},{patient_lng});
      way["healthcare"](around:{radius_m},{patient_lat},{patient_lng});
      node["shop"~"chemist|pharmacy"](around:{radius_m},{patient_lat},{patient_lng});
      way["shop"~"chemist|pharmacy"](around:{radius_m},{patient_lat},{patient_lng});
    );
    out center tags;
    """
    key = _cache_key(patient_lat, patient_lng, radius_m)
    now = time.time()
    hit = _find_cached(patient_lat, patient_lng, radius_m, now)
    if hit:
        return hit["elements"], None               # fresh cache: zero network

    headers = {"User-Agent": "CareRoute-demo/1.0 (learning project)"}
    t0 = time.perf_counter()
    elements, last_error = _race_mirrors(query, headers)
    if elements is None and time.perf_counter() - t0 < _RETRY_IF_FAILED_WITHIN_S:
        print("[osm] all mirrors failed fast — one more race")
        time.sleep(_RETRY_BACKOFF_S)
        elements, last_error = _race_mirrors(query, headers)
    print(f"[osm] overpass fetch r={radius_m} took "
          f"{time.perf_counter() - t0:.1f}s ok={elements is not None}")
    if elements is not None:
        with _CACHE_LOCK:
            _CACHE[key] = {"fetched_at": now, "elements": elements}
        _cache_save()
        return elements, None

    hit = _find_cached(patient_lat, patient_lng, radius_m, now, fresh=False)
    if hit:
        # stale-if-error: every mirror is down, but we have seen this
        # area before. Serve the expired data and say so.
        age_h = (now - hit["fetched_at"]) / 3600
        print(
            f"[osm] all mirrors down — serving cached data ({age_h:.0f} h old)")
        return hit["elements"], None
    # Every mirror is down AND this area was never fetched before.
    return None, _unreachable(last_error)


# Where raw OSM elements come from. Both return the same element shape, so
# everything below _fetch_nearby is source-agnostic.
#   overpass (default) — public mirrors; fine for a demo, NOT for load tests.
#   postgis            — self-hosted import (geo/import.sh); the scaled design.
OSM_SOURCE = os.getenv("OSM_SOURCE", "overpass").strip().lower()
_SOURCES = {"overpass": _elements_from_overpass,
            "postgis": _elements_from_postgis}
if OSM_SOURCE not in _SOURCES:
    raise ValueError(f"OSM_SOURCE must be one of {sorted(_SOURCES)}, "
                     f"got {OSM_SOURCE!r}")


def _fetch_nearby(patient_lat, patient_lng, radius_m, specialty=None):
    """Shared facility fetch. Returns (facilities, error_or_None).

    find_providers, find_general_facilities, find_pharmacies and the emergency
    path all use this, so the source, the distance math and the specialty
    matching live in exactly one place.
    """
    elements, err = _SOURCES[OSM_SOURCE](patient_lat, patient_lng, radius_m)
    if err:
        return None, err
    matches_specialty = _specialty_matcher(specialty) if specialty else None
    out = []
    for el in elements:
        tags = el.get("tags", {})
        # node has lat/lon directly; way has it under 'center'.
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lng = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lng is None:
            continue
        # A cache hit may come from a LARGER-radius fetch, so enforce the
        # radius the caller actually asked for.
        if _haversine_km(patient_lat, patient_lng, lat, lng) * 1000 > radius_m:
            continue
        name = tags.get("name")
        # Skip POIs with no name. The broadened query surfaces bare
        # healthcare=* nodes (unnamed labs/clinics); "go to Unnamed
        # Facility" is not actionable, and one even ranked #1 by
        # distance. A place you cannot name cannot be recommended.
        if not name:
            continue
        amenity = (tags.get("amenity") or "").lower()
        healthcare = (tags.get("healthcare") or "").lower()
        shop = (tags.get("shop") or "").lower()
        # A lowered blob of the signals that reveal what a place IS. Used to
        # spot pharmacies and to filter narrow cosmetic/single-specialty clinics
        # out of GENERAL lists. "_"-prefixed so it is stripped before the
        # payload reaches the model.
        signal = " ".join((name.lower(), amenity, healthcare, shop,
                           tags.get("healthcare:speciality", "").lower()))
        is_pharmacy = ("pharmacy" in amenity or "pharmacy" in healthcare
                       or shop in ("chemist", "pharmacy"))
        item = {
            "name": name,
            # Prefer amenity, then healthcare, then shop, so a place tagged only
            # healthcare=clinic or shop=chemist still shows a useful type.
            "facility": (tags.get("amenity") or tags.get("healthcare")
                         or tags.get("shop") or "healthcare"),
            "lat": lat,
            "lng": lng,
            "distance_km": _haversine_km(patient_lat, patient_lng, lat, lng),
            "is_pharmacy": is_pharmacy,
            "_signal": signal,
            # OSM's emergency=yes|no tag (hospitals with / without an ER).
            # "_"-prefixed: used by emergency.py, stripped before the model.
            "_er": (tags.get("emergency") or "").lower(),
        }
        if matches_specialty:
            # Bengaluru OSM facilities often carry healthcare:speciality as a
            # long semicolon list, sometimes batch-pasted across many POIs.
            # Substring-matching the whole blob confirmed a DENTAL clinic for
            # Psychiatry. So: match token-by-token, and RECORD THE EVIDENCE —
            # matched_via + the raw tag — so no match is ever opaque again.
            raw_tag = tags.get("healthcare:speciality", "")
            tokens = [t.strip().lower()
                      for t in raw_tag.split(";") if t.strip()]
            if matches_specialty(name):
                item["specialty_match"] = True
                item["matched_via"] = "name"
            elif any(matches_specialty(t) for t in tokens):
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


# Narrow, mostly-cosmetic single-specialty clinics that cannot serve a general/
# undifferentiated complaint. find_general_facilities runs only AFTER the
# specialist search already failed, so a skin/hair/laser/dental/eye boutique
# next door is noise, not a useful "general" option (a hair clinic kept getting
# suggested for diarrhoea). Drop these from general lists — never from
# specialist results, and never a pharmacy (a chemist is useful for OTC needs).
_NARROW_SPECIALTY = (
    "skin", "hair", "laser", "cosmetic", "aesthetic", "derma",
    "dental", "dentist", "orthodont", "ophthal", "optical", "optician",
    "fertility", "ivf", "veterinary",
)


def _is_narrow_specialty(item) -> bool:
    if item.get("is_pharmacy") or "hospital" in (item.get("facility") or ""):
        return False  # pharmacies and full hospitals are always broad enough
    return any(kw in item.get("_signal", "") for kw in _NARROW_SPECIALTY)


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
            f.pop("_signal", None)
            f.pop("_er", None)
        return top

    # No specialty match at this radius. Instead of a bare count, hand back the
    # nearest GENERAL facilities right now (labelled non-specialist, with road
    # distance) so a minor complaint isn't forced to chase a far specialist —
    # the agent can still widen the radius for the real specialist and let the
    # user choose. Names live under 'general_alternatives' and carry
    # is_specialist_match=false, so the answer guard counts them as OFFERED
    # (unverified), never confirmed specialists.
    # Drop narrow cosmetic/single-specialty clinics from the nearby general
    # options (a hair clinic is not a useful alternative for an unrelated
    # complaint); keep them only if that would otherwise leave nothing.
    _general = [f for f in _dedupe(facilities) if not _is_narrow_specialty(f)]
    alternatives = (_general or _dedupe(facilities))[:3]
    annotate_road_distance(patient_lat, patient_lng, alternatives)
    alternatives = [{"name": f["name"], "facility": f["facility"],
                     "distance_km": f["distance_km"],
                     "drive_min_no_traffic": f.get("drive_min_no_traffic"),
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
    # Drop narrow cosmetic/single-specialty clinics — a skin/hair/laser boutique
    # can't help with an undifferentiated complaint, and proximity alone kept
    # surfacing them (a hair clinic for diarrhoea). Keep them only if filtering
    # would leave nothing, so we never return empty when facilities exist.
    _general = [f for f in facilities if not _is_narrow_specialty(f)]
    facilities = _general or facilities
    # Shortlist on straight-line, upgrade to road distance, then re-rank.
    shortlist = facilities[:max(k + 4, 8)]
    annotate_road_distance(patient_lat, patient_lng, shortlist)
    shortlist.sort(key=lambda f: f["distance_km"])
    top = shortlist[:k]
    for f in top:
        f["is_specialist_match"] = False
        f.pop("_signal", None)
        f.pop("_er", None)
    return {
        "disclaimer": ("These are general healthcare facilities, NOT verified "
                       "specialists. Present them only as general options."),
        "facilities": top,
    }


def find_pharmacies(patient_lat: float, patient_lng: float,
                    k: int = 3, radius_m: int = 8000) -> list | dict:
    """Nearest PHARMACIES / chemists — where a user goes to OBTAIN medicines.

    Use this for requests to buy or pick up a medicine or OTC drug (painkillers,
    antacids, ORS, cold medicine, etc.). Reuses the same Overpass fetch, then
    keeps only pharmacy/chemist results. Returns match_found=false — NOT a
    hospital — when no pharmacy is mapped nearby, so the user gets an honest
    "none in our data" instead of a hospital in disguise.
    """
    radius_m = _clamp_radius(radius_m)
    facilities, err = _fetch_nearby(patient_lat, patient_lng, radius_m)
    if err:
        return err
    pharmacies = _dedupe([f for f in facilities if f.get("is_pharmacy")])
    if not pharmacies:
        return {
            "match_found": False,
            "reason": (f"No pharmacy is mapped in OpenStreetMap within "
                       f"{radius_m / 1000:.1f} km of the patient."),
            "hint": ("Do NOT offer a hospital or clinic as a pharmacy. Tell the "
                     "user no pharmacy was found in the map data nearby; suggest "
                     "they widen the search or check locally."),
        }
    shortlist = pharmacies[:max(k + 4, 8)]
    annotate_road_distance(patient_lat, patient_lng, shortlist)
    shortlist.sort(key=lambda f: f["distance_km"])
    top = shortlist[:k]
    for f in top:
        f.pop("_signal", None)
        f.pop("_er", None)
    return {
        "disclaimer": "Nearby pharmacies/chemists for obtaining medicines.",
        "pharmacies": top,
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
    print("\nPHARMACIES:")
    print(find_pharmacies(12.9352, 77.6245, k=3))
