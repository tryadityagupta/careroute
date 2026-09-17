"""
emergency.py — emergency triage helper.

get_emergency_help(lat, lng) returns the LOCAL emergency number to call now,
plus (best-effort) the nearest hospitals, which have emergency departments.

DESIGN PRINCIPLE: the NUMBER is the critical output and must never depend on a
network call succeeding. We resolve the caller's country from a cached
reverse-geocode when we can, fall back to a coarse offline region check, and
finally to 112 — a GSM-standard number that reaches emergency services across
the EU, India, and much of the world. The nearest-hospital list is a nice-to-
have layered on top; if Overpass is down, we still hand back the number.

CareRoute cannot dispatch help. It can only tell the user who to call, fast.
Emergency numbers are best-effort by location; a real product would verify them
against an authoritative per-country source.
"""

import json
import os
import time

import requests

import osm  # reuse its Overpass fetch, dedupe, and road-distance annotation

# ISO 3166-1 alpha-2 country code -> what to dial. 112 is the safe default and
# works across the EU + India + most GSM networks. Non-112 countries override.
_NUMBERS = {
    "IN": "112 (ambulance: 108)",
    "US": "911", "CA": "911",
    "GB": "999 (or 112)",
    "AU": "000 (or 112)",
    "NZ": "111",
    "JP": "119 (ambulance)",
    "KR": "119",
    "AE": "998 (ambulance)",
    # EU / EEA — all reachable on 112:
    "IE": "112", "FR": "112", "DE": "112", "ES": "112", "IT": "112",
    "NL": "112", "BE": "112", "PT": "112", "SE": "112", "DK": "112",
    "FI": "112", "PL": "112", "AT": "112", "GR": "112", "CZ": "112",
    "RO": "112", "HU": "112", "NO": "112", "CH": "112",
}
_DEFAULT_NUMBER = "112"

_NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
_GEO_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "geo_cache.json")
_GEO_TTL_S = 30 * 24 * 3600


def _geo_cache_load():
    try:
        with open(_GEO_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


_GEO_CACHE = _geo_cache_load()


def _offline_country(lat, lng):
    """Crude bounding-box fallback so the locations we most care about are still
    right when Nominatim is unreachable. Boxes, not real borders — on purpose."""
    if 6 <= lat <= 37 and 68 <= lng <= 98:
        return "IN"
    if 24 <= lat <= 72 and -170 <= lng <= -52:
        return "US"                     # North America; US/CA share 911
    return None


def _country_code(lat, lng):
    key = f"{lat:.1f},{lng:.1f}"        # ~11 km cell — plenty for a country
    hit = _GEO_CACHE.get(key)
    if hit and time.time() - hit["at"] < _GEO_TTL_S:
        return hit["cc"]

    cc = None
    try:
        resp = requests.get(
            _NOMINATIM,
            params={"lat": lat, "lon": lng, "format": "json", "zoom": 3},
            headers={"User-Agent": "CareRoute-demo/1.0 (learning project)"},
            timeout=6,
        )
        resp.raise_for_status()
        cc = (resp.json().get("address", {}).get(
            "country_code") or "").upper() or None
    except (requests.RequestException, ValueError) as e:
        print(
            f"[emergency] reverse-geocode failed ({str(e)[:60]}); offline fallback")

    if not cc:
        cc = _offline_country(lat, lng)

    if cc:
        _GEO_CACHE[key] = {"at": time.time(), "cc": cc}
        try:
            os.makedirs(os.path.dirname(_GEO_CACHE_PATH), exist_ok=True)
            with open(_GEO_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(_GEO_CACHE, f)
        except OSError:
            pass
    return cc


def local_emergency_number(lat, lng):
    """Return (number_string, country_code_or_None) for a location."""
    cc = _country_code(lat, lng)
    return _NUMBERS.get(cc, _DEFAULT_NUMBER), cc


def get_emergency_help(patient_lat: float, patient_lng: float) -> dict:
    """Return the LOCAL emergency number to call NOW, plus the nearest hospitals
    (which have emergency departments). Call this FIRST for any medical
    emergency — seizure, stroke signs, major trauma, severe bleeding, chest pain
    with cardiac features, fainting/unconsciousness, or trouble breathing —
    BEFORE any specialist search. The number is always returned; the hospital
    list is best-effort and may be empty if the directory is unreachable.
    """
    number, cc = local_emergency_number(patient_lat, patient_lng)

    hospitals = []
    try:
        facilities, err = osm._fetch_nearby(patient_lat, patient_lng, 8000)
        if not err and facilities:
            hosp = [f for f in osm._dedupe(facilities)
                    if f.get("facility") == "hospital"][:3]
            osm.annotate_road_distance(patient_lat, patient_lng, hosp)
            hospitals = [{"name": h["name"], "distance_km": h["distance_km"],
                          "duration_min": h.get("duration_min"),
                          "distance_type": h.get("distance_type")}
                         for h in hosp]
    except Exception as e:                          # never let hospitals break the number
        print(f"[emergency] hospital lookup failed (non-fatal): {str(e)[:60]}")

    return {
        "emergency": True,
        "emergency_number": number,
        "country": cc,
        "advice": (f"This may be a medical emergency. Tell the user to call {number} "
                   "immediately, or go to the nearest emergency department. State "
                   "this FIRST, before any specialist recommendation."),
        "nearest_hospitals": hospitals,             # hospitals all have an ER
        "disclaimer": ("CareRoute cannot contact emergency services — the user must "
                       "call. Numbers are best-effort by location; 112 reaches "
                       "emergency services in many countries if unsure."),
    }


if __name__ == "__main__":
    import pprint
    pprint.pp(get_emergency_help(12.9352, 77.6245))   # Bengaluru
