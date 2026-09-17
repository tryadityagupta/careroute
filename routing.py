"""
routing.py — real road distance + ETA via OSRM (free, OpenStreetMap-based).

WHY THIS EXISTS
Straight-line (haversine) distance is fine for RANKING nearby options, but the
number looks wrong next to Google/Apple Maps: it ignores roads, rivers, and
one-ways. A user in Jaipur saw "3.78 km" for a clinic Maps put 170 m away.
OSRM computes distance and time along the actual road network, so the number
matches what a user will really drive.

DESIGN (the interview-worthy part)
  * haversine stays UPSTREAM as a cheap, no-network pre-filter to shortlist the
    nearest few candidates. We never ask OSRM about far-away places.
  * OSRM is called ONCE per query via its Table service (one source -> many
    destinations), so a shortlist of 8 costs a single HTTP request, not 8.
  * GRACEFUL FALLBACK: if OSRM is unreachable or rate-limited, we keep the
    haversine distance and label it, rather than failing the whole lookup.
    Correctness of "is there a provider" never depends on OSRM — only the
    accuracy of the distance number does.

The public demo server (router.project-osrm.org) is fine for a low-volume
portfolio project but has NO SLA and is rate-limited. Point OSRM_BASE_URL at a
self-hosted OSRM (or a paid routing provider) for real load.
"""

import os
from math import radians, sin, cos, sqrt, atan2

import requests

_OSRM_BASE = os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org")
_OSRM_TIMEOUT_S = 8


def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km. Kept as the fast, offline fallback."""
    R = 6371
    d_lat = radians(lat2 - lat1)
    d_lng = radians(lng2 - lng1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(d_lng / 2) ** 2
    return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 2)


def road_distances(origin_lat, origin_lng, dests):
    """Road distance + duration from one origin to many destinations.

    dests: list of (lat, lng) tuples.
    Returns a list aligned with dests of {"distance_km", "duration_min"},
    or None if OSRM is unavailable (so the caller can fall back to haversine).
    """
    if not dests:
        return []

    # OSRM speaks {lng},{lat}. The first coordinate is the source; the rest are
    # destinations. sources=0 tells OSRM to compute only FROM the origin, which
    # keeps the response a single row instead of a full NxN matrix.
    coords = ";".join(
        [f"{origin_lng},{origin_lat}"] + [f"{lng},{lat}" for lat, lng in dests]
    )
    url = f"{_OSRM_BASE}/table/v1/driving/{coords}"
    params = {"sources": "0", "annotations": "distance,duration"}

    try:
        resp = requests.get(url, params=params, timeout=_OSRM_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok":
            print(f"[routing] OSRM returned code={data.get('code')}; "
                  "falling back to haversine")
            return None

        # One source row. Column 0 is origin->origin (0), destinations start at 1.
        dist_row = (data.get("distances") or [[]])[0]
        dur_row = (data.get("durations") or [[]])[0]

        out = []
        for i in range(1, len(dests) + 1):
            d_m = dist_row[i] if i < len(dist_row) else None
            t_s = dur_row[i] if i < len(dur_row) else None
            if d_m is None:
                # e.g. server built without the 'distance' annotation, or an
                # unroutable point. Fall back wholesale so results stay
                # consistent (all road, or all straight-line — never mixed).
                print("[routing] OSRM gave no distance for a point; "
                      "falling back to haversine")
                return None
            out.append({
                "distance_km": round(d_m / 1000, 2),
                "duration_min": round(t_s / 60) if t_s is not None else None,
            })
        return out

    except (requests.RequestException, ValueError, KeyError, IndexError) as e:
        print(f"[routing] OSRM unavailable ({str(e)[:80]}); using haversine")
        return None


def annotate_road_distance(origin_lat, origin_lng, items):
    """Upgrade a shortlist of provider dicts to real road distance, in place.

    Each item must have 'lat', 'lng', and a straight-line 'distance_km'. On
    success this overwrites distance_km with road distance and adds
    'duration_min' and distance_type='road'. On OSRM failure it leaves the
    haversine distance untouched and marks distance_type='straight_line', so
    the number is never silently misrepresented. Returns items for chaining.
    """
    if not items:
        return items

    road = road_distances(origin_lat, origin_lng,
                          [(it["lat"], it["lng"]) for it in items])

    if road is None:
        for it in items:
            it.setdefault("distance_type", "straight_line")
        return items

    for it, r in zip(items, road):
        it["distance_km"] = r["distance_km"]
        it["duration_min"] = r["duration_min"]
        it["distance_type"] = "road"
    return items


if __name__ == "__main__":
    # Quick manual check (needs network). Koramangala -> two Bengaluru points.
    demo = [
        {"name": "A", "lat": 12.9719, "lng": 77.6412, "distance_km": 0.0},
        {"name": "B", "lat": 12.9260, "lng": 77.6762, "distance_km": 0.0},
    ]
    annotate_road_distance(12.9352, 77.6245, demo)
    for d in demo:
        print(d["name"], d.get("distance_type"), d["distance_km"], "km",
              d.get("duration_min"), "min")
