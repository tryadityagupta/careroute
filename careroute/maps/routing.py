"""
maps/routing.py — real road distance + ETA via OSRM.

WHY: straight-line distance is fine for RANKING, but the number looks wrong
next to Maps (a clinic 170 m away by road once showed as 3.78 km). OSRM walks
the actual road graph.

DESIGN
  * haversine stays upstream as a free pre-filter; OSRM only sees a shortlist.
  * ONE Table-service request per lookup (one source -> many destinations).
  * GRACEFUL FALLBACK: if OSRM is down, distances stay straight-line and are
    labelled distance_type="straight_line" — never silently misrepresented,
    and "is there a provider" never depends on OSRM.

Point OSRM_BASE_URL at the self-hosted `osrm` Compose service for load tests.

ASYNC: one shared httpx.AsyncClient (pooled keep-alive connections) is
injected by the container; a request waiting on OSRM yields the event loop.
"""

from __future__ import annotations

import time

import httpx

from careroute.maps.cache import LRUCache


class OsrmRouter:
    def __init__(self, base_url: str, *, timeout_s: float = 3.0,
                 cache_size: int = 20000,
                 client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        # Road distances between two points barely change; key on ~11 m cells.
        self._cache = LRUCache(cache_size)
        # One pooled client per process: keep-alive instead of a new TCP
        # (and TLS) handshake per lookup.
        self._http = client or httpx.AsyncClient()

    @staticmethod
    def _key(origin_lat, origin_lng, lat, lng):
        return (round(origin_lat, 4), round(origin_lng, 4), round(lat, 4), round(lng, 4))

    async def road_distances(self, origin_lat: float, origin_lng: float,
                       dests: list[tuple[float, float]]) -> list[dict] | None:
        """[{distance_km, duration_min}] aligned with dests, or None if OSRM
        is unavailable (the caller falls back to haversine)."""
        if not dests:
            return []
        keys = [self._key(origin_lat, origin_lng, lat, lng) for lat, lng in dests]
        cached = [self._cache.get(k) for k in keys]
        if all(c is not None for c in cached):
            return cached                                   # zero network

        # OSRM speaks lng,lat. Coordinate 0 is the source; sources=0 keeps the
        # response to a single row instead of an NxN matrix.
        coords = ";".join([f"{origin_lng},{origin_lat}"]
                          + [f"{lng},{lat}" for lat, lng in dests])
        url = f"{self.base_url}/table/v1/driving/{coords}"
        params = {"sources": "0", "annotations": "distance,duration"}

        t0 = time.perf_counter()
        try:
            resp = await self._http.get(url, params=params, timeout=self.timeout_s)
            print(f"[routing] OSRM took {time.perf_counter() - t0:.1f}s")
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "Ok":
                print(f"[routing] OSRM returned code={data.get('code')}; "
                      "falling back to haversine")
                return None
            dist_row = (data.get("distances") or [[]])[0]
            dur_row = (data.get("durations") or [[]])[0]
            out = []
            for i in range(1, len(dests) + 1):
                d_m = dist_row[i] if i < len(dist_row) else None
                t_s = dur_row[i] if i < len(dur_row) else None
                if d_m is None:
                    # All road or all straight-line — never a mixed list.
                    print("[routing] OSRM gave no distance for a point; "
                          "falling back to haversine")
                    return None
                out.append({"distance_km": round(d_m / 1000, 2),
                            "duration_min": round(t_s / 60) if t_s is not None else None})
            for k, v in zip(keys, out):
                self._cache.put(k, v)
            return out
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
            print(f"[routing] OSRM unavailable ({str(e)[:80]}); using haversine")
            return None

    async def annotate(self, origin_lat: float, origin_lng: float, items: list[dict]) -> list[dict]:
        """Upgrade a shortlist of facility dicts to road distance, IN PLACE.

        Success: overwrites distance_km, adds drive_min_no_traffic and
        distance_type="road". Failure: keeps haversine, marks
        distance_type="straight_line". Returns items for chaining.
        """
        if not items:
            return items
        road = await self.road_distances(origin_lat, origin_lng,
                                   [(it["lat"], it["lng"]) for it in items])
        if road is None:
            for it in items:
                it.setdefault("distance_type", "straight_line")
            return items
        for it, r in zip(items, road):
            it["distance_km"] = r["distance_km"]
            # OSRM's duration is FREE-FLOW. The key name says so because the
            # model reads field names: "duration_min" got presented as an ETA.
            it["drive_min_no_traffic"] = r["duration_min"]
            it["distance_type"] = "road"
        return items
