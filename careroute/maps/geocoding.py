"""
maps/geocoding.py — place name <-> coordinates, via Nominatim.

Forward geocoding used to live in tools.py and reverse geocoding in
emergency.py, each with its own base URL, User-Agent and timeout logic. One
class now owns both, so "which Nominatim, how gently" is decided once.

NOMINATIM_URL points at the self-hosted `nominatim` Compose service. Unset,
it falls back to the public server, whose usage policy is at most 1 request
per second — so ONLY the public server is throttled (throttling a self-hosted
instance would serialise every geocode on the replica for no reason).
"""

from __future__ import annotations

import re
import threading
import time

import requests

from careroute.config import PUBLIC_NOMINATIM
from careroute.maps.cache import LRUCache

_USER_AGENT = "CareRoute/1.0 (care-routing demo)"


class NominatimGeocoder:
    # Filler that only confuses a geocoder ("near Forum Mall").
    _FILLER = re.compile(r"\b(near|opp(osite)?|behind|next to|beside|in front of|"
                         r"close to|around)\b\.?", re.I)
    _PIN = re.compile(r"\b\d{6}\b")                    # Indian PIN code
    _HOUSE_NO = re.compile(r"(#|no\.?\s*)?[\d/\-\s]+[a-z]?", re.I)

    def __init__(self, base_url: str = PUBLIC_NOMINATIM, *, countries: str = "in",
                 cache_size: int = 10000, session: requests.Session | None = None):
        self.base_url = base_url.rstrip("/")
        self.countries = countries.strip()
        self.throttled = self.base_url == PUBLIC_NOMINATIM
        self._http = session or requests.Session()
        # A named place resolves to the same point every time: never look one
        # up twice. Bounded, so user-typed strings can't grow memory forever.
        self._cache = LRUCache(cache_size)
        self._throttle_lock = threading.Lock()
        self._last_call = 0.0

    # --- HTTP ----------------------------------------------------------------
    def _wait_turn(self) -> None:
        if not self.throttled:
            return
        with self._throttle_lock:
            wait = 1.0 - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    def _search(self, query: str) -> list[dict]:
        """One raw /search call. Tests override this method."""
        self._wait_turn()
        params = {"q": query, "format": "jsonv2", "limit": 1}
        if self.countries:
            params["countrycodes"] = self.countries
        resp = self._http.get(f"{self.base_url}/search", params=params,
                              headers={"User-Agent": _USER_AGENT},
                              timeout=10 if self.throttled else 3)
        resp.raise_for_status()
        return resp.json()

    def reverse_country(self, lat: float, lng: float) -> str | None:
        """ISO country code for a point, or None. Raises nothing: the caller
        (the emergency number) must never depend on this succeeding."""
        try:
            resp = self._http.get(
                f"{self.base_url}/reverse",
                params={"lat": lat, "lon": lng, "format": "json", "zoom": 3},
                headers={"User-Agent": _USER_AGENT},
                timeout=6 if self.throttled else 3)
            resp.raise_for_status()
            cc = (resp.json().get("address", {}).get("country_code") or "").upper()
            return cc or None
        except (requests.RequestException, ValueError) as e:
            print(f"[geocoding] reverse-geocode failed ({str(e)[:60]})")
            return None

    # --- Query planning --------------------------------------------------------
    @classmethod
    def fallback_queries(cls, place: str) -> list[str]:
        """Most specific first, then progressively broader.

        Users type "Prestige Shantiniketan, Whitefield, Bangalore". The
        apartment is often not in OSM, but the area after it is — so try the
        whole thing, then drop leading comma-separated parts. House numbers
        and PIN codes are stripped; the list is bounded (public Nominatim is
        1 req/s) but always keeps the broad tail, where long Indian postal
        addresses actually resolve.
        """
        cleaned = cls._FILLER.sub(" ", place)
        cleaned = cls._PIN.sub(" ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        parts = [p for p in parts if not cls._HOUSE_NO.fullmatch(p)]
        suffixes = [", ".join(parts[i:]) for i in range(len(parts))] or [cleaned]
        if len(suffixes) > 5:
            tail = suffixes[:-2] if len(parts) > 3 else suffixes
            suffixes = suffixes[:2] + [q for q in tail[-3:] if q not in suffixes[:2]]
        queries = list(dict.fromkeys(suffixes))
        if place.strip() not in queries:
            queries.insert(0, place.strip())
        return queries[:6]

    # --- Public API --------------------------------------------------------
    def geocode(self, place: str) -> dict:
        """{lat, lng, display_name, matched_query, approximate, broad}, or an
        error dict, or {match_found: False, reason}.

        approximate=True: only a broader part of what was typed was found.
        broad=True: the match is a whole city or bigger.
        """
        key = (place or "").strip().lower()
        if not key:
            return {"error": "empty place"}
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        queries = self.fallback_queries(place)
        for q in queries:
            try:
                hits = self._search(q)
            except requests.RequestException as e:
                return {"error": f"Geocoding failed: {e}"}
            if not hits:
                continue
            top = hits[0]
            # place_rank <= 16: city/district or bigger. A FALLBACK that only
            # matched a whole city would search around the city centre, km from
            # the user, so it is rejected; a city the user typed is accepted
            # but flagged broad.
            rank = int(top.get("place_rank") or 30)
            approximate = q != queries[0]
            if approximate and rank <= 16:
                break
            out = {"lat": float(top["lat"]), "lng": float(top["lon"]),
                   "display_name": top.get("display_name", q),
                   "matched_query": q, "approximate": approximate,
                   "broad": rank <= 16}
            self._cache.put(key, out)
            return out
        return {"match_found": False,
                "reason": f"Could not find a place named '{place}'."}

    def clear_cache(self) -> None:
        self._cache.clear()
