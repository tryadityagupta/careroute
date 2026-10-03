"""
domain/emergency.py — emergency triage helper.

get_help(lat, lng) returns the LOCAL emergency number to call now, plus
(best-effort) the nearest hospitals.

DESIGN PRINCIPLE: the NUMBER is the critical output and must never depend on a
network call succeeding. Country comes from a cached reverse-geocode when
possible, then a coarse offline box check, then 112 — a GSM-standard number
that reaches emergency services across the EU, India and much of the world.
The hospital list is layered on top; if the directory is down, the number
still goes out.

CareRoute cannot dispatch help; it tells the user who to call, fast. A real
product would verify numbers against an authoritative per-country source.
"""

from __future__ import annotations

import asyncio
import re
import time

from careroute.maps.cache import LRUCache
from careroute.maps.geocoding import NominatimGeocoder
from careroute.providers.osm.directory import OsmProviderDirectory


class EmergencyNumberResolver:
    # ISO 3166-1 alpha-2 -> what to dial. 112 is the safe default.
    NUMBERS = {
        "IN": "112 (ambulance: 108)",
        "US": "911", "CA": "911",
        "GB": "999 (or 112)",
        "AU": "000 (or 112)",
        "NZ": "111",
        "JP": "119 (ambulance)",
        "KR": "119",
        "AE": "998 (ambulance)",
        **{cc: "112" for cc in ("IE", "FR", "DE", "ES", "IT", "NL", "BE", "PT", "SE",
                                "DK", "FI", "PL", "AT", "GR", "CZ", "RO", "HU", "NO", "CH")},
    }
    DEFAULT = "112"
    TTL_S = 30 * 24 * 3600

    def __init__(self, geocoder: NominatimGeocoder, cache_size: int = 50000):
        self.geocoder = geocoder
        # In memory, bounded. (It used to rewrite a JSON file on disk inside
        # the request path for every new ~11 km cell.)
        self._cache = LRUCache(cache_size)

    @staticmethod
    def offline_country(lat: float, lng: float) -> str | None:
        """Crude boxes, not borders — right for the places we care most about
        when Nominatim is unreachable."""
        if 6 <= lat <= 37 and 68 <= lng <= 98:
            return "IN"
        if 24 <= lat <= 72 and -170 <= lng <= -52:
            return "US"                          # US/CA share 911
        return None

    async def country_code(self, lat: float, lng: float) -> str | None:
        key = f"{lat:.1f},{lng:.1f}"             # ~11 km: plenty for a country
        hit = self._cache.get(key)
        if hit and time.time() - hit[1] < self.TTL_S:
            return hit[0]
        cc = await self.geocoder.reverse_country(lat, lng) or self.offline_country(lat, lng)
        if cc:
            self._cache.put(key, (cc, time.time()))
        return cc

    async def number_for(self, lat: float, lng: float) -> tuple[str, str | None]:
        cc = await self.country_code(lat, lng)
        return self.NUMBERS.get(cc, self.DEFAULT), cc


class EmergencyService:
    # OSM tags many single-doctor or single-specialty practices as hospitals
    # ("Dr Ramesh Dalwai Spine Surgeon" was offered as an ER for chest pain).
    NOT_AN_ER = re.compile(
        r"^dr\.?\s|\bspine\b|\beye\b|ophthalm|dental|dentist|\bskin\b|\bhair\b|"
        r"cosmetic|\bivf\b|fertility|physiotherap|ayurved|homo?eopath|\bdiagnostic", re.I)
    SEARCH_RADIUS_M = 8000

    def __init__(self, numbers: EmergencyNumberResolver, directory: OsmProviderDirectory | None):
        self.numbers = numbers
        self.directory = directory               # None: number only

    @classmethod
    def er_candidates(cls, facilities: list[dict]) -> list[dict]:
        return [f for f in OsmProviderDirectory.dedupe(facilities)
                if "hospital" in (f.get("facility") or "")
                and f.get("_er") != "no"
                and not cls.NOT_AN_ER.search(f["name"])]

    async def nearest_hospitals(self, lat: float, lng: float, k: int = 3) -> list[dict]:
        if self.directory is None:
            return []
        try:
            facilities, err = await self.directory.fetch_nearby(lat, lng, self.SEARCH_RADIUS_M)
            if err or not facilities:
                return []
            hosp = self.er_candidates(facilities)[:k]
            await self.directory.router.annotate(lat, lng, hosp)
            return [{"name": h["name"], "distance_km": h["distance_km"],
                     "drive_min_no_traffic": h.get("drive_min_no_traffic"),
                     "distance_type": h.get("distance_type"),
                     # "yes" only when OSM says so — never promise an ER.
                     "emergency_department": "yes" if h.get("_er") == "yes" else "not confirmed"}
                    for h in hosp]
        except Exception as e:                   # never let hospitals break the number
            print(f"[emergency] hospital lookup failed (non-fatal): {str(e)[:60]}")
            return []

    async def get_help(self, patient_lat: float, patient_lng: float) -> dict:
        # The number (reverse geocode) and the hospitals (directory + OSRM) are
        # independent, so they run CONCURRENTLY: the emergency answer costs the
        # slower of the two lookups, not their sum. Neither can fail the other:
        # nearest_hospitals never raises, and the number has offline fallbacks.
        (number, cc), hospitals = await asyncio.gather(
            self.numbers.number_for(patient_lat, patient_lng),
            self.nearest_hospitals(patient_lat, patient_lng))
        return {
            "emergency": True,
            "emergency_number": number,
            "country": cc,
            "advice": (f"This may be a medical emergency. Tell the user to call {number} "
                       "immediately, or go to the nearest emergency department. State "
                       "this FIRST, before any specialist recommendation. An ambulance "
                       "is preferable to driving themselves: the crew can start care on "
                       "the way. Drive times are without traffic."),
            "nearest_hospitals": hospitals,
            "disclaimer": ("CareRoute cannot contact emergency services — the user must "
                           "call. Numbers are best-effort by location; 112 reaches "
                           "emergency services in many countries if unsure."),
        }
