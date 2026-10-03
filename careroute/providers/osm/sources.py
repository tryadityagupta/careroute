"""
providers/osm/sources.py — where raw OpenStreetMap elements come from.

Two interchangeable sources behind one interface (OSM_SOURCE picks one):

    OverpassSource  public mirrors raced in parallel + on-disk cache with
                    stale-if-error. Fine for a demo; NEVER load-test it.
    PostgisSource   self-hosted import (infra/geo/import.sh). The scaled design.

Both return the same element shape ({lat, lon, tags} or {center: {...}}), so
OsmProviderDirectory is source-agnostic. Both report an unreachable
directory with the SAME typed error payload, because "the lookup failed" and
"nothing was found" are different facts and the model once conflated them.
"""

from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from careroute.maps.postgis import GeoDatabase, GeoUnavailable
from careroute.providers.osm.matching import PHARMACY_NAME_RE

Elements = list[dict]


def unreachable_payload(last_error) -> dict:
    """Payload for 'the directory could not be queried'.

    A TRANSPORT failure is not "no specialists nearby": the model once told a
    cardiac patient no cardiologist was found while one sat 3.3 km away. The
    payload says so in data, because a prompt is advisory and data is enforced.
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


class ElementSource(ABC):
    name: str = "abstract"

    @abstractmethod
    def elements(self, lat: float, lng: float, radius_m: int) -> tuple[Elements | None, dict | None]:
        """(elements, None) on success, (None, error_payload) on failure."""


class PostgisSource(ElementSource):
    """One indexed spatial query; no cache needed (every replica shares it)."""
    name = "postgis"

    def __init__(self, db: GeoDatabase):
        self.db = db

    def elements(self, lat, lng, radius_m):
        t0 = time.perf_counter()
        try:
            rows = self.db.nearby_elements(lat, lng, radius_m)
        except GeoUnavailable as e:
            print(f"[osm] postgis lookup failed: {e}")
            return None, unreachable_payload(e)
        print(f"[osm] postgis r={radius_m} rows={len(rows)} "
              f"took {(time.perf_counter() - t0) * 1000:.0f}ms")
        return rows, None


class OverpassDiskCache:
    """Per-(~1 km cell, radius) cache of Overpass fetches, persisted as JSON.

    The Overpass query never mentions the specialty (filtering is client-side),
    so ONE fetch serves every specialty and every nearby user. Distances are
    always recomputed from the caller's exact point, so a hit is never wrong.
    """

    TTL_S = 7 * 24 * 3600
    # Part of the key: bump when the Overpass QUERY changes shape, so entries
    # fetched with the old query are never served.
    QUERY_VERSION = 4

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict = self._load()

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        try:
            os.makedirs(self.path.parent, exist_ok=True)
            with self._lock, open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
        except (OSError, RuntimeError) as e:     # RuntimeError: dict changed mid-dump
            print(f"[osm] cache write failed (non-fatal): {e}")

    def _prefix(self, lat, lng) -> str:
        # 2 dp ~= 1.1 km cells: nearby users share a fetch.
        return f"v{self.QUERY_VERSION}|{lat:.2f},{lng:.2f},r"

    def find(self, lat, lng, radius_m, now, *, fresh=True):
        """Smallest cached radius >= radius_m in this cell (a 30 km fetch
        contains the 8 km answer). fresh=False also accepts expired entries,
        for stale-if-error."""
        prefix = self._prefix(lat, lng)
        best = None
        for key, hit in list(self._data.items()):
            if not key.startswith(prefix):
                continue
            try:
                r = int(key[len(prefix):])
            except ValueError:
                continue
            if r < radius_m or (fresh and now - hit["fetched_at"] >= self.TTL_S):
                continue
            if best is None or r < best[0]:
                best = (r, hit)
        return best[1] if best else None

    def put(self, lat, lng, radius_m, elements, now) -> None:
        with self._lock:
            self._data[f"{self._prefix(lat, lng)}{int(radius_m)}"] = {
                "fetched_at": now, "elements": elements}
        self._save()


class OverpassSource(ElementSource):
    """Public Overpass mirrors — three independent operators, raced in parallel."""
    name = "overpass"

    ENDPOINTS = (
        "https://overpass-api.de/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
    )
    # A dense-city 8 km query routinely takes ~12 s; this matches the query's
    # own [timeout:25]. Mirrors race, so the worst case stays ~25 s.
    HTTP_TIMEOUT_S = 25
    RETRY_IF_FAILED_WITHIN_S = 8
    RETRY_BACKOFF_S = 1.5
    _HEADERS = {"User-Agent": "CareRoute-demo/1.0 (learning project)"}

    def __init__(self, cache: OverpassDiskCache, endpoints=ENDPOINTS):
        self.cache = cache
        self.endpoints = tuple(endpoints)

    @staticmethod
    def build_query(lat, lng, radius_m) -> str:
        # Both tagging schemes (amenity=*, healthcare=*), chemist shops, and
        # generic shops whose NAME says chemist. infra/geo/healthcare.lua
        # mirrors this filter for PostGIS.
        a = f"(around:{radius_m},{lat},{lng})"
        return f"""
    [out:json][timeout:25];
    (
      node["amenity"~"hospital|clinic|doctors|pharmacy"]{a};
      way["amenity"~"hospital|clinic|doctors|pharmacy"]{a};
      node["healthcare"]{a};
      way["healthcare"]{a};
      node["shop"~"chemist|pharmacy|medical_supply|medical"]{a};
      way["shop"~"chemist|pharmacy|medical_supply|medical"]{a};
      node["shop"]["name"~"{PHARMACY_NAME_RE}",i]{a};
      way["shop"]["name"~"{PHARMACY_NAME_RE}",i]{a};
    );
    out center tags;
    """

    def _post(self, url: str, query: str) -> Elements:
        resp = requests.post(url, data={"data": query}, headers=self._HEADERS,
                             timeout=self.HTTP_TIMEOUT_S)
        resp.raise_for_status()
        # .json() inside the worker: a 200 with an HTML error page counts as a
        # failure and the race continues.
        return resp.json().get("elements", [])

    def _race(self, query: str):
        """First mirror to succeed wins: (elements, None) or (None, last_error)."""
        ex = ThreadPoolExecutor(max_workers=len(self.endpoints))
        futures = {ex.submit(self._post, u, query): u for u in self.endpoints}
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
            ex.shutdown(wait=False, cancel_futures=True)

    def elements(self, lat, lng, radius_m):
        now = time.time()
        hit = self.cache.find(lat, lng, radius_m, now)
        if hit:
            return hit["elements"], None                 # fresh cache: zero network

        query = self.build_query(lat, lng, radius_m)
        t0 = time.perf_counter()
        elements, last_error = self._race(query)
        if elements is None and time.perf_counter() - t0 < self.RETRY_IF_FAILED_WITHIN_S:
            print("[osm] all mirrors failed fast — one more race")
            time.sleep(self.RETRY_BACKOFF_S)
            elements, last_error = self._race(query)
        print(f"[osm] overpass fetch r={radius_m} took "
              f"{time.perf_counter() - t0:.1f}s ok={elements is not None}")
        if elements is not None:
            self.cache.put(lat, lng, radius_m, elements, now)
            return elements, None

        stale = self.cache.find(lat, lng, radius_m, now, fresh=False)
        if stale:
            age_h = (now - stale["fetched_at"]) / 3600
            print(f"[osm] all mirrors down — serving cached data ({age_h:.0f} h old)")
            return stale["elements"], None
        return None, unreachable_payload(last_error)
