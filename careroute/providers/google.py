"""
providers/google.py — real providers via Google Places API (Text Search).

Setup: enable "Places API (New)" + billing, create a key restricted to it,
set GOOGLE_MAPS_API_KEY and USE_REAL_PROVIDERS=google. Text Search is a Pro
SKU with a free monthly allowance; set a budget alert anyway.

The three lookups used to be three copies of the same request code; they now
share one _search() and differ only in the query text and result shape.
"""

from __future__ import annotations

import requests

from careroute.maps.distance import haversine_km
from careroute.providers.base import ProviderDirectory


class GooglePlacesDirectory(ProviderDirectory):
    backend_name = "google-places"
    URL = "https://places.googleapis.com/v1/places:searchText"
    MAX_RADIUS_M = 50000             # the API's cap on a locationBias circle
    # FieldMask = pay only for the fields you ask for. Always set it.
    FIELD_MASK = "places.displayName,places.formattedAddress,places.location"

    def __init__(self, api_key: str, session: requests.Session | None = None):
        self.api_key = api_key
        self._http = session or requests.Session()

    def _clamp(self, radius_m) -> int:
        return max(500, min(int(radius_m), self.MAX_RADIUS_M))

    def _search(self, text: str, lat: float, lng: float, radius_m: int):
        """Returns (results, None) or (None, error_dict). Results carry
        distance_km and are sorted nearest-first (Places ranks by its own
        relevance; we re-rank strictly by distance)."""
        if not self.api_key:
            return None, {"error": "GOOGLE_MAPS_API_KEY not set"}
        body = {"textQuery": text,
                "locationBias": {"circle": {"center": {"latitude": lat, "longitude": lng},
                                            "radius": float(radius_m)}},
                "maxResultCount": 10}
        headers = {"Content-Type": "application/json", "X-Goog-Api-Key": self.api_key,
                   "X-Goog-FieldMask": self.FIELD_MASK}
        try:
            resp = self._http.post(self.URL, headers=headers, json=body, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            return None, {"error": f"Places API call failed: {e}"}
        out = []
        for p in resp.json().get("places", []):
            loc = p.get("location", {})
            plat, plng = loc.get("latitude"), loc.get("longitude")
            if plat is None or plng is None:
                continue
            out.append({"name": p.get("displayName", {}).get("text", "Unknown"),
                        "facility": p.get("formattedAddress", ""),
                        "lat": plat, "lng": plng,
                        "distance_km": haversine_km(lat, lng, plat, plng)})
        out.sort(key=lambda x: x["distance_km"])
        return out, None

    def find_providers(self, specialty, patient_lat, patient_lng, k=3, radius_m=8000):
        results, err = self._search(f"{specialty} doctor", patient_lat, patient_lng,
                                    self._clamp(radius_m))
        if err:
            return err
        if not results:
            return {"match_found": False,
                    "reason": f"Google Places returned no results for '{specialty}' near the patient.",
                    "hint": "Retry with a larger radius_m, or try a broader specialty term."}
        return results[:k]

    def find_general_facilities(self, patient_lat, patient_lng, k=3, radius_m=8000):
        radius_m = self._clamp(radius_m)
        results, err = self._search("hospital or clinic", patient_lat, patient_lng, radius_m)
        if err:
            return err
        if not results:
            return {"match_found": False,
                    "reason": f"No healthcare facilities within {radius_m / 1000:.1f} km."}
        facilities = [{k_: v for k_, v in r.items() if k_ not in ("lat", "lng")}
                      | {"is_specialist_match": False} for r in results[:k]]
        return {"disclaimer": ("These are general healthcare facilities, NOT verified "
                               "specialists. Present them only as general options."),
                "facilities": facilities}

    def find_pharmacies(self, patient_lat, patient_lng, k=3, radius_m=8000):
        radius_m = self._clamp(radius_m)
        results, err = self._search("pharmacy or chemist", patient_lat, patient_lng, radius_m)
        if err:
            return err
        if not results:
            return {"match_found": False,
                    "reason": f"No pharmacy found within {radius_m / 1000:.1f} km.",
                    "hint": "Do NOT offer a hospital or clinic as a pharmacy."}
        pharmacies = [{k_: v for k_, v in r.items() if k_ not in ("lat", "lng")}
                      | {"is_pharmacy": True} for r in results[:k]]
        return {"disclaimer": "Nearby pharmacies/chemists for obtaining medicines.",
                "pharmacies": pharmacies}
