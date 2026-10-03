"""
providers/dummy.py — the offline directory: 10 mock Bengaluru providers.

Deterministic and network-free, so the agent harness tests run against REAL
tool results without a database or the internet.
"""

from __future__ import annotations

from careroute.maps.distance import haversine_km
from careroute.providers.base import ProviderDirectory


class JsonProviderDirectory(ProviderDirectory):
    backend_name = "dummy-json"

    def __init__(self, providers: list[dict]):
        self._providers = providers           # read-only reference data

    def find_providers(self, specialty, patient_lat, patient_lng, k=3, radius_m=8000):
        spec_matches = [
            {**p, "distance_km": haversine_km(patient_lat, patient_lng, p["lat"], p["lng"])}
            for p in self._providers if p["specialty"].lower() == specialty.lower()
        ]
        # Miss type 1: the specialty doesn't exist here at all. A bigger
        # radius can't fix that, so say what IS available.
        if not spec_matches:
            return {
                "match_found": False,
                "reason": f"No providers with specialty '{specialty}' exist in the directory.",
                "available_specialties": sorted({p["specialty"] for p in self._providers}),
                "hint": ("Pick the most clinically appropriate specialty from "
                         "available_specialties and call find_providers again."),
            }
        in_radius = [p for p in spec_matches if p["distance_km"] * 1000 <= radius_m]
        # Miss type 2: it exists, just not this close. A bigger radius CAN fix it.
        if not in_radius:
            nearest = min(spec_matches, key=lambda p: p["distance_km"])
            return {
                "match_found": False,
                "reason": (f"No {specialty} providers within {radius_m / 1000:.1f} km "
                           f"(nearest is {nearest['distance_km']} km away)."),
                "hint": "Call find_providers again with a larger radius_m.",
            }
        in_radius.sort(key=lambda p: p["distance_km"])
        return in_radius[:k]

    def find_general_facilities(self, patient_lat, patient_lng, k=3, radius_m=8000):
        scored = [{**p, "distance_km": haversine_km(patient_lat, patient_lng, p["lat"], p["lng"]),
                   "is_specialist_match": False} for p in self._providers]
        scored = [p for p in scored if p["distance_km"] * 1000 <= radius_m]
        if not scored:
            return {"match_found": False,
                    "reason": f"No facilities of any kind within {radius_m / 1000:.1f} km."}
        scored.sort(key=lambda p: p["distance_km"])
        return {
            "disclaimer": ("These are general healthcare providers, NOT matches for "
                           "the requested specialty. Present them only as general "
                           "options, never as specialists."),
            "facilities": scored[:k],
        }

    def find_pharmacies(self, patient_lat, patient_lng, k=3, radius_m=8000):
        # The demo JSON is not a pharmacy directory: be honest, never point at
        # a clinic.
        return {"match_found": False,
                "reason": "Pharmacy lookup needs a real backend "
                          "(USE_REAL_PROVIDERS=osm or google)."}
