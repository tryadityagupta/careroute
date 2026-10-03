"""
api/location.py — turn what the user gave us into coordinates.

A typed place wins over GPS. Errors are HTTP 422s whose text the page shows
as-is, so they are written for a patient, not a developer.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException

from careroute.maps.geocoding import NominatimGeocoder

MAX_PLACE_CHARS = 200


@dataclass
class ResolvedLocation:
    lat: float
    lng: float
    area: str
    info: dict          # what the page shows: {"label", "source", ...}


class LocationResolver:
    def __init__(self, geocoder: NominatimGeocoder):
        self.geocoder = geocoder

    @staticmethod
    def short_place(display_name: str) -> str:
        """'HSR Layout, Bengaluru South, Bengaluru Urban, Karnataka, ...'
        -> 'HSR Layout, Bengaluru South, Bengaluru Urban'."""
        return ", ".join(p.strip() for p in display_name.split(",")[:3])

    def resolve(self, location_text, lat, lng, *, required: bool) -> ResolvedLocation | None:
        """None when nothing was given and it isn't required (a follow-up that
        doesn't change location)."""
        text = (location_text or "").strip()
        if text:
            if len(text) > MAX_PLACE_CHARS:
                raise HTTPException(422, "That location is too long. Try just your "
                                         "area or a landmark, plus the city.")
            g = self.geocoder.geocode(text)
            if "lat" not in g:
                if g.get("error"):
                    raise HTTPException(422, "We couldn't look that place up right "
                                             "now. Try again in a moment, or allow "
                                             "location access.")
                raise HTTPException(422, f"We couldn't find \u201c{text}\u201d. Try "
                                         "your layout or area with the city, e.g. \u201cHSR "
                                         "Layout Sector 2, Bengaluru\u201d, or a well-known "
                                         "landmark near you.")
            label = self.short_place(g["display_name"])
            return ResolvedLocation(g["lat"], g["lng"], label, {
                "label": label, "source": "typed",
                "approximate": g.get("approximate", False),
                "broad": g.get("broad", False)})
        if lat is not None and lng is not None:
            return ResolvedLocation(lat, lng, "Current location",
                                    {"label": "your current location", "source": "gps"})
        if required:
            raise HTTPException(400, "We need your location. Allow location access, "
                                     "or type your area or a nearby landmark.")
        return None
