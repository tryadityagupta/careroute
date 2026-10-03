"""
providers/osm/directory.py — the OpenStreetMap provider directory.

Free, no key. Data-quality reality: OSM specialty tagging is sparse, and two
bugs came from that, both fixed by STRUCTURE rather than prompting:

  1. Ranking specialty matches first and silently falling back to
     nearest-anything recommended a dental clinic for anxiety attacks.
     -> results are FILTERED to real matches.
  2. Putting nearby general facilities in the miss payload "for context" got
     them listed as specialists. -> they come back under
     general_alternatives, each labelled is_specialist_match=false, and the
     answer guard counts them as OFFERED, never confirmed.

Lesson: a prompt instruction is advisory; a data structure and a code guard
are enforced.
"""

from __future__ import annotations

from careroute.maps.distance import haversine_km
from careroute.maps.routing import OsrmRouter
from careroute.providers.base import ProviderDirectory
from careroute.providers.osm.matching import (SpecialtyMatcher, is_narrow_specialty,
                                              looks_like_pharmacy)
from careroute.providers.osm.sources import ElementSource

# Keys used internally, stripped before a payload reaches the model.
_PRIVATE_KEYS = ("_tag_tokens", "_signal", "_er")


def _strip_private(items: list[dict]) -> None:
    for f in items:
        for k in _PRIVATE_KEYS:
            f.pop(k, None)


class OsmProviderDirectory(ProviderDirectory):
    backend_name = "openstreetmap"
    # Never trust model-supplied arguments: a hallucinated radius_m=9999999
    # must not hammer the source or blow the timeout.
    MIN_RADIUS_M = 500
    MAX_RADIUS_M = 30000

    def __init__(self, source: ElementSource, router: OsrmRouter):
        self.source = source
        self.router = router

    # --- shared plumbing -------------------------------------------------------
    @classmethod
    def clamp_radius(cls, radius_m) -> int:
        return max(cls.MIN_RADIUS_M, min(int(radius_m), cls.MAX_RADIUS_M))

    @staticmethod
    def dedupe(items: list[dict]) -> list[dict]:
        """Big facilities exist as a node AND a way (NIMHANS showed up twice).
        Keep the nearest instance; callers pass distance-sorted lists."""
        seen, out = set(), []
        for f in items:
            key = f["name"].lower()
            if key != "unnamed facility" and key in seen:
                continue
            seen.add(key)
            out.append(f)
        return out

    async def fetch_nearby(self, lat: float, lng: float, radius_m: int,
                     specialty: str | None = None):
        """(facilities, None) sorted nearest-first, or (None, error_payload).

        Used by all three lookups AND the emergency service, so source, distance
        math and specialty matching live in exactly one place.
        """
        elements, err = await self.source.elements(lat, lng, radius_m)
        if err:
            return None, err
        matches = SpecialtyMatcher(specialty) if specialty else None
        out = []
        for el in elements:
            tags = el.get("tags", {})
            plat = el.get("lat") or el.get("center", {}).get("lat")
            plng = el.get("lon") or el.get("center", {}).get("lon")
            if plat is None or plng is None:
                continue
            dist = haversine_km(lat, lng, plat, plng)
            # A cache hit may come from a LARGER-radius fetch.
            if dist * 1000 > radius_m:
                continue
            name = tags.get("name")
            if not name:          # a place you cannot name cannot be recommended
                continue
            amenity = (tags.get("amenity") or "").lower()
            healthcare = (tags.get("healthcare") or "").lower()
            shop = (tags.get("shop") or "").lower()
            item = {
                "name": name,
                "facility": (tags.get("amenity") or tags.get("healthcare")
                             or tags.get("shop") or "healthcare"),
                "lat": plat,
                "lng": plng,
                "distance_km": dist,
                "is_pharmacy": looks_like_pharmacy(name, amenity, healthcare, shop),
                # What the place IS, lowered: for the narrow-clinic filter.
                "_signal": " ".join((name.lower(), amenity, healthcare, shop,
                                     tags.get("healthcare:speciality", "").lower())),
                # OSM emergency=yes|no — used by the emergency service.
                "_er": (tags.get("emergency") or "").lower(),
            }
            if matches:
                # Speciality tags are often long ";" lists, sometimes pasted
                # across many POIs; substring-matching the blob confirmed a
                # DENTAL clinic for Psychiatry. Match token by token and record
                # the evidence so no match is ever opaque.
                raw_tag = tags.get("healthcare:speciality", "")
                tokens = [t.strip().lower() for t in raw_tag.split(";") if t.strip()]
                if matches(name):
                    item.update(specialty_match=True, matched_via="name")
                elif any(matches(t) for t in tokens):
                    item.update(specialty_match=True, matched_via="speciality_tag",
                                speciality_tag=raw_tag, _tag_tokens=len(tokens))
                else:
                    item["specialty_match"] = False
            out.append(item)
        out.sort(key=lambda r: r["distance_km"])
        return out, None

    async def _shortlist_by_road(self, lat, lng, items: list[dict], k: int, sort_key) -> list[dict]:
        """Shortlist on the cheap straight-line order, upgrade JUST those to
        road distance (one OSRM call), re-rank, take k."""
        shortlist = items[:max(k + 4, 8)]
        await self.router.annotate(lat, lng, shortlist)
        shortlist.sort(key=sort_key)
        top = shortlist[:k]
        _strip_private(top)
        return top

    # --- the three lookups -----------------------------------------------------
    async def find_providers(self, specialty, patient_lat, patient_lng, k=3, radius_m=8000):
        radius_m = self.clamp_radius(radius_m)
        facilities, err = await self.fetch_nearby(patient_lat, patient_lng, radius_m, specialty)
        if err:
            return err

        matches = self.dedupe([f for f in facilities if f["specialty_match"]])
        if matches:
            # Evidence before distance: specialty in the NAME beats a tag, and a
            # dedicated tag beats a 20-specialty mega-list.
            def by_evidence(f):
                return (f.get("matched_via") != "name", f.get("_tag_tokens", 0),
                        f["distance_km"])
            matches.sort(key=by_evidence)
            return await self._shortlist_by_road(patient_lat, patient_lng, matches, k, by_evidence)

        # Miss: hand back the nearest GENERAL facilities now (labelled, with road
        # distance) so a minor complaint isn't forced to chase a far specialist.
        # Narrow boutiques are dropped unless that would leave nothing.
        general = [f for f in self.dedupe(facilities) if not is_narrow_specialty(f)]
        alternatives = (general or self.dedupe(facilities))[:3]
        await self.router.annotate(patient_lat, patient_lng, alternatives)
        alternatives = [{"name": f["name"], "facility": f["facility"],
                         "distance_km": f["distance_km"],
                         "drive_min_no_traffic": f.get("drive_min_no_traffic"),
                         "distance_type": f.get("distance_type"),
                         "is_specialist_match": False} for f in alternatives]
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

    async def find_general_facilities(self, patient_lat, patient_lng, k=3, radius_m=8000):
        radius_m = self.clamp_radius(radius_m)
        facilities, err = await self.fetch_nearby(patient_lat, patient_lng, radius_m)
        if err:
            return err
        if not facilities:
            return {"match_found": False,
                    "reason": f"No healthcare facilities at all within {radius_m / 1000:.1f} km."}
        facilities = self.dedupe(facilities)
        facilities = [f for f in facilities if not is_narrow_specialty(f)] or facilities
        top = await self._shortlist_by_road(patient_lat, patient_lng, facilities, k,
                                      lambda f: f["distance_km"])
        for f in top:
            f["is_specialist_match"] = False
        return {"disclaimer": ("These are general healthcare facilities, NOT verified "
                               "specialists. Present them only as general options."),
                "facilities": top}

    async def find_pharmacies(self, patient_lat, patient_lng, k=3, radius_m=8000):
        radius_m = self.clamp_radius(radius_m)
        facilities, err = await self.fetch_nearby(patient_lat, patient_lng, radius_m)
        if err:
            return err
        pharmacies = self.dedupe([f for f in facilities if f.get("is_pharmacy")])
        if not pharmacies:
            return {"match_found": False,
                    "reason": (f"No pharmacy is mapped in OpenStreetMap within "
                               f"{radius_m / 1000:.1f} km of the patient."),
                    "hint": ("Do NOT offer a hospital or clinic as a pharmacy. Tell the "
                             "user no pharmacy was found in the map data nearby; suggest "
                             "they widen the search or check locally.")}
        top = await self._shortlist_by_road(patient_lat, patient_lng, pharmacies, k,
                                      lambda f: f["distance_km"])
        return {
            "disclaimer": "Nearby pharmacies/chemists for obtaining medicines.",
            "pharmacies": top,
            "coverage_note": (
                "Source is OpenStreetMap, which is community-mapped and misses "
                "many small neighbourhood chemists. These are the nearest MAPPED "
                "pharmacies, not necessarily the nearest that exist. If the user "
                "says there is a closer one, believe them: say it is likely not "
                "in the map data yet — do not repeat the same list as if it "
                "were complete."),
        }
