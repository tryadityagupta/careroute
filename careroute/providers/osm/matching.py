"""
providers/osm/matching.py — deciding what an OSM facility IS.

Three questions, each answered by rules learnt from real misses:
  * Does this facility match a specialty?           SpecialtyMatcher
  * Is it a pharmacy / chemist?                     looks_like_pharmacy()
  * Is it a narrow boutique (skin/hair/dental...)?  is_narrow_specialty()

OSM is community-mapped, so specialty tagging is sparse and inconsistent:
"orthopaedics" (British), clinic NAMES like "Sparsh Ortho Care", and Indian
chemists mapped as shop=medical_supply or a generic shop called "X Medicals".
Every rule below exists because of a wrong answer someone actually got.

KEEP IN SYNC with infra/geo/healthcare.lua (which import filter decides what
reaches PostGIS) — PHARMACY_NAME_RE is mirrored there.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Specialty matching. Keys are regexes matched at a WORD START of the
# requested specialty (a bare "ent" would otherwise fire for
# "Gastroenterology" and "Dentistry"); include/exclude are matched against
# facility names and speciality-tag tokens. An EXCLUDE handles ambiguous
# short forms: "ortho" must not match an orthodontic dental clinic.
# ---------------------------------------------------------------------------
_SPECIALTY_PATTERNS = [
    (("ortho",), r"orthop(a)?ed|\bortho\b|bone\s*(and|&)\s*joint|joint\s*replacement|fracture",
     r"orthodont|dental|dentist|\bteeth\b"),
    (("pediatr", "paediatr", "child"),
     r"pa?ediatr|children'?s\s*(hospital|clinic)|\bchild\s*care\b", None),
    (("gyn", "obstet", "women"),
     r"gyn(a)?ecolog|obstetric|maternity|women'?s\s*(hospital|clinic)|\bivf\b", None),
    (("ent\\b", "otolaryng", "ear\\b"),
     r"\bent\b|otolaryng|otorhinolaryng|ear,?\s*nose", None),
    (("cardio", "heart"), r"cardi|\bheart\b", None),
    (("derma", "skin"), r"dermatolog|\bskin\b", None),
    (("neuro",), r"neurolog|neuro\s*(care|clinic|hospital)|\bneuro\b", r"neurosurg(?!ery)"),
    (("psychiat", "mental"), r"psychiatr|mental\s*health|de-?addiction", None),
    (("ophthalm", "eye"), r"ophthalm|\beye\b", None),
    (("gastro",), r"gastro", None),
    (("uro",), r"urolog|\buro\b", None),
    (("oncol", "cancer"), r"oncolog|cancer", None),
    (("pulmon", "chest", "respir"), r"pulmon|\bchest\b|respirat", None),
    (("nephro", "kidney"), r"nephrolog|kidney|dialysis", None),
    (("endocrin", "diabet"), r"endocrin|diabet", None),
    (("dent",), r"dent|orthodont", None),
    (("general", "family"),
     r"general\s*(medicine|practice|physician)|family\s*(medicine|doctor)", None),
]


class SpecialtyMatcher:
    """matcher = SpecialtyMatcher("Orthopedics"); matcher("Sparsh Ortho Care")"""

    def __init__(self, specialty: str):
        self.specialty = specialty
        spec = specialty.lower()
        self._include = self._exclude = None
        for keys, inc, exc in _SPECIALTY_PATTERNS:
            if any(re.search(r"\b" + k, spec) for k in keys):
                self._include = re.compile(inc, re.I)
                self._exclude = re.compile(exc, re.I) if exc else None
                break
        # Unknown specialty: a crude stem ('psychiatry' -> 'psychiatr' matches
        # 'Psychiatric'). A heuristic, not lemmatisation.
        self._stem = None if self._include else spec.rstrip("sy")

    def __call__(self, text: str) -> bool:
        if self._include is None:
            return self._stem in text.lower()
        return bool(self._include.search(text)) and not (
            self._exclude and self._exclude.search(text))


# ---------------------------------------------------------------------------
# Pharmacy detection. Overpass regexes are POSIX ERE: keep this pattern simple.
# ---------------------------------------------------------------------------
PHARMACY_NAME_RE = "medical|pharma|chemist|drug ?store|aushadh"
_PHARMACY_NAME = re.compile(PHARMACY_NAME_RE, re.I)
_PHARMACY_SHOPS = ("chemist", "pharmacy", "medical_supply", "medical")


def looks_like_pharmacy(name: str, amenity: str, healthcare: str, shop: str) -> bool:
    if "pharmacy" in amenity or "pharmacy" in healthcare:
        return True
    if shop in _PHARMACY_SHOPS:
        return True
    # Name-based only for SHOPS: "Manipal Medical Centre" (amenity=hospital)
    # is not a chemist and must not be relabelled.
    return bool(shop) and not amenity and bool(_PHARMACY_NAME.search(name))


# ---------------------------------------------------------------------------
# Narrow single-specialty boutiques: noise in a GENERAL list (a hair clinic
# kept being suggested for diarrhoea). Never filtered from specialist results,
# never applied to pharmacies or full hospitals.
# ---------------------------------------------------------------------------
_NARROW_SPECIALTY = (
    "skin", "hair", "laser", "cosmetic", "aesthetic", "derma",
    "dental", "dentist", "orthodont", "ophthal", "optical", "optician",
    "fertility", "ivf", "veterinary",
)


def is_narrow_specialty(item: dict) -> bool:
    if item.get("is_pharmacy") or "hospital" in (item.get("facility") or ""):
        return False
    return any(kw in item.get("_signal", "") for kw in _NARROW_SPECIALTY)
