"""
synthea_to_patients.py
----------------------
Turn a Synthea CSV export into the patients.json shape CareRoute expects.

CareRoute's tools.py looks patients up like this:
    _PATIENTS.get(patient_id) -> {name, age, area, lat, lng, history, current_medications}
so that's exactly the record we produce here.

Only THREE of the ~19 Synthea files are needed:
    patients.csv      -> identity, birthdate (age), location
    conditions.csv    -> "history"
    medications.csv   -> "current_medications" (active = no STOP date)
(allergies.csv is added as a bonus field; everything else — claims, observations,
imaging, devices, payers... — is billing/telemetry noise CareRoute doesn't use.)

Why we remap location to Bangalore by default:
    CareRoute matches patients to nearby providers by lat/lng within an 8 km
    radius, and its providers are all in Bangalore. Synthea patients live in
    Massachusetts, so with US coordinates find_providers() returns nothing.
    Remapping keeps the demo coherent. Pass --location us to keep the originals.

Usage:
    python synthea_to_patients.py --input ./synthea_sample_data_csv_latest \
                                  --output ./data/patients.json
"""

import argparse
import json
import random
import re
from datetime import date

import pandas as pd

# "today" for age calculation; fixed = reproducible
REFERENCE_DATE = date(2026, 9, 16)

# Bangalore neighbourhoods with centre coordinates. All sit inside the provider
# lat/lng envelope (12.84–13.00, 77.55–77.75), so every patient lands within
# range of at least a few providers.
BLR_AREAS = [
    ("Koramangala", 12.9352, 77.6245), ("Indiranagar", 12.9719, 77.6412),
    ("Whitefield", 12.9698, 77.7500), ("Jayanagar", 12.9250, 77.5938),
    ("HSR Layout", 12.9116, 77.6389), ("Marathahalli", 12.9591, 77.6974),
    ("Malleshwaram", 13.0035, 77.5647), ("BTM Layout", 12.9166, 77.6101),
    ("JP Nagar", 12.9063, 77.5857), ("Electronic City", 12.8452, 77.6602),
    ("Bellandur", 12.9260, 77.6762), ("Rajajinagar", 12.9915, 77.5551),
]

# Synthea encodes social determinants as "conditions" too. Drop the obvious
# non-clinical ones so a patient's history reads like a medical history, not a
# census form.
NON_CLINICAL = [
    "employment", "labor force", "education", "social isolation",
    "limited social contact", "social contact", "medication review",
    "stress (finding)", "victim", "violence", "reports of", "unemployed",
    "risk activity", "housing", "refugee", "criminal", "transport",
    "part-time", "full-time", "received", "misuses", "unhealthy alcohol",
    "lack of", "awaiting transplant",
]

HISTORY_CAP = 12
MEDS_CAP = 10


def clean_name(first, last):
    """Synthea appends digits to names ('Guillermo498'); strip them."""
    f = re.sub(r"\d+", "", str(first)).strip()
    l = re.sub(r"\d+", "", str(last)).strip()
    return f"{f} {l}".strip()


def clean_med(desc):
    """Drop the bracketed brand name, keep the drug + dose + form."""
    return re.sub(r"\s+", " ", re.sub(r"\[.*?\]", "", str(desc))).strip()


def age_from(birthdate):
    b = pd.to_datetime(birthdate).date()
    return REFERENCE_DATE.year - b.year - (
        (REFERENCE_DATE.month, REFERENCE_DATE.day) < (b.month, b.day)
    )


def is_clinical(desc):
    d = str(desc).lower()
    return not any(token in d for token in NON_CLINICAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=".",
                    help="folder with the Synthea CSVs")
    ap.add_argument("--output", default="patients.json")
    ap.add_argument(
        "--location", choices=["bangalore", "us"], default="bangalore")
    args = ap.parse_args()

    patients = pd.read_csv(f"{args.input}/patients.csv")
    conditions = pd.read_csv(f"{args.input}/conditions.csv")
    medications = pd.read_csv(f"{args.input}/medications.csv")
    try:
        allergies = pd.read_csv(f"{args.input}/allergies.csv")
    except FileNotFoundError:
        allergies = pd.DataFrame(columns=["PATIENT", "DESCRIPTION"])

    # Pre-group the child tables by patient for fast lookup.
    cond_by_pt = (conditions[conditions["DESCRIPTION"].map(is_clinical)]
                  .groupby("PATIENT")["DESCRIPTION"])
    # "current" medications = ones with no STOP date (still active).
    active_meds = medications[medications["STOP"].isna()]
    meds_by_pt = active_meds.groupby("PATIENT")["DESCRIPTION"]
    allergy_by_pt = allergies.groupby("PATIENT")["DESCRIPTION"] \
        if len(allergies) else None

    def uniq(series_group, key, transform=lambda x: x, cap=None):
        if series_group is None or key not in series_group.groups:
            return []
        seen, out = set(), []
        for v in series_group.get_group(key):
            t = transform(v)
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out[:cap] if cap else out

    out = {}
    for i, row in patients.reset_index(drop=True).iterrows():
        pid = f"P{i + 1:03d}"                  # clean, demo-friendly key
        # keep the UUID for provenance
        syn_id = row["Id"]

        if args.location == "bangalore":
            rng = random.Random(syn_id)           # deterministic per patient
            area, clat, clng = rng.choice(BLR_AREAS)
            lat = round(clat + rng.uniform(-0.01, 0.01), 6)   # ~1 km jitter
            lng = round(clng + rng.uniform(-0.01, 0.01), 6)
        else:
            area = f"{row['CITY']}, {row['STATE']}"
            lat, lng = round(row["LAT"], 6), round(row["LON"], 6)

        out[pid] = {
            "patient_id": pid,
            "synthea_id": syn_id,                 # data lineage back to the source
            "name": clean_name(row["FIRST"], row["LAST"]),
            "age": age_from(row["BIRTHDATE"]),
            "gender": row["GENDER"],
            "area": area,
            "lat": lat,
            "lng": lng,
            "history": uniq(cond_by_pt, syn_id, cap=HISTORY_CAP),
            "current_medications": uniq(meds_by_pt, syn_id, clean_med, MEDS_CAP),
            "allergies": uniq(allergy_by_pt, syn_id, cap=5),
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(out)} patients -> {args.output}")


if __name__ == "__main__":
    main()
