"""tests/test_dummy_directory.py — the offline directory and its two miss types."""

import pytest

from careroute.config import PROJECT_ROOT
from careroute.providers.dummy import JsonProviderDirectory
from careroute.storage.seed_data import SeedDataSource

seed = SeedDataSource(PROJECT_ROOT / "data")
d = JsonProviderDirectory(seed.providers())
P001 = seed.patients()["P001"]

pytestmark = pytest.mark.anyio


async def test_nearest_cardiologists_sorted():
    near = await d.find_providers("Cardiology", P001["lat"], P001["lng"], k=3)
    assert isinstance(near, list) and near
    assert [p["distance_km"] for p in near] == sorted(p["distance_km"] for p in near)


async def test_unknown_specialty_lists_what_exists():
    miss = await d.find_providers("Dentistry", P001["lat"], P001["lng"])
    assert miss["match_found"] is False and "Cardiology" in miss["available_specialties"]


async def test_radius_miss_says_widen():
    miss = await d.find_providers("Neurology", P001["lat"], P001["lng"], radius_m=2000)
    assert miss["match_found"] is False and "larger radius_m" in miss["hint"]


async def test_general_facilities_are_labelled():
    out = await d.find_general_facilities(P001["lat"], P001["lng"], k=2)
    assert all(f["is_specialist_match"] is False for f in out["facilities"])
