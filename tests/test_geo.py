"""
tests/test_geo.py — the self-hosted provider directory, end to end.

Runs the REAL infra/geo/import.sh (osm2pgsql + healthcare.lua + atomic schema
swap) on a hand-made fixture, then drives OsmProviderDirectory over
PostgisSource. No Overpass, OSRM or Nominatim: OSRM points at a closed port,
so distances fall back to straight-line, which is itself checked.

    GEO_TEST_DATABASE_URL=postgresql://careroute:careroute@localhost:5432/careroute \\
        python -m pytest tests/test_geo.py
Skips cleanly without osm2pgsql / psql / the env var.
"""

import os
import shutil
import subprocess

import httpx
import pytest

from careroute.config import PROJECT_ROOT
from careroute.domain.emergency import EmergencyNumberResolver, EmergencyService
from careroute.maps.postgis import GeoDatabase
from careroute.maps.routing import OsrmRouter
from careroute.providers.osm.directory import OsmProviderDirectory
from careroute.providers.osm.sources import PostgisSource
from tests.fakes import FakeGeocoder

DSN = os.getenv("GEO_TEST_DATABASE_URL", "").strip()
GEO = PROJECT_ROOT / "infra" / "geo"
FIXTURE = GEO / "fixtures" / "koramangala_sample.osm"
P001 = (12.9352, 77.6245)                                  # Koramangala

pytestmark = [
    pytest.mark.skipif(not DSN or not shutil.which("osm2pgsql") or not shutil.which("psql"),
                       reason="needs GEO_TEST_DATABASE_URL, osm2pgsql and psql"),
    pytest.mark.anyio,
]


def import_fixture(dsn: str) -> None:
    subprocess.run(["bash", str(GEO / "import.sh")], check=True, capture_output=True,
                   env={**os.environ, "GEO_DATABASE_URL": dsn, "PBF": str(FIXTURE),
                        "STYLE": str(GEO / "healthcare.lua")})


@pytest.fixture(scope="module")
def imported():
    """Import the fixture ONCE per module (sync: it's a subprocess)."""
    import_fixture(DSN)


@pytest.fixture
async def db(imported):
    """A fresh async pool per test: each test runs on its own event loop."""
    geo_db = GeoDatabase(DSN)
    yield geo_db
    await geo_db.aclose()


@pytest.fixture
async def osm(db):
    async with httpx.AsyncClient() as http:       # OSRM on a closed port: straight-line
        yield OsmProviderDirectory(PostgisSource(db), OsrmRouter("http://127.0.0.1:9",
                                                                 client=http))


def emergency_for(directory):
    return EmergencyService(EmergencyNumberResolver(FakeGeocoder(country="IN")), directory)


def _names(items):
    return [f["name"] for f in items]


async def test_import_keeps_named_healthcare_only(db):
    assert (await db.dataset_info())["rows"] == 7    # unnamed clinic and the cafe dropped


async def test_expression_index_matches_the_query(db):
    pool = await db.pool()
    async with pool.connection() as conn:
        await conn.execute("SET enable_seqscan = off")
        cur = await conn.execute("EXPLAIN " + db.NEARBY_SQL,
                                 {"lat": P001[0], "lng": P001[1], "radius_m": 8000})
        plan = "\n".join(r[0] for r in await cur.fetchall())
        await conn.execute("RESET enable_seqscan")
    assert "healthcare_geog_idx" in plan


async def test_radius_is_metres_and_excludes_far_points(db):
    near = await db.nearby_elements(*P001, 8000)
    names = {e["tags"]["name"] for e in near}
    assert "Mysuru City Hospital" not in names                 # ~130 km away
    assert {"Sakra World Hospital", "Heart Care Clinic",
            "St. John's Medical College Hospital"} <= names
    assert set(near[0]) == {"lat", "lon", "tags"}               # Overpass-shaped


async def test_specialty_via_tag_british_spelling(osm):
    out = await osm.find_providers("Orthopedics", *P001, k=3)
    assert isinstance(out, list) and _names(out) == ["Sparsh Care"]
    assert out[0]["matched_via"] == "speciality_tag"


async def test_closed_way_centroid_is_findable(osm):
    out = await osm.find_providers("Cardiology", *P001, k=3)
    assert _names(out) == ["Heart Care Clinic"]
    assert 12.944 < out[0]["lat"] < 12.946                       # centroid, not a corner
    assert out[0]["distance_type"] == "straight_line"           # OSRM is down


async def test_pharmacy_from_shop_tag(osm):
    out = await osm.find_pharmacies(*P001, k=3)
    assert _names(out["pharmacies"]) == ["Apollo Pharmacy"]


async def test_emergency_includes_multipolygon_hospital(osm):
    out = await emergency_for(osm).get_help(*P001)
    names = _names(out["nearest_hospitals"])
    assert out["emergency_number"].startswith("112")
    assert "St. John's Medical College Hospital" in names       # relation import
    assert "Sakra World Hospital" in names


async def test_directory_down_is_reported_not_disguised():
    """DB unreachable -> 'upstream_unavailable', never 'no specialists found';
    the emergency path still returns the number."""
    dead = GeoDatabase("postgresql://nobody@127.0.0.1:9/none")
    async with httpx.AsyncClient() as http:
        osm = OsmProviderDirectory(PostgisSource(dead),
                                   OsrmRouter("http://127.0.0.1:9", client=http))
        try:
            miss = await osm.find_providers("Cardiology", *P001)
            sos = await emergency_for(osm).get_help(*P001)
        finally:
            await dead.aclose()
    assert miss["error_type"] == "upstream_unavailable"
    assert sos["emergency_number"].startswith("112") and sos["nearest_hospitals"] == []
