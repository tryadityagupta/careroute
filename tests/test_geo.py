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

pytestmark = pytest.mark.skipif(
    not DSN or not shutil.which("osm2pgsql") or not shutil.which("psql"),
    reason="needs GEO_TEST_DATABASE_URL, osm2pgsql and psql")


def import_fixture(dsn: str) -> None:
    subprocess.run(["bash", str(GEO / "import.sh")], check=True, capture_output=True,
                   env={**os.environ, "GEO_DATABASE_URL": dsn, "PBF": str(FIXTURE),
                        "STYLE": str(GEO / "healthcare.lua")})


@pytest.fixture(scope="module")
def db():
    import_fixture(DSN)
    geo_db = GeoDatabase(DSN)
    yield geo_db
    geo_db.close()


@pytest.fixture(scope="module")
def osm(db):
    return OsmProviderDirectory(PostgisSource(db), OsrmRouter("http://127.0.0.1:9"))


def emergency_for(directory):
    return EmergencyService(EmergencyNumberResolver(FakeGeocoder(country="IN")), directory)


def _names(items):
    return [f["name"] for f in items]


def test_import_keeps_named_healthcare_only(db):
    assert db.dataset_info()["rows"] == 7     # unnamed clinic and the cafe dropped


def test_expression_index_matches_the_query(db):
    with db.pool().connection() as conn:
        conn.execute("SET enable_seqscan = off")
        plan = "\n".join(r[0] for r in conn.execute(
            "EXPLAIN " + db.NEARBY_SQL,
            {"lat": P001[0], "lng": P001[1], "radius_m": 8000}).fetchall())
        conn.execute("RESET enable_seqscan")
    assert "healthcare_geog_idx" in plan


def test_radius_is_metres_and_excludes_far_points(db):
    near = db.nearby_elements(*P001, 8000)
    names = {e["tags"]["name"] for e in near}
    assert "Mysuru City Hospital" not in names                 # ~130 km away
    assert {"Sakra World Hospital", "Heart Care Clinic",
            "St. John's Medical College Hospital"} <= names
    assert set(near[0]) == {"lat", "lon", "tags"}               # Overpass-shaped


def test_specialty_via_tag_british_spelling(osm):
    out = osm.find_providers("Orthopedics", *P001, k=3)
    assert isinstance(out, list) and _names(out) == ["Sparsh Care"]
    assert out[0]["matched_via"] == "speciality_tag"


def test_closed_way_centroid_is_findable(osm):
    out = osm.find_providers("Cardiology", *P001, k=3)
    assert _names(out) == ["Heart Care Clinic"]
    assert 12.944 < out[0]["lat"] < 12.946                       # centroid, not a corner
    assert out[0]["distance_type"] == "straight_line"           # OSRM is down


def test_pharmacy_from_shop_tag(osm):
    assert _names(osm.find_pharmacies(*P001, k=3)["pharmacies"]) == ["Apollo Pharmacy"]


def test_emergency_includes_multipolygon_hospital(osm):
    out = emergency_for(osm).get_help(*P001)
    names = _names(out["nearest_hospitals"])
    assert out["emergency_number"].startswith("112")
    assert "St. John's Medical College Hospital" in names       # relation import
    assert "Sakra World Hospital" in names


def test_directory_down_is_reported_not_disguised():
    """DB unreachable -> 'upstream_unavailable', never 'no specialists found';
    the emergency path still returns the number."""
    dead = GeoDatabase("postgresql://nobody@127.0.0.1:9/none")
    osm = OsmProviderDirectory(PostgisSource(dead), OsrmRouter("http://127.0.0.1:9"))
    try:
        miss = osm.find_providers("Cardiology", *P001)
        sos = emergency_for(osm).get_help(*P001)
    finally:
        dead.close()
    assert miss["error_type"] == "upstream_unavailable"
    assert sos["emergency_number"].startswith("112") and sos["nearest_hospitals"] == []
