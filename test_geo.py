"""
test_geo.py — the self-hosted provider directory, end to end.

Runs the REAL geo/import.sh (osm2pgsql + healthcare.lua + atomic schema swap)
on a hand-made fixture, then drives osm.find_providers & friends with
OSM_SOURCE=postgis. No Overpass, no OSRM, no Nominatim: OSRM is pointed at a
closed port so distances fall back to straight-line, which is itself checked.

Needs osm2pgsql, psql, and a PostGIS database the test may write to:
    GEO_TEST_DATABASE_URL=postgresql://careroute:careroute@localhost:5432/careroute \\
        python -m pytest -q test_geo.py
Skips cleanly when either is missing, so the default test run is unaffected.
"""

import os
import shutil
import subprocess

import pytest

DSN = os.getenv("GEO_TEST_DATABASE_URL", "").strip()
HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "geo", "fixtures", "koramangala_sample.osm")
P001 = (12.9352, 77.6245)                       # Koramangala

pytestmark = pytest.mark.skipif(
    not DSN or not shutil.which("osm2pgsql") or not shutil.which("psql"),
    reason="needs GEO_TEST_DATABASE_URL, osm2pgsql and psql")


@pytest.fixture(scope="module")
def geo():
    """Import the fixture, then point osm/geo_db/routing at it."""
    subprocess.run(
        ["bash", os.path.join(HERE, "geo", "import.sh")], check=True,
        env={**os.environ, "GEO_DATABASE_URL": DSN, "PBF": FIXTURE,
             "STYLE": os.path.join(HERE, "geo", "healthcare.lua")})

    import geo_db
    import osm
    import routing
    saved = (geo_db._DSN, geo_db._pool, osm.OSM_SOURCE, routing._OSRM_BASE)
    geo_db._DSN, geo_db._pool = DSN, None
    osm.OSM_SOURCE = "postgis"
    routing._OSRM_BASE = "http://127.0.0.1:9"     # closed port: OSRM "down"
    routing._ROUTE_CACHE.clear()
    yield osm, geo_db
    if geo_db._pool is not None:
        geo_db._pool.close()
    geo_db._DSN, geo_db._pool, osm.OSM_SOURCE, routing._OSRM_BASE = saved


def _names(items):
    return [f["name"] for f in items]


def test_import_keeps_named_healthcare_only(geo):
    _, geo_db = geo
    info = geo_db.dataset_info()
    # 7 named healthcare objects; the unnamed clinic and the cafe are dropped.
    assert info["rows"] == 7


def test_expression_index_matches_the_query(geo):
    """The app's ST_DWithin(geom::geography, ...) must be able to use the
    (geom::geography) index; a plain geometry index would not be used."""
    _, geo_db = geo
    with geo_db._get_pool().connection() as conn:
        conn.execute("SET enable_seqscan = off")
        plan = "\n".join(r[0] for r in conn.execute(
            "EXPLAIN " + geo_db._NEARBY_SQL,
            {"lat": P001[0], "lng": P001[1], "radius_m": 8000}).fetchall())
        conn.execute("RESET enable_seqscan")
    assert "healthcare_geog_idx" in plan


def test_radius_is_metres_and_excludes_far_points(geo):
    _, geo_db = geo
    near = geo_db.nearby_elements(*P001, 8000)
    names = {e["tags"]["name"] for e in near}
    assert "Mysuru City Hospital" not in names         # ~130 km away
    assert {"Sakra World Hospital", "Heart Care Clinic",
            "St. John's Medical College Hospital"} <= names
    # Elements are Overpass-shaped, so osm.py's parsing is source-agnostic.
    assert set(near[0]) == {"lat", "lon", "tags"}


def test_specialty_via_tag_british_spelling(geo):
    osm, _ = geo
    out = osm.find_providers("Orthopedics", *P001, k=3)
    assert isinstance(out, list) and _names(out) == ["Sparsh Care"]
    assert out[0]["matched_via"] == "speciality_tag"


def test_closed_way_centroid_is_findable(geo):
    osm, _ = geo
    out = osm.find_providers("Cardiology", *P001, k=3)
    assert _names(out) == ["Heart Care Clinic"]
    # the centroid, not a corner
    assert 12.944 < out[0]["lat"] < 12.946
    # OSRM is down, so the distance must be labelled straight-line, not road.
    assert out[0]["distance_type"] == "straight_line"


def test_pharmacy_from_shop_tag(geo):
    osm, _ = geo
    out = osm.find_pharmacies(*P001, k=3)
    assert _names(out["pharmacies"]) == ["Apollo Pharmacy"]


def test_emergency_includes_multipolygon_hospital(geo):
    import emergency
    real = emergency._country_code
    emergency._country_code = lambda lat, lng: "IN"
    try:
        out = emergency.get_emergency_help(*P001)
    finally:
        emergency._country_code = real
    names = _names(out["nearest_hospitals"])
    assert out["emergency_number"].startswith("112")
    assert "St. John's Medical College Hospital" in names  # relation import
    assert "Sakra World Hospital" in names


def test_directory_down_is_reported_not_disguised(geo):
    """DB unreachable -> 'upstream_unavailable', never 'no specialists found';
    and the emergency path still returns the number (step 4 will load-test
    this)."""
    osm, geo_db = geo
    import emergency
    saved = (geo_db._DSN, geo_db._pool)
    geo_db._DSN, geo_db._pool = "postgresql://nobody@127.0.0.1:9/none", None
    real_cc = emergency._country_code
    emergency._country_code = lambda lat, lng: "IN"
    try:
        miss = osm.find_providers("Cardiology", *P001)
        sos = emergency.get_emergency_help(*P001)
    finally:
        if geo_db._pool is not None:
            geo_db._pool.close()
        geo_db._DSN, geo_db._pool = saved
        emergency._country_code = real_cc
    assert miss["error_type"] == "upstream_unavailable"
    assert sos["emergency_number"].startswith("112")
    assert sos["nearest_hospitals"] == []
