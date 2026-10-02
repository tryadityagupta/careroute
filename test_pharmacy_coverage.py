"""
test_pharmacy_coverage.py — the prod miss of 2026-10-02, offline.

A user in Panathur asked for the nearest pharmacy and got chemists 1.1+ km
away while one sat ~200 m from them. Causes, each pinned here:
  * chemists mapped as shop=medical_supply / shop=yes "X Medicals" were
    invisible to the pharmacy filter;
  * a long Indian postal address never geocoded (fallback cap dropped the
    road/area suffixes Nominatim can resolve).
No Overpass, no OSRM, no Nominatim.
"""

import os

os.environ.setdefault("OSRM_BASE_URL", "http://127.0.0.1:9")   # closed port

import osm      # noqa: E402
import tools    # noqa: E402

USER = (12.93212, 77.70465)          # the "give me my location" answer


def _el(lat, lon, **tags):
    return {"lat": lat, "lon": lon, "tags": tags}


FAKE_OSM = [
    _el(12.9420, 77.7080, name="Janani Clinic", healthcare="pharmacy"),
    _el(12.9330, 77.7170, name="Sowbhagya Medicals", shop="chemist"),
    _el(12.9330, 77.7068, name="Sri Sai Medicals", shop="medical_supply"),
    _el(12.9325, 77.7055, name="Ganesh Pharma & General", shop="convenience"),
    _el(12.9318, 77.7049, name="Manipal Medical Centre", amenity="clinic"),
    _el(12.9310, 77.7040, name="Cafe Medical Road", amenity="cafe"),
]


def _patch(monkeypatch):
    monkeypatch.setitem(osm._SOURCES, osm.OSM_SOURCE,
                        lambda lat, lng, r: (FAKE_OSM, None))


def test_medical_supply_and_named_shops_count_as_pharmacies(monkeypatch):
    _patch(monkeypatch)
    names = [p["name"] for p in
             osm.find_pharmacies(*USER, k=5)["pharmacies"]]
    assert "Sri Sai Medicals" in names
    assert "Ganesh Pharma & General" in names
    # A clinic called "... Medical Centre" is NOT a chemist.
    assert "Manipal Medical Centre" not in names
    assert names[0] in ("Ganesh Pharma & General", "Sri Sai Medicals")


def test_long_indian_address_reaches_road_and_area():
    q = tools._fallback_queries(
        "Chamunda medicals and general store, shanthi nivas, 68/2, "
        "1st cross road, Bhoganahalli Main Rd, Panathur, Bengaluru, "
        "Karnataka 560087")
    assert any(x.startswith("Bhoganahalli Main Rd") for x in q)
    assert any(x.startswith("Panathur") for x in q)
    assert not any("68/2" in x for x in q[1:])
    assert len(q) <= 6


def test_short_addresses_unchanged():
    assert tools._fallback_queries(
        "Prestige Shantiniketan, Whitefield, Bangalore") == [
        "Prestige Shantiniketan, Whitefield, Bangalore",
        "Whitefield, Bangalore", "Bangalore"]
