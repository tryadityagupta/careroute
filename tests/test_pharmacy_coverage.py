"""
tests/test_pharmacy_coverage.py — the prod miss of 2026-10-02, offline.

A user in Panathur got chemists 1.1+ km away while one sat ~200 m from them:
  * chemists mapped as shop=medical_supply / shop=yes "X Medicals" were
    invisible to the pharmacy filter;
  * a long Indian postal address never geocoded.
"""

from careroute.maps.geocoding import NominatimGeocoder
from careroute.maps.routing import OsrmRouter
from careroute.providers.osm.directory import OsmProviderDirectory
from tests.fakes import StaticSource

USER = (12.93212, 77.70465)


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


def test_medical_supply_and_named_shops_count_as_pharmacies():
    osm = OsmProviderDirectory(StaticSource(FAKE_OSM), OsrmRouter("http://127.0.0.1:9"))
    names = [p["name"] for p in osm.find_pharmacies(*USER, k=5)["pharmacies"]]
    assert "Sri Sai Medicals" in names and "Ganesh Pharma & General" in names
    assert "Manipal Medical Centre" not in names          # a clinic is not a chemist
    assert names[0] in ("Ganesh Pharma & General", "Sri Sai Medicals")


def test_long_indian_address_reaches_road_and_area():
    q = NominatimGeocoder.fallback_queries(
        "Chamunda medicals and general store, shanthi nivas, 68/2, "
        "1st cross road, Bhoganahalli Main Rd, Panathur, Bengaluru, Karnataka 560087")
    assert any(x.startswith("Bhoganahalli Main Rd") for x in q)
    assert any(x.startswith("Panathur") for x in q)
    assert not any("68/2" in x for x in q[1:])
    assert len(q) <= 6


def test_short_addresses_unchanged():
    assert NominatimGeocoder.fallback_queries(
        "Prestige Shantiniketan, Whitefield, Bangalore") == [
        "Prestige Shantiniketan, Whitefield, Bangalore", "Whitefield, Bangalore", "Bangalore"]
