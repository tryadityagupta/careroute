"""
maps/distance.py — great-circle distance.

There used to be FOUR copies of this function (tools.py, osm.py, places.py,
routing.py). One now; everything imports it from here.
"""

from math import atan2, cos, radians, sin, sqrt

EARTH_RADIUS_KM = 6371


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Distance over the Earth's surface in km, rounded to 10 m.

    Cheap and offline, so it is used to pre-filter and shortlist; OSRM then
    upgrades the shortlist to real road distance.
    """
    d_lat = radians(lat2 - lat1)
    d_lng = radians(lng2 - lng1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lng / 2) ** 2
    return round(EARTH_RADIUS_KM * 2 * atan2(sqrt(a), sqrt(1 - a)), 2)
