"""
geo_db.py — the provider directory as a PostGIS spatial query.

Replaces the Overpass HTTP call in osm.py when OSM_SOURCE=postgis. The data is
the same OpenStreetMap data, imported from a Geofabrik extract by
geo/import.sh, so this changes WHERE the data is served from, not what it is.

Why this matters for scale (and for the load test):
  * Overpass is a shared public service: 2-25 s per query, rate-limited, and
    hammering it in a load test breaks its usage policy. This query takes
    milliseconds and is ours to load-test.
  * It is stateless. osm.py's on-disk JSON cache was per-replica state; with
    PostGIS every replica reads the same table and needs no cache at all.

The rows come back in Overpass's element shape ({lat, lon, tags}), so all of
osm.py's filtering — specialty regexes, narrow-clinic filter, ER detection —
runs unchanged on either source.

Env:
  GEO_DATABASE_URL   defaults to DATABASE_URL. Separate so production can point
                     reads at a replica without touching the checkpointer.
  GEO_POOL_MAX       max pooled connections per replica (default 10).
"""

import os
import threading

from dotenv import load_dotenv

load_dotenv()  # self-sufficient: env is read at import time

_DSN = (os.getenv("GEO_DATABASE_URL")
        or os.getenv("DATABASE_URL") or "").strip()
_POOL_MAX = int(os.getenv("GEO_POOL_MAX", "10"))
# A spatial lookup should take milliseconds. If one doesn't, fail it rather
# than let it hold a pooled connection while users queue behind it.
_STATEMENT_TIMEOUT_MS = int(os.getenv("GEO_STATEMENT_TIMEOUT_MS", "2000"))

_pool = None
_pool_lock = threading.Lock()


class GeoUnavailable(Exception):
    """The provider directory could not be queried (DB down, table missing)."""


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                if not _DSN:
                    raise GeoUnavailable(
                        "OSM_SOURCE=postgis but neither GEO_DATABASE_URL nor "
                        "DATABASE_URL is set")
                from psycopg_pool import ConnectionPool
                _pool = ConnectionPool(
                    _DSN, min_size=1, max_size=_POOL_MAX, open=True,
                    kwargs={"autocommit": True,
                            "options": f"-c statement_timeout={_STATEMENT_TIMEOUT_MS}"},
                )
    return _pool


# ST_DWithin on geography = true metres on the spheroid, and it matches the
# (geom::geography) expression index created by geo/import.sh, so this is an
# index scan, not a table scan. Ordering is left to osm.py, which re-ranks by
# evidence quality and road distance anyway.
_NEARBY_SQL = """
SELECT ST_Y(geom) AS lat, ST_X(geom) AS lon, tags
FROM geo.healthcare
WHERE ST_DWithin(geom::geography,
                 ST_SetSRID(ST_MakePoint(%(lng)s, %(lat)s), 4326)::geography,
                 %(radius_m)s)
"""


def nearby_elements(lat: float, lng: float, radius_m: int) -> list[dict]:
    """Every named healthcare POI within radius_m metres, Overpass-shaped."""
    try:
        with _get_pool().connection(timeout=2) as conn:
            rows = conn.execute(_NEARBY_SQL, {"lat": lat, "lng": lng,
                                              "radius_m": radius_m}).fetchall()
    except GeoUnavailable:
        raise
    except Exception as e:     # psycopg errors, pool timeout, missing table
        raise GeoUnavailable(f"{type(e).__name__}: {str(e)[:120]}") from e
    return [{"lat": r[0], "lon": r[1], "tags": r[2]} for r in rows]


def dataset_info() -> dict:
    """Row count and data date of the imported directory, for /healthz.
    Raises GeoUnavailable if the import has never been run."""
    try:
        with _get_pool().connection(timeout=2) as conn:
            row = conn.execute(
                "SELECT rows, extract_date, imported_at FROM geo.dataset"
            ).fetchone()
    except GeoUnavailable:
        raise
    except Exception as e:
        raise GeoUnavailable(f"{type(e).__name__}: {str(e)[:120]}") from e
    if not row:
        raise GeoUnavailable("geo.dataset is empty")
    return {"rows": row[0], "extract_date": row[1],
            "imported_at": row[2].isoformat()}
