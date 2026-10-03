"""
maps/postgis.py — the provider directory as a PostGIS spatial query.

The data is OpenStreetMap, imported from a Geofabrik extract by
infra/geo/import.sh; this changes WHERE it is served from, not what it is.

  * Overpass: shared public service, 2-25 s per query, rate-limited, and
    hammering it in a load test breaks its usage policy.
  * PostGIS: an indexed query in milliseconds, ours to load-test, and
    stateless — every replica reads the same table, no per-replica cache.

Rows come back in Overpass's element shape ({lat, lon, tags}) so the OSM
directory's filtering runs unchanged on either source.

ASYNC: psycopg's AsyncConnectionPool. An async pool must be OPENED on the
running event loop, so it is opened lazily on first use (under an
asyncio.Lock) and closed by Container.aclose() at shutdown.
"""

from __future__ import annotations

import asyncio

from careroute.runtime import ensure_psycopg_compatible_loop


class GeoUnavailable(Exception):
    """The provider directory could not be queried (DB down, table missing)."""


class GeoDatabase:
    # ST_DWithin on geography = true metres, and it matches the
    # (geom::geography) expression index that import.sh creates, so this is
    # an index scan. Ordering is left to the directory (it re-ranks anyway).
    NEARBY_SQL = """
SELECT ST_Y(geom) AS lat, ST_X(geom) AS lon, tags
FROM geo.healthcare
WHERE ST_DWithin(geom::geography,
                 ST_SetSRID(ST_MakePoint(%(lng)s, %(lat)s), 4326)::geography,
                 %(radius_m)s)
"""

    def __init__(self, dsn: str, *, pool_max: int = 10,
                 statement_timeout_ms: int = 2000):
        self.dsn = dsn
        self.pool_max = pool_max
        # A spatial lookup should take milliseconds; if one doesn't, fail it
        # rather than hold a pooled connection while users queue behind it.
        self.statement_timeout_ms = statement_timeout_ms
        self._pool = None
        self._lock: asyncio.Lock | None = None

    async def pool(self):
        """The async connection pool, opened on first use."""
        if self._pool is None:
            if self._lock is None:
                self._lock = asyncio.Lock()
            async with self._lock:
                if self._pool is None:
                    if not self.dsn:
                        raise GeoUnavailable(
                            "OSM_SOURCE=postgis but neither GEO_DATABASE_URL "
                            "nor DATABASE_URL is set")
                    ensure_psycopg_compatible_loop()     # clear error on Windows
                    from psycopg_pool import AsyncConnectionPool
                    pool = AsyncConnectionPool(
                        self.dsn, min_size=1, max_size=self.pool_max, open=False,
                        kwargs={"autocommit": True,
                                "options": f"-c statement_timeout={self.statement_timeout_ms}"})
                    try:
                        await pool.open(wait=True, timeout=2)
                    except Exception:
                        await pool.close()
                        raise
                    self._pool = pool
        return self._pool

    async def nearby_elements(self, lat: float, lng: float, radius_m: int) -> list[dict]:
        """Every named healthcare POI within radius_m metres, Overpass-shaped."""
        try:
            pool = await self.pool()
            async with pool.connection(timeout=2) as conn:
                cur = await conn.execute(self.NEARBY_SQL, {
                    "lat": lat, "lng": lng, "radius_m": radius_m})
                rows = await cur.fetchall()
        except GeoUnavailable:
            raise
        except Exception as e:         # psycopg errors, pool timeout, missing table
            raise GeoUnavailable(f"{type(e).__name__}: {str(e)[:120]}") from e
        return [{"lat": r[0], "lon": r[1], "tags": r[2]} for r in rows]

    async def dataset_info(self) -> dict:
        """Row count and data date, for /healthz. Raises GeoUnavailable if the
        import has never been run."""
        try:
            pool = await self.pool()
            async with pool.connection(timeout=2) as conn:
                cur = await conn.execute(
                    "SELECT rows, extract_date, imported_at FROM geo.dataset")
                row = await cur.fetchone()
        except GeoUnavailable:
            raise
        except Exception as e:
            raise GeoUnavailable(f"{type(e).__name__}: {str(e)[:120]}") from e
        if not row:
            raise GeoUnavailable("geo.dataset is empty")
        return {"rows": row[0], "extract_date": row[1],
                "imported_at": row[2].isoformat()}

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
