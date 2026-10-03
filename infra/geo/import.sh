#!/usr/bin/env bash
# infra/geo/import.sh — load healthcare POIs from the clipped extract into PostGIS.
#
#   docker compose --profile geo-setup run --rm geo-import
#
# Zero-downtime refresh: osm2pgsql writes into schema geo_import, we index and
# ANALYZE it there, then swap it in for schema geo inside ONE transaction.
# Running replicas keep answering from the old table until the commit and from
# the new one right after; they never see an empty or half-indexed table.
# Re-run it monthly with a newer extract to refresh the directory.
set -euo pipefail

DSN="${GEO_DATABASE_URL:-${DATABASE_URL:?set DATABASE_URL or GEO_DATABASE_URL}}"
PBF="${PBF:-/data/osm/karnataka.osm.pbf}"
STYLE="${STYLE:-/geo/healthcare.lua}"
META="${PBF%.osm.pbf}.extract_date"

[ -f "$PBF" ] || { echo "missing $PBF; run geo-fetch first" >&2; exit 1; }
EXTRACT_DATE="$(cat "$META" 2>/dev/null || echo unknown)"

echo "[geo-import] preparing staging schema"
psql "$DSN" -v ON_ERROR_STOP=1 -q <<'SQL'
CREATE EXTENSION IF NOT EXISTS postgis;
DROP SCHEMA IF EXISTS geo_import CASCADE;
CREATE SCHEMA geo_import;
SQL

echo "[geo-import] osm2pgsql $(basename "$PBF") (extract date: $EXTRACT_DATE)"
# No --slim: the extract is state-sized, so node locations fit in RAM and the
# import is much faster. Nothing here is ever updated incrementally; a refresh
# is a full re-import plus swap.
GEO_IMPORT_SCHEMA=geo_import osm2pgsql --create --output=flex \
    --style="$STYLE" --database="$DSN" "$PBF"

echo "[geo-import] indexing, then swapping into schema geo"
psql "$DSN" -v ON_ERROR_STOP=1 -q -v extract_date="$EXTRACT_DATE" <<'SQL'
-- The app queries in metres with ST_DWithin(geom::geography, ...). An index
-- on the same EXPRESSION is what lets that use the index; a plain index on
-- geom (degrees) would not be used for a metre-radius search.
CREATE INDEX healthcare_geog_idx
    ON geo_import.healthcare USING gist ((geom::geography));
ANALYZE geo_import.healthcare;

CREATE TABLE geo_import.dataset AS
SELECT 'openstreetmap (Geofabrik southern-zone, clipped to Karnataka)'::text AS source,
       :'extract_date'::text AS extract_date,
       now() AS imported_at,
       (SELECT count(*) FROM geo_import.healthcare) AS rows;

BEGIN;
DROP SCHEMA IF EXISTS geo CASCADE;
ALTER SCHEMA geo_import RENAME TO geo;
COMMIT;
SQL

psql "$DSN" -At -c "SELECT 'rows=' || rows || ' extract_date=' || extract_date FROM geo.dataset"
echo "[geo-import] done"