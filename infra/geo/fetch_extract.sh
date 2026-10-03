#!/usr/bin/env bash
# infra/geo/fetch_extract.sh — download OSM data for Karnataka from Geofabrik.
#
#   docker compose --profile geo-setup run --rm geo-fetch
#
# Geofabrik has no per-state file for Indian states; the smallest region that
# covers Karnataka is "southern-zone" (~550 MB: KA, KL, TN, AP, TS, PY, LD).
# We download it once, verify the checksum, and clip it to a Karnataka bounding
# box with osmium. Everything downstream (PostGIS, OSRM, Nominatim) then
# processes one state instead of five, which is what keeps OSRM's preprocessing
# comfortable on a laptop.
#
# The box deliberately spills a little into Goa, Kerala, Tamil Nadu, Andhra and
# Maharashtra: a user near a state border should still see the hospital just
# across it.
set -euo pipefail

OUT_DIR="${OUT_DIR:-/data/osm}"
REGION_URL="${REGION_URL:-https://download.geofabrik.de/asia/india/southern-zone-latest.osm.pbf}"
# min_lon,min_lat,max_lon,max_lat
BBOX="${BBOX:-74.0,11.5,78.6,18.5}"
SRC="$OUT_DIR/southern-zone.osm.pbf"
DST="$OUT_DIR/karnataka.osm.pbf"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

if [ ! -f "$SRC" ] || [ "${FORCE:-0}" = "1" ]; then
    echo "[geo-fetch] downloading $REGION_URL"
    curl -fL --retry 3 -o "$SRC.part" "$REGION_URL"
    curl -fsSL -o "$SRC.md5" "$REGION_URL.md5"
    # Geofabrik's .md5 names the dated file; compare the hash only.
    expected="$(cut -d' ' -f1 "$SRC.md5")"
    actual="$(md5sum "$SRC.part" | cut -d' ' -f1)"
    [ "$expected" = "$actual" ] || { echo "[geo-fetch] md5 mismatch" >&2; exit 1; }
    mv "$SRC.part" "$SRC"
else
    echo "[geo-fetch] $SRC exists (FORCE=1 to re-download)"
fi

# Record the data date BEFORE clipping; it ends up in geo.dataset so you can
# always tell how old the provider directory is.
osmium fileinfo -g header.option.osmosis_replication_timestamp "$SRC" \
    > "${DST%.osm.pbf}.extract_date" 2>/dev/null || echo unknown > "${DST%.osm.pbf}.extract_date"

echo "[geo-fetch] clipping to bbox $BBOX"
# complete_ways: a road crossing the box edge is kept whole, so OSRM doesn't
# get dangling half-roads at the border.
osmium extract --bbox "$BBOX" --strategy complete_ways --overwrite \
    -o "$DST" "$SRC"

ls -lh "$SRC" "$DST"
echo "[geo-fetch] done: $DST (data as of $(cat "${DST%.osm.pbf}.extract_date"))"