#!/bin/sh
# geo/osrm_prep.sh — build the OSRM routing graph for the clipped extract.
#
#   docker compose --profile geo-setup run --rm osrm-prep
#
# Uses the MLD pipeline (extract -> partition -> customize). CH would answer a
# little faster, but MLD preprocesses faster, needs less RAM, and is what you'd
# pick if you later wanted live traffic weights (re-run customize only).
# The car profile matches what routing.py asks for (/table/v1/driving).
set -eu

PBF="${PBF:-/data/osm/karnataka.osm.pbf}"
BASE="${PBF%.osm.pbf}.osrm"

[ -f "$PBF" ] || { echo "missing $PBF; run geo-fetch first" >&2; exit 1; }
if [ -f "$BASE.mldgr" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[osrm-prep] $BASE already built (FORCE=1 to rebuild)"
    exit 0
fi

echo "[osrm-prep] extract"
osrm-extract -p /opt/car.lua "$PBF"
echo "[osrm-prep] partition"
osrm-partition "$BASE"
echo "[osrm-prep] customize"
osrm-customize "$BASE"
echo "[osrm-prep] done: $BASE"