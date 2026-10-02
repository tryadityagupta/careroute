-- geo/healthcare.lua — osm2pgsql flex style for CareRoute's provider directory.
--
-- Mirrors the Overpass query in osm.py tag-for-tag, so osm.py's filtering
-- (specialty matching, narrow-clinic filter, ER detection) sees the same data
-- whether it came from Overpass or from here:
--
--   amenity  in hospital|clinic|doctors|pharmacy
--   healthcare = *            (the newer tagging scheme)
--   shop     in chemist|pharmacy
--
-- Two deliberate differences from the Overpass query:
--   * Unnamed POIs are dropped at import. osm.py discarded them anyway ("a
--     place you cannot name cannot be recommended"), so storing them only
--     makes every spatial query read rows it will throw away.
--   * Multipolygon RELATIONS are included. Large hospital campuses are often
--     mapped as relations, which the Overpass query (node + way only) missed.
--
-- Every object becomes ONE point (ways/relations -> centroid), the same thing
-- Overpass's `out center` gave us. The full tag set is kept as jsonb so
-- osm.py can read healthcare:speciality, emergency, etc. exactly as before.
--
-- Run by geo/import.sh, which imports into schema geo_import and then swaps
-- it in atomically, so a refresh never leaves the app with a half-built table.

local SCHEMA = os.getenv("GEO_IMPORT_SCHEMA") or "geo_import"

local healthcare = osm2pgsql.define_table({
    name = "healthcare",
    schema = SCHEMA,
    ids = { type = "any", id_column = "osm_id", type_column = "osm_type" },
    columns = {
        { column = "name", type = "text", not_null = true },
        { column = "amenity", type = "text" },
        { column = "healthcare", type = "text" },
        { column = "shop", type = "text" },
        { column = "tags", type = "jsonb", not_null = true },
        { column = "geom", type = "point", projection = 4326, not_null = true },
    },
})

local AMENITY = { hospital = true, clinic = true, doctors = true, pharmacy = true }
local SHOP = { chemist = true, pharmacy = true }

local function is_healthcare(tags)
    return AMENITY[tags.amenity] or tags.healthcare ~= nil or SHOP[tags.shop]
end

local function insert(object, geom)
    local tags = object.tags
    if not tags.name or not is_healthcare(tags) or geom == nil then
        return
    end
    healthcare:insert({
        name = tags.name,
        amenity = tags.amenity,
        healthcare = tags.healthcare,
        shop = tags.shop,
        tags = tags,
        geom = geom,
    })
end

function osm2pgsql.process_node(object)
    insert(object, object:as_point())
end

function osm2pgsql.process_way(object)
    if object.is_closed then
        insert(object, object:as_polygon():centroid())
    else
        insert(object, object:as_linestring():centroid())
    end
end

function osm2pgsql.process_relation(object)
    if object.tags.type == "multipolygon" then
        insert(object, object:as_multipolygon():centroid())
    end
end