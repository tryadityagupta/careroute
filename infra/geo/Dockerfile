# Tools image for the one-off geo setup jobs (geo-fetch, geo-import).
# Ubuntu 24.04 pins osm2pgsql 1.11 and osmium 1.16, the versions the import
# style and test_geo.py were verified against.
FROM ubuntu:24.04
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    osm2pgsql osmium-tool postgresql-client curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY . /geo
RUN chmod +x /geo/*.sh
WORKDIR /geo