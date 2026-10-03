"""
careroute/container.py — the composition root: the ONE place objects are built.

Every class in the package receives its collaborators through its constructor
(dependency injection); this file decides which concrete ones. Swapping Redis
for memory, Overpass for PostGIS or OpenAI for a scripted model is a decision
made here, from Settings — never by an `if` buried inside a module.

Everything is built LAZILY (cached_property) on first access, which gives two
properties tests rely on:
  * nothing connects to Redis/Postgres/the network just because you made a
    Container;
  * any part can be replaced before first use, e.g.
        c = Container(settings)
        c.geocoder = FakeGeocoder(...)        # before c.agent is touched
        c.agent.model = ScriptedModel([...])  # swap the model any time
"""

from __future__ import annotations

from functools import cached_property

import requests

from careroute.agent.checkpointer import CheckpointerFactory
from careroute.agent.graph import CareRouteAgent
from careroute.agent.toolkit import LangChainToolkit
from careroute.agent.tracing import TurnTracer
from careroute.api.location import LocationResolver
from careroute.api.services import CareService, ChatService
from careroute.config import Settings
from careroute.domain.emergency import EmergencyNumberResolver, EmergencyService
from careroute.domain.patients import PatientRepository
from careroute.domain.tools import CareTools
from careroute.maps.geocoding import NominatimGeocoder
from careroute.maps.postgis import GeoDatabase
from careroute.maps.routing import OsrmRouter
from careroute.observability.interaction_log import InteractionLogger
from careroute.providers.base import ProviderDirectory
from careroute.providers.dummy import JsonProviderDirectory
from careroute.providers.osm.directory import OsmProviderDirectory
from careroute.providers.osm.sources import (ElementSource, OverpassDiskCache,
                                             OverpassSource, PostgisSource)
from careroute.security.gate import RequestGate
from careroute.security.limits import (MemoryDailyCap, MemoryRateLimiter,
                                       RedisDailyCap, RedisRateLimiter)
from careroute.storage.patient_store import MemoryPatientStore, RedisPatientStore
from careroute.storage.redis_client import RedisConnection
from careroute.storage.seed_data import SeedDataSource
from careroute.storage.sessions import MemorySessionStore, RedisSessionStore


class Container:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.settings.validate()

    def describe(self) -> str:
        s = self.settings
        return (f"[careroute] backend: {self.directory.backend_name}"
                f" (osm source: {s.osm_source}) | sessions/patients/rate-limits: "
                f"{'redis' if s.use_redis else 'memory (single replica only)'} | "
                f"checkpoints: {'postgres' if s.use_postgres else 'memory (single replica only)'}"
                f"\n[careroute] {self.gate.describe()}")

    # --- infrastructure --------------------------------------------------------
    @cached_property
    def redis(self) -> RedisConnection | None:
        s = self.settings
        if not s.use_redis:
            return None
        return RedisConnection(s.redis_url, timeout_s=s.redis_timeout_s, key_prefix=s.key_prefix)

    @cached_property
    def http(self) -> requests.Session:
        """One pooled HTTP session for OSRM / Nominatim / Google (keep-alive)."""
        return requests.Session()

    @cached_property
    def geo_db(self) -> GeoDatabase:
        s = self.settings
        return GeoDatabase(s.geo_dsn, pool_max=s.geo_pool_max,
                           statement_timeout_ms=s.geo_statement_timeout_ms)

    @cached_property
    def checkpointers(self) -> CheckpointerFactory:
        s = self.settings
        return CheckpointerFactory(s.database_url, pool_max=s.pg_pool_max,
                                   auto_migrate=s.pg_auto_migrate)

    # --- storage -----------------------------------------------------------
    @cached_property
    def sessions(self):
        s = self.settings
        if self.redis:
            return RedisSessionStore(self.redis, ttl_s=s.session_ttl_s, turn_lock_s=s.turn_lock_s)
        return MemorySessionStore(ttl_s=s.session_ttl_s, turn_lock_s=s.turn_lock_s,
                                  max_sessions=s.max_sessions)

    @cached_property
    def patient_store(self):
        s = self.settings
        if self.redis:
            return RedisPatientStore(self.redis, ttl_s=s.session_ttl_s)
        return MemoryPatientStore(ttl_s=s.session_ttl_s)

    @cached_property
    def seed_data(self) -> SeedDataSource:
        s = self.settings
        return SeedDataSource(s.data_dir, blob_account_url=s.blob_account_url,
                              blob_container=s.blob_container)

    @cached_property
    def patients(self) -> PatientRepository:
        return PatientRepository(self.seed_data.patients(), self.patient_store)

    # --- maps and providers --------------------------------------------------
    @cached_property
    def geocoder(self) -> NominatimGeocoder:
        s = self.settings
        return NominatimGeocoder(s.nominatim_url, countries=s.geocode_countries, session=self.http)

    @cached_property
    def router(self) -> OsrmRouter:
        s = self.settings
        return OsrmRouter(s.osrm_base_url, timeout_s=s.osrm_timeout_s, session=self.http)

    @cached_property
    def osm_source(self) -> ElementSource:
        if self.settings.osm_source == "postgis":
            return PostgisSource(self.geo_db)
        return OverpassSource(OverpassDiskCache(self.settings.data_dir / "osm_cache.json"))

    @cached_property
    def osm_directory(self) -> OsmProviderDirectory:
        """Always built: the emergency service finds hospitals through OSM
        whichever backend serves specialist searches."""
        return OsmProviderDirectory(self.osm_source, self.router)

    @cached_property
    def directory(self) -> ProviderDirectory:
        backend = self.settings.provider_backend
        if backend == "osm":
            return self.osm_directory
        if backend == "google":
            from careroute.providers.google import GooglePlacesDirectory
            return GooglePlacesDirectory(self.settings.google_maps_api_key, session=self.http)
        return JsonProviderDirectory(self.seed_data.providers())

    @cached_property
    def emergency(self) -> EmergencyService:
        return EmergencyService(EmergencyNumberResolver(self.geocoder), self.osm_directory)

    # --- agent -------------------------------------------------------------
    @cached_property
    def care_tools(self) -> CareTools:
        return CareTools(self.patients, self.directory, self.geocoder, self.emergency)

    @cached_property
    def toolkit(self) -> LangChainToolkit:
        return LangChainToolkit(self.care_tools)

    @cached_property
    def model(self):
        """The real model. Tests replace this (or agent.model) with a script."""
        from langchain_openai import ChatOpenAI
        s = self.settings
        llm = ChatOpenAI(model=s.openai_model, timeout=s.llm_timeout_s,
                         max_retries=s.llm_max_retries)
        return llm.bind_tools(self.toolkit.tools)

    @cached_property
    def agent(self) -> CareRouteAgent:
        return CareRouteAgent(self.model, self.toolkit.tools, checkpointers=self.checkpointers,
                              tracer=TurnTracer(self.settings.log_coord_dp),
                              max_llm_calls=self.settings.max_llm_calls)

    # --- HTTP-facing ---------------------------------------------------------
    @cached_property
    def gate(self) -> RequestGate:
        s = self.settings
        if self.redis:
            limiter = RedisRateLimiter(self.redis, s.rate_per_min, s.burst)
            daily = RedisDailyCap(self.redis, s.daily_call_cap)
        else:
            limiter = MemoryRateLimiter(s.rate_per_min, s.burst)
            daily = MemoryDailyCap(s.daily_call_cap)
        return RequestGate(limiter, daily, api_keys=s.api_keys, trust_xff=s.trust_xff)

    @cached_property
    def interaction_log(self) -> InteractionLogger:
        s = self.settings
        return InteractionLogger(log_pii=s.log_pii, coord_dp=s.log_coord_dp, log_file=s.log_file)

    @cached_property
    def locations(self) -> LocationResolver:
        return LocationResolver(self.geocoder)

    def _service_args(self) -> dict:
        return dict(agent=self.agent, patients=self.patients, locations=self.locations,
                    log=self.interaction_log, backend_name=self.directory.backend_name)

    @cached_property
    def care_service(self) -> CareService:
        return CareService(**self._service_args())

    @cached_property
    def chat_service(self) -> ChatService:
        return ChatService(**self._service_args(), sessions=self.sessions)
