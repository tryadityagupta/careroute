"""
tests/fakes.py — test doubles, injected through the Container.

Before the OOP refactor, tests swapped module globals (tools._nominatim,
osm._fetch_nearby, agent_langgraph.llm_with_tools ...) and had to set env
vars BEFORE importing anything. Now a test builds a Container and replaces
the parts it cares about. No network, no API key, no import-order tricks.
"""

from __future__ import annotations

import os
import uuid

from careroute.config import Settings
from careroute.container import Container
from careroute.maps.geocoding import NominatimGeocoder

os.environ.setdefault("OPENAI_API_KEY", "offline-test")   # never used

LOC = {"patient_lat": 12.9352, "patient_lng": 77.6245}    # P001, Koramangala


def make_settings(**overrides) -> Settings:
    """Real env (so CI's REDIS_URL/DATABASE_URL select shared mode), never the
    .env file, the offline dummy backend, and a private Redis namespace (Redis
    state outlives the process, so a previous run's empty rate-limit bucket
    would otherwise cause 429s)."""
    base = Settings.from_env(load_env_file=False).replace(
        provider_backend="", key_prefix=f"careroute-test-{uuid.uuid4().hex[:8]}",
        rate_per_min=1000, burst=1000,
        osrm_base_url="http://127.0.0.1:9")      # closed port: straight-line
    return base.replace(**overrides)


class ScriptedModel:
    """Duck-types a tool-bound chat model: .invoke() pops the next scripted
    AIMessage and records what the graph sent (so tests can read the prompt)."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.script.pop(0)


class FailingModel:
    """A model whose every call raises (e.g. openai.APIConnectionError)."""

    def __init__(self, exc: Exception):
        self.exc = exc

    def invoke(self, messages):
        raise self.exc


def call(name, args, i):
    return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}


class FakeGeocoder(NominatimGeocoder):
    """known: {query: (lat, lng, place_rank)}; anything else is not found."""

    def __init__(self, known: dict | None = None, country: str | None = "IN"):
        super().__init__("http://geocoder.invalid")
        self.known = known or {}
        self.country = country

    def _search(self, query):
        if query in self.known:
            lat, lng, rank = self.known[query]
            return [{"lat": str(lat), "lon": str(lng), "place_rank": rank,
                     "display_name": f"{query}, Bengaluru, Karnataka, India"}]
        return []

    def reverse_country(self, lat, lng):
        return self.country


class StaticSource:
    """An OSM ElementSource returning fixed elements (or a fixed error)."""
    name = "static"

    def __init__(self, elements=None, error=None):
        self._elements = elements or []
        self._error = error

    def elements(self, lat, lng, radius_m):
        return (None, self._error) if self._error else (self._elements, None)


def make_container(model=None, **settings_overrides) -> Container:
    c = Container(make_settings(**settings_overrides))
    c.geocoder = FakeGeocoder()
    c.osm_source = StaticSource()          # emergency hospital lookups: no network
    c.model = model or ScriptedModel([])
    return c
