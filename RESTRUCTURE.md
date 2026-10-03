# Restructure: flat modules → `careroute/` package (OOP)

## How to apply

Do it on a branch, in one commit, so `git log --follow` keeps history.

```bash
git checkout -b refactor/oop-package
git tag pre-oop                       # keeps the old two-engine layout linkable

# 1) Moves (content unchanged or comment-only, so git tracks them as renames)
mkdir -p web infra/nginx scripts
git mv index.html log_viewer.html web/
git mv deploy/nginx.conf infra/nginx/nginx.conf
git mv geo infra/geo
git mv synthea_to_patients.py checktags.py scripts/

# 2) Old modules, replaced by the package (see the map below)
git rm server.py agent_langgraph.py agent.py tools.py osm.py places.py \
       routing.py emergency.py geo_db.py sessions.py patient_store.py \
       shared_state.py security.py request_log.py data_source.py \
       checkpoints.py mock_llm.py mocktest.py \
       test_agent_langgraph.py test_geo.py test_mock_llm.py \
       test_pharmacy_coverage.py test_stateless.py

# 3) Copy in every new/updated file from this delivery, then:
python -m pytest                                   # memory mode
REDIS_URL=redis://localhost:6379/14 \
DATABASE_URL=postgresql://careroute:careroute@localhost:5432/careroute \
GEO_TEST_DATABASE_URL=postgresql://careroute:careroute@localhost:5432/careroute \
    python -m pytest -rs                           # shared mode + PostGIS + 3 replicas
git add -A && git commit -m "refactor: OOP package layout, DI container, no import-time side effects"
```

Running the app changes from `uvicorn server:app` to
`uvicorn --factory careroute.api.app:create_app`; the mock LLM from
`uvicorn mock_llm:app` to `uvicorn mock_llm.app:app`; migrations from
`python checkpoints.py migrate` to `python -m careroute.storage.checkpoints migrate`.
Dockerfile, docker-compose.yml and CI are already updated.

## Old → new map

| Old file | New home |
|---|---|
| `server.py` | `careroute/api/app.py` (routes, `/healthz`), `api/services.py` (`CareService`, `ChatService`, `TurnPromptBuilder`), `api/location.py` (`LocationResolver`), `api/schemas.py`, `api/errors.py` |
| `agent_langgraph.py` | `careroute/agent/graph.py` (`CareRouteAgent`), `prompts.py`, `state.py`, `guard.py` (`AnswerGuard`), `toolkit.py` (`LangChainToolkit`), `tracing.py` (`TurnTracer`), `messages.py`, `checkpointer.py` (`CheckpointerFactory`) |
| `agent.py`, `mocktest.py` | deleted — the reference loop lives at tag `pre-oop` |
| `tools.py` | `careroute/domain/tools.py` (`CareTools`), `domain/patients.py` (`PatientRepository`), `maps/geocoding.py` (`NominatimGeocoder`) |
| `osm.py` | `careroute/providers/osm/directory.py` (`OsmProviderDirectory`), `osm/sources.py` (`OverpassSource`, `PostgisSource`, `OverpassDiskCache`), `osm/matching.py` (`SpecialtyMatcher`, pharmacy and narrow-clinic rules) |
| `places.py` | `careroute/providers/google.py` (`GooglePlacesDirectory`) |
| — | `careroute/providers/base.py` (`ProviderDirectory` contract), `providers/dummy.py` (`JsonProviderDirectory`, was the dummy half of tools.py) |
| `routing.py` | `careroute/maps/routing.py` (`OsrmRouter`) |
| `geo_db.py` | `careroute/maps/postgis.py` (`GeoDatabase`) |
| 4 copies of `_haversine_km` | `careroute/maps/distance.py` (one copy) |
| `emergency.py` | `careroute/domain/emergency.py` (`EmergencyService`, `EmergencyNumberResolver`) |
| `sessions.py` | `careroute/storage/sessions.py` (`SessionStore` → Memory / Redis) |
| `patient_store.py` | `careroute/storage/patient_store.py` (`PatientStore` → Memory / Redis) |
| `shared_state.py` | `careroute/config.py` (`Settings`) + `storage/redis_client.py` (`RedisConnection`) |
| `checkpoints.py` | `careroute/storage/checkpoints.py` (`CheckpointAdmin`) |
| `data_source.py` | `careroute/storage/seed_data.py` (`SeedDataSource`) |
| `security.py` | `careroute/security/limits.py` (`RateLimiter`, `DailyCap` → Memory / Redis), `security/gate.py` (`RequestGate`) |
| `request_log.py` | `careroute/observability/interaction_log.py` (`InteractionLogger`) |
| `mock_llm.py` | `mock_llm/app.py` (`create_app`, `FaultInjector`, `MockConfig`), `mock_llm/policy.py` (`AgentPolicy`, `EchoPolicy`) |
| every `if __name__ == "__main__"` self-test | `tests/test_*.py` (pytest) |
| `test_agent_langgraph.py` | `tests/test_agent_graph.py` + `tests/test_api.py` |
| — | `careroute/container.py` (wires everything), `careroute/__main__.py` (CLI demo), `pyproject.toml` (pytest config) |

## Design rules the new layout follows

1. **Nothing happens at import.** Only `config.py` reads the environment;
   nothing connects to Redis, Postgres or the network until first use.
2. **Constructor injection.** Each class receives its collaborators;
   `container.py` decides which concrete ones. Tests pass fakes instead of
   monkeypatching module globals.
3. **Interfaces where there are real alternatives** — `ProviderDirectory`,
   `ElementSource`, `SessionStore`, `PatientStore`, `RateLimiter`, `DailyCap`.
   Pure helpers (`haversine_km`, matching rules) stay functions: wrapping
   stateless math in a class adds ceremony, not design.

## Behaviour changes (everything else is identical)

Verified by running the same 9 conversation turns against the old and the new
server with one mock model: all 9 answers and location payloads identical.

| Change | Why | Test |
|---|---|---|
| The answer guard now applies on **every** turn. A list passes if a tool confirmed something this turn, or if every numbered item names a provider confirmed earlier. | Before, one confirmed result anywhere in a conversation switched the guard off for all later turns (an invented list on turn 2 passed untouched). | `test_guard_still_works_on_later_turns`, `test_guard_unit_rules` |
| Patient tools are bound to the conversation's own patient. | The model chose `patient_id`, so "show me P002's record" returned another record. | `test_tools_refuse_other_patients_records`, `test_unbound_runs_keep_old_behaviour` |
| LLM timeout 30 s (was the SDK's 600 s); an LLM outage returns 503 with the emergency number instead of a bare 500. | Step 4's dependency-failure test. | `test_llm_outage_is_503_with_the_emergency_number` |
| Turn 1 skips the "does this thread exist?" Postgres read. | The server already knows it is turn 1. | covered by `test_stateless.py` |
| Geocode, route and country caches are bounded LRUs; the country cache no longer rewrites a JSON file during requests. HTTP clients reuse connections. | Unbounded memory and disk writes in the request path under load. | `test_agent_graph.py` geocode tests |
| Compose Postgres `max_connections=300`. | 2 pools × 10 per replica = 160 connections at 8 replicas; the default is 100. | — |
| Docker image runs as a non-root user and ships only what the service needs. | Hygiene. | — |
