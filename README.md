# CareRoute — Agentic Provider-Matching Assistant

A small **agentic AI** service: given a patient and their complaint, an LLM agent
reasons about the needed specialty, retrieves the patient's clinical record, and
finds the nearest matching healthcare providers by real driving distance. It
handles emergencies, pharmacy runs, and multi-turn follow-ups, and ships behind a
FastAPI web app with rate limiting, privacy-conscious logging, and a CI/CD
pipeline to Azure.

Built as a learning project to understand the agentic (tool-calling) pattern —
the same shape as a real care-coordination / referral system — it has since grown
into a small but complete service. Two interchangeable agent engines sit behind
one interface: a hand-rolled loop and a LangGraph port, with behavioural parity
proven by an offline test harness.

## What it demonstrates
- **Agentic loop** (reason -> act -> observe -> repeat) — implemented twice:
  a LangGraph `StateGraph` (`careroute/agent/graph.py`), ported from an
  earlier hand-rolled loop with behaviour parity pinned by golden-transcript tests
- **Tool calling / function calling** — the LLM chooses tools; our code executes
  them. Five tools: patient-record retrieval, specialist search, general-facility
  fallback, pharmacy lookup, and emergency triage
- **Condition -> specialty reasoning** done by the model, not hardcoded, steered
  toward the *least-specific* fitting specialty (sparse OSM tags punish
  over-specific choices)
- **Emergency triage first** — a suspected emergency short-circuits the specialty
  search: the agent returns the local emergency number and nearest ER, and that
  answer bypasses the specialist guard
- **Multi-turn conversations** — a LangGraph checkpointer keeps each conversation's
  thread and patient record alive across HTTP requests, so later turns refine the
  routing (adding meds, or raising a new need like "now I need painkillers")
- **Model-driven recovery** — a specialty miss comes back as a structured result
  (`match_found: false` + a hint), and the agent retries with a larger radius or a
  re-mapped specialty, or offers labelled non-specialist alternatives, before
  falling back honestly
- **Deterministic output guard** — a code-level check withholds any provider list
  that no tool confirmed, and refuses to let an upstream outage be reported as
  "no specialists found"
- **Real road distance + ETA via OSRM** — driving distance and duration that match
  what a user sees in Maps, with haversine kept as a cheap no-network pre-filter
  and graceful fallback (never fails the lookup)
- **Outage resilience** — three independently operated OpenStreetMap mirrors, a
  7-day on-disk fetch cache (~1 km location cells), and stale-if-error serving, so
  a dead upstream degrades the demo instead of blanking it
- **Offline regression harness** — golden-transcript tests drive the graph with a
  scripted model: no API key, no network, deterministic
- **Real patient records** — 108 Synthea-derived patients, loaded from local disk
  in dev or Azure Blob in production by the same code path
- **Production concerns** — per-IP rate limiting, optional API-key gate, a daily
  circuit breaker, privacy-conscious structured logging, and a
  test -> build -> deploy -> smoke-test CI/CD pipeline
- **Guardrails**: a max-steps cap, per-tool error handling, and tool-side clamping
  of model-supplied arguments (search radius)

## Project layout — where to go to change what

```
careroute/                 the application (one Python package)
  config.py                every setting / env var, read once into Settings
  container.py             builds and wires every object (composition root)
  __main__.py              CLI demo: python -m careroute "..."
  api/                     HTTP layer
    app.py                 create_app(): routes, CORS, /healthz, /version
    schemas.py             request bodies (CareRequest, ChatRequest)
    services.py            CareService, ChatService, TurnPromptBuilder
    location.py            LocationResolver (typed place / GPS -> coordinates)
    errors.py              store or LLM down -> 503 with the emergency number
  agent/                   the LangGraph agent
    graph.py               CareRouteAgent: nodes, routing, run_single / run_turn
    prompts.py             system prompt + emergency context
    state.py               CareRouteState and its per-turn resets
    guard.py               AnswerGuard: the deterministic answer check
    toolkit.py             LangChain tool adapters (schemas the model reads)
    tracing.py, messages.py  per-turn trace for logs; message helpers
    checkpointer.py        MemorySaver or Postgres checkpointer
  domain/                  business logic behind the tools
    tools.py               CareTools: the 7 tools the agent can call
    patients.py            PatientRepository: demo records + live records
    emergency.py           EmergencyService + EmergencyNumberResolver
  providers/               provider directories (one interface, three backends)
    base.py, dummy.py, google.py
    osm/                   directory.py, sources.py (Overpass | PostGIS), matching.py
  maps/                    distance, routing (OSRM), geocoding (Nominatim), postgis, cache
  storage/                 Redis connection, sessions, patient store, seed data,
                           checkpoints admin CLI (python -m careroute.storage.checkpoints)
  security/                limits.py (token bucket, daily cap), gate.py (FastAPI deps)
  observability/           interaction_log.py (privacy-conscious JSON lines)
mock_llm/                  fake OpenAI server for load tests (app.py, policy.py)
web/                       index.html (chat UI), log_viewer.html
infra/                     nginx/ (load balancer), geo/ (OSM import, OSRM prep)
scripts/                   one-off tools: synthea_to_patients.py, checktags.py
tests/                     every test, pytest
```

Rule of thumb: a **behaviour** change lives in `domain/`, `providers/` or
`agent/`; a **wiring** change (which implementation, which URL) lives in
`config.py` + `container.py`; nothing else reads environment variables.

## Architecture

```mermaid
flowchart TD
    B["Browser — web/index.html"]
    B -->|"POST /chat (multi-turn) · POST /care (single-shot)"| S

    subgraph FastAPI["FastAPI — careroute/api"]
        S["Request handler"]
        SEC["security/<br/>rate limit + optional API key"]
        SESS["storage/<br/>sessions + patient records"]
        S -.-> SEC
        S -.-> SESS
    end

    S --> AG

    subgraph Engine["Agent engine"]
        AG["agent/graph.py — CareRouteAgent (StateGraph)"]
    end

    AG <-->|"tool descriptions · act (tool call) · observe (result)"| LLM["gpt-4o-mini"]

    AG -->|tool call| T
    T -->|result| AG
    AG -->|no more tool calls| G["guard<br/>deterministic answer check"]
    G --> U["Answer to user"]

    subgraph Tools["domain/tools.py — CareTools"]
        T["get_patient_record · find_providers<br/>find_general_facilities · find_pharmacies<br/>get_emergency_help"]
    end

    T --> BK["Provider backend (swappable)<br/>dummy JSON · OSM (mirrors + cache) · Google Places"]
    T --> RT["maps/routing.py — OSRM road distance / ETA"]
    T --> EM["domain/emergency.py — local number + nearest ER"]
```

On a specialty miss the OSM backend's `find_providers` returns
`{"match_found": false, "hint": ..., "general_alternatives": [...]}` instead of a
silent nearest-anything list — the model reads the hint and either loops back with
a larger `radius_m` (or a different specialty), or offers the labelled
non-specialist alternatives, before answering.

## The agent engines

`careroute/agent/graph.py` is a behaviour-parity port of an earlier hand-rolled
loop (kept in git history at tag `pre-oop`). The loop was already a graph; LangGraph just makes the shape explicit:

```mermaid
flowchart LR
    START([START]) --> AG[agent]
    AG -->|"tool calls · budget left"| TL[tools]
    TL -->|"reason · act · observe"| AG
    AG -->|no tool calls| GU["guard<br/>deterministic answer check"]
    AG -->|budget spent| ST["stop<br/>max-steps cap"]
    GU --> E([END])
    ST --> E([END])
```

State carries the conversation plus everything the old loop kept in local
variables: `confirmed_providers` (names a successful `find_providers` returned),
`offered_facilities` (anything unverified), `tool_errors` (tools that FAILED — not
the same fact as tools that found nothing), `is_emergency` (an emergency answer
skips the specialist guard), and `llm_calls` (the step budget, counted in model
calls so the cap keeps its meaning).

The guard is the last line of defence, in code, because prompt instructions alone
did not hold: a recommendation list with no confirmed specialist gets withheld,
and an answer produced after a directory outage gets prefixed with an explicit
"this is NOT a confirmed result" warning. Emergency answers are exempt — they are
a call-for-help plus hospitals, not a specialist claim.

Parity is pinned by `tests/test_agent_graph.py`: a scripted stand-in model replays
fixed tool-call sequences ("golden transcripts"), so the tests verify the graph's
behaviour — tracking, guard, step cap, outage handling, tool schemas — with no API
key and no network.

## Multi-turn conversations

The web app talks to `/chat`, the multi-turn sibling of `/care`. The first turn
omits `session_id` and starts a conversation; the response returns one, which the
client sends back on every follow-up. Under the hood:

- The same graph is compiled with a **`MemorySaver` checkpointer**; the
  `session_id` *is* the LangGraph `thread_id`, so a later turn sees the whole
  transcript and the model has real context.
- A stable `patient_id` keeps the patient record alive in memory for the length of
  the session, so location and the accumulating medication/history list survive
  across turns instead of being rebuilt each request.
- **Per-turn vs per-conversation state**: `llm_calls`, `tool_errors`, and
  `is_emergency` reset each turn (they describe *this* message); `confirmed_providers`
  and `offered_facilities` persist, so re-mentioning an earlier tool-confirmed
  provider is not treated as a hallucination by the guard.
- `storage/sessions.py` sweeps idle sessions (30-minute TTL, hard cap of 5000) and frees
  their patient records.

Sessions, patient records and checkpoints live in shared stores when
configured, so any replica can serve any turn — see **Stateless replicas** below.

## Stateless replicas

Any request can land on any replica, and any replica can be killed at any
moment, without losing a conversation. Everything a later request depends on
lives outside the process:

| State | Single replica (no config) | Shared (`REDIS_URL` + `DATABASE_URL`) |
|---|---|---|
| Conversation transcript | LangGraph `MemorySaver` | Postgres checkpointer (`langgraph-checkpoint-postgres`) |
| Sessions + turn counter | `MemorySessionStore` | Redis hash with native TTL |
| Live patient records | dict | Redis JSON with native TTL (`storage/patient_store.py`) |
| Rate limit + daily cap | dict | Redis, atomic Lua token bucket on Redis's clock |

Design points:

- **Turn lock.** Statelessness creates a new race: two turns of one
  conversation (a double click, a retry) running at once on two replicas would
  both read the same checkpoint and lose a turn. A per-session Redis lock
  (`SET NX EX`, compare-and-delete release) makes turns of a conversation
  sequential; the second gets a 409. Different conversations stay fully
  parallel.
- **One checkpoint per turn.** Turns run with `durability="exit"`, writing one
  checkpoint when the turn ends instead of one per graph step (~5-10x fewer
  writes).
- **Failure modes are chosen, not accidental.** Redis or Postgres down → 503
  with an "if this is an emergency call 112" message; the rate limiter fails
  *open* (a Redis blip shouldn't take the service down); `/healthz` takes a
  replica that can't reach its stores out of rotation.
- **Fail fast on misconfiguration.** `CAREROUTE_REQUIRE_SHARED_STATE=1` makes a
  replica refuse to boot without Redis/Postgres, instead of silently running
  in memory mode and splitting conversations.
- **Retention.** Redis forgets idle sessions by TTL; `python -m
  careroute.storage.checkpoints prune` deletes idle conversations from Postgres in batches (run it on a
  schedule), so symptom/medication history isn't kept longer than needed.

**Proof:** `tests/test_stateless.py` boots real replica *processes* against real
Redis and Postgres with a mock LLM (`mock_llm/`, no API key), and CI runs it
before every deploy:

```
PASS  turn 1 on A, turn 2 on B: B saw history + meds -> 'turns_seen=2 | meds=ibuprofen, cetirizine'
PASS  A killed (SIGKILL); turn 3 on B, turn 4 on brand-new C -> 'turns_seen=4 | meds=ibuprofen, cetirizine'
PASS  10 requests alternating B/C: 5 allowed in total, then 429
PASS  two simultaneous turns on B and C -> [200, 409]; the next turn sees exactly 3 turns
```

On the previous in-memory design, turn 2 on another replica silently started
a **new** conversation (new session id, history and medications gone, no error
logged), so the failure would only have shown up as confused answers under load.

Run several replicas locally behind nginx:

```bash
docker compose up --build --scale app=3     # http://localhost:8000
```

## Emergency triage

`get_emergency_help` (in `careroute/domain/emergency.py`) is called first when the complaint looks
like an emergency — seizure, stroke signs, major trauma, heavy bleeding, chest pain
with cardiac features, fainting, or trouble breathing. It returns:

- **The local emergency number to dial now** — resolved from a cached Nominatim
  reverse-geocode, with a coarse offline bounding-box fallback and `112` (a
  GSM-standard number) as the final default. The number never depends on a network
  call succeeding.
- **The nearest hospitals** (which all have emergency departments), best-effort via
  the OSM fetch — layered on top, and omitted silently if Overpass is down.

The agent leads its answer with the number, and the emergency answer bypasses the
specialist guard.

## Distance and ETA

Ranking used to be straight-line (haversine) distance, which looked wrong next to
Maps (a clinic 170 m away showed as 3.78 km). `careroute/maps/routing.py` now upgrades results to
**real road distance and driving time via OSRM**:

- Haversine stays upstream as a cheap, no-network pre-filter to shortlist the
  nearest ~8 candidates — OSRM is never asked about far-away places.
- OSRM is called **once per query** via its Table service (one source -> many
  destinations), so a shortlist of 8 costs one HTTP request.
- **Graceful fallback**: if OSRM is unreachable, results keep the haversine distance
  and are labelled `distance_type: straight_line` rather than failing. Correctness
  of "is there a provider" never depends on OSRM — only the accuracy of the number.

Point `OSRM_BASE_URL` at a self-hosted OSRM or a paid routing provider for real
load; the public demo server has no SLA and is rate-limited.

## Patient data

`data/patients.json` holds **108 patients derived from Synthea** (synthetic FHIR
data), remapped to Bengaluru neighbourhoods so they fall within range of the demo
providers. Each record has demographics, coarse location, clinical history,
current medications, and allergies.

- `synthea_to_patients.py` is the converter (drops Synthea's non-clinical
  "conditions", keeps active meds, remaps Massachusetts coordinates to Bengaluru
  deterministically). Run it only to regenerate the data.
- `careroute/storage/seed_data.py` follows the 12-factor idea: **local `./data/*.json` in dev,
  Azure Blob in production**, chosen purely by env vars (`BLOB_ACCOUNT_URL`). The
  code never changes between laptop and prod; only configuration does. Nothing
  holds a password — Azure uses the Container App's managed identity, and locally
  `DefaultAzureCredential` falls back to your `az login` session.

## Setup (all backends)

Python 3.11 (the version used by the Docker image and CI; 3.10+ should work).

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in the values you need
```

Versions are pinned: the LangChain ecosystem moves fast (the prebuilt
`create_react_agent` was deprecated in favour of `create_agent` within months), so
unpinned installs rot.

`.env.example` documents every variable. The minimum to run the agent:

```
OPENAI_API_KEY=sk-...
# optional — pick a provider backend (see below):
# USE_REAL_PROVIDERS=osm
```

> **`.env` gotcha:** keep comments on their own lines. `python-dotenv` reads
> `KEY=            # note` as the literal value `# note`, not as empty — which once
> silently turned API-key auth on and caused 401s.

## Choosing a provider backend

`find_providers` (and its `find_general_facilities` / `find_pharmacies` siblings)
has three interchangeable implementations behind one signature. `USE_REAL_PROVIDERS`
selects one **once, at import time** — set it before starting, and restart the
server after changing it. On startup the terminal prints which one is live:

```
[tools] find_providers backend: openstreetmap
```

| Backend | `USE_REAL_PROVIDERS` | Data | Needs |
|---|---|---|---|
| Dummy JSON | *(unset)* | 10 mock Bengaluru providers | nothing (offline) |
| OpenStreetMap | `osm` | real facilities via Overpass API | internet for the first fetch per area; cached for 7 days after |
| Google Places | `google` | real providers via Places API | `GOOGLE_MAPS_API_KEY` + billing |

Only the OSM and Google backends do real road distance (OSRM), pharmacy lookup, and
the labelled general-alternatives fallback; the dummy backend is a simpler offline
stand-in for tests and quick demos.

### 1) Dummy backend (default — works offline)

```bash
# CLI demo
python -m careroute

# Web demo -> open http://localhost:8000
uvicorn --factory careroute.api.app:create_app --reload
```

### 2) OpenStreetMap backend (real facilities, free, no key)

PowerShell (Windows):
```powershell
$env:USE_REAL_PROVIDERS="osm"; uvicorn --factory careroute.api.app:create_app --reload
```

bash / zsh (macOS, Linux):
```bash
USE_REAL_PROVIDERS=osm uvicorn --factory careroute.api.app:create_app --reload
```

Or simply add `USE_REAL_PROVIDERS=osm` to `.env` and run
`uvicorn --factory careroute.api.app:create_app --reload` — `careroute/config.py` reads `.env` once, at startup.

The OSM backend rotates across three independently operated public Overpass
instances (FOSSGIS, VK Maps, Private.coffee) and caches every successful fetch to
`data/osm_cache.json` (gitignored). Running a lookup once while a mirror is
up pre-warms the cache for the demo area, which makes the web demo outage-proof for
that area for a week.

### 2b) Self-hosted map data (what Docker Compose runs, and what load tests need)

The public Overpass, OSRM and Nominatim servers are shared, best-effort
services; load-testing against them breaks their usage policies. So the
Compose stack hosts all three itself, from one Karnataka OpenStreetMap extract:

| Need | Public service (demo) | Self-hosted (Compose) |
|---|---|---|
| Find providers | Overpass mirrors, 2–25 s | PostGIS spatial query, ~50 ms at 8 km |
| Road distance / ETA | router.project-osrm.org | `osrm` container (MLD, car profile) |
| Place name → lat/lng, country | nominatim.openstreetmap.org (1 req/s) | `nominatim` container, unthrottled |

Same OSM data, same code path: `infra/geo/healthcare.lua` imports exactly the tags
the Overpass query asked for, and PostGIS rows come back in Overpass's element
shape, so every filter in `careroute/providers/osm/` runs unchanged. `OSM_SOURCE` picks the
source (`overpass` default, `postgis` in Compose).

One-time setup (Docker should have about 8 GB of RAM available):

```bash
docker compose --profile geo-setup run --rm geo-fetch    # ~550 MB download, clip to Karnataka
docker compose --profile geo-setup run --rm geo-import   # healthcare POIs -> PostGIS
docker compose --profile geo-setup run --rm osrm-prep    # build the routing graph
docker compose up --build --scale app=3                  # nominatim imports on first start
```

Nominatim's first start imports the extract into its own database, which takes
a while; it is persisted in the `nominatim-data` volume. Until it is ready,
geocoding by place name fails, and everything else works. `/healthz` reports
the directory's row count and data date under `info.provider_directory`.

Refreshing the data: re-run `geo-fetch` with `FORCE=1`, then `geo-import`.
The import builds schema `geo_import` and swaps it in for `geo` in one
transaction, so running replicas never see a half-built table.

### 3) Google Places backend (real providers, key + billing required)

Add to `.env`:
```
USE_REAL_PROVIDERS=google
GOOGLE_MAPS_API_KEY=AIza...
```
then:
```bash
uvicorn --factory careroute.api.app:create_app --reload
```

PowerShell one-liner without `.env`:
```powershell
$env:USE_REAL_PROVIDERS="google"; $env:GOOGLE_MAPS_API_KEY="AIza..."; uvicorn --factory careroute.api.app:create_app --reload
```

## Security and rate limiting

`/care` and `/chat` run the agent (several LLM calls plus real map lookups), so the
real exposure is **volume**. `careroute/security/` puts three independent, env-toggled
layers in front of them (the limiter classes are framework-agnostic; see `tests/test_security.py`):

| Layer | Env var | Default |
|---|---|---|
| Per-IP token-bucket rate limit | `CAREROUTE_RATE_PER_MIN`, `CAREROUTE_BURST` | 20/min, burst 5 (on) |
| Optional API-key gate (`X-API-Key`, constant-time compare) | `CAREROUTE_API_KEYS` | empty = open |
| Optional daily circuit breaker (global cap on `/care` calls) | `CAREROUTE_DAILY_CALL_CAP` | 0 = off |

Behind Azure Container Apps ingress the real client IP arrives in
`X-Forwarded-For`, trusted by default (`CAREROUTE_TRUST_XFF=1`). CORS is locked with
`CAREROUTE_ALLOWED_ORIGINS` (defaults to `*` for local dev). The hard budget stop is
still the monthly spend limit set in the OpenAI and Google Cloud billing consoles.

## Logging and privacy

`careroute/observability/interaction_log.py` emits one JSON line per interaction to stdout (and, if
`CAREROUTE_LOG_FILE` is set, to a file too), so on Azure the platform ships it to
Log Analytics with no extra infrastructure — query it with KQL.

This app handles symptoms, medications, and location, so logging is
privacy-conscious **by default**: it records the text and answer needed to analyse
usage, coarsens coordinates to ~1.1 km (2 dp), and **omits direct identifiers**
(name, exact coordinates). Set `CAREROUTE_LOG_PII=1` to also log name and exact
coordinates — and then treat the log store as sensitive.

## Tests (no LLM key needed except where noted)

### Mock LLM (load tests without an OpenAI bill)

`mock_llm/` serves `/v1/chat/completions`. Point `OPENAI_BASE_URL` at it and
the whole stack runs for real with only the model faked. In its default
`agent` mode it behaves like a careful tool-calling model: it reads the
transcript, calls `get_emergency_help` / `find_pharmacies` / `geocode_place` /
`find_providers` as the prompt asks, widens the radius on a miss (8 → 16 →
30 km), retries once on a directory error, and writes the final answer only
from tool results. A turn costs 2–4 model calls, like the real model. It keeps
no state, so any number of mock replicas answer identically.

| Endpoint | Purpose |
|---|---|
| `POST /admin/fault` | `{"mode": "down"\|"ratelimit"\|"slow"\|"flaky"\|"ok"}`, switched mid-run |
| `GET /admin/stats` | model calls, tool calls by name, prompt/completion tokens |
| `POST /admin/reset` | zero the counters, clear faults |

Latency is uniform between `MOCK_LLM_LATENCY_MIN_S` and `MOCK_LLM_LATENCY_MAX_S`
(default 0.8–2.0 s). `MOCK_LLM_MODE=echo` keeps the original
`turns_seen=N | meds=...` probe that `tests/test_stateless.py` relies on. Token counts
are estimates (about 4 characters per token), good enough for a cost model, not
for billing.

Every test lives in `tests/` and runs with pytest. Tests that need Redis,
Postgres, PostGIS or osm2pgsql skip cleanly when those are not configured.

```bash
python -m pytest                                  # offline: memory mode, dummy data

# Shared mode: also runs the Redis/Postgres store tests, the PostGIS import
# tests (needs osm2pgsql + psql), and the 3-replica statelessness proof.
REDIS_URL=redis://localhost:6379/14 \
DATABASE_URL=postgresql://careroute:careroute@localhost:5432/careroute \
GEO_TEST_DATABASE_URL=postgresql://careroute:careroute@localhost:5432/careroute \
    python -m pytest -rs
```

| File | What it pins down |
|---|---|
| `test_agent_graph.py` | graph with a scripted model: guard (incl. later turns), tracking, step cap, outage handling, emergency stickiness, patient binding, tool schemas |
| `test_api.py` | HTTP layer: location handling, record cleanup, 422 / 503 / 429 paths |
| `test_storage.py` | session + patient stores and the turn lock (memory and Redis) |
| `test_security.py` | rate limiter + daily cap (memory and Redis) |
| `test_dummy_directory.py` | offline directory and both structured-miss paths |
| `test_interaction_log.py` | PII omitted, coordinates coarsened |
| `test_geo.py` | the real `infra/geo/import.sh` on a fixture, then PostGIS lookups |
| `test_pharmacy_coverage.py` | chemist detection and long Indian addresses |
| `test_mock_llm.py` | mock model policy, OpenAI wire format, faults, full stack |
| `test_stateless.py` | real replica processes: replica hop, hard kill, shared rate limit, turn lock |

CI runs the suite in both modes before any build.

## Deployment

CareRoute runs as a single Docker container on **Azure Container Apps**. The
container serves the FastAPI app (`careroute.api.app:create_app`) with uvicorn on port 8000, and Azure
fronts it with public HTTPS ingress.

### Architecture

```mermaid
flowchart TD
    NET["Internet"] -->|HTTPS| FA

    subgraph Container["Azure Container Apps · CareRoute container"]
        direction TB
        FA["FastAPI<br/>serves web/index.html + /care + /chat"]
        LG["LangGraph agent"]
        TO["tools"]
        FA --> LG --> TO
    end

    TO --> OA["OpenAI"]
    TO --> OV["OpenStreetMap / Overpass"]
    TO --> OS["OSRM · road distance"]
```

Supporting resources:

| Resource | Purpose |
|---|---|
| Resource group | Groups everything below |
| Container registry (ACR, Basic) | Stores the container image |
| Container Apps environment | Hosts the app |
| Container App (0.5 CPU, 1 GiB) | The running service |
| Log Analytics workspace | Application and container logs |

Secrets and config on the Container App:

- **Secret** `OPENAI_API_KEY` — referenced by the container, never baked into the image
- **Env var** `USE_REAL_PROVIDERS=osm` — switches provider lookup from the dummy backend to live OpenStreetMap
- **Env var** `GIT_SHA` — set by CI to the deployed commit, surfaced at `/version`

### Prerequisites

- Docker
- Azure CLI, logged in (`az login`)
- An OpenAI API key

### Set your own values

The manual commands below use these variables. Fill them in with your own names.

```bash
RG="<resource-group>"     # a group you create for this app
LOCATION="<region>"       # e.g. southindia, eastus
ACR="<registry-name>"     # globally unique, letters and numbers only
ENVIRONMENT="<env-name>"  # Container Apps environment
APP="<app-name>"          # your Container App
IMAGE="careroute"         # image repo name inside the registry
```

### 1. Build and test locally

```bash
docker build -t careroute:local .
docker run --rm -p 8000:8000 --env-file .env careroute:local
# open http://localhost:8000
```

### 2. Create the resource group and registry

```bash
az group create --name "$RG" --location "$LOCATION"

az acr create --resource-group "$RG" --name "$ACR" --sku Basic
```

### 3. Build the image in ACR

```bash
az acr build --registry "$ACR" --image "$IMAGE:v1" .
```

### 4. Create the Container Apps environment

```bash
az containerapp env create \
  --name "$ENVIRONMENT" \
  --resource-group "$RG" \
  --location "$LOCATION"
```

### 5. Create the Container App

```bash
az containerapp create \
  --name "$APP" \
  --resource-group "$RG" \
  --environment "$ENVIRONMENT" \
  --image "$ACR.azurecr.io/$IMAGE:v1" \
  --registry-server "$ACR.azurecr.io" \
  --target-port 8000 \
  --ingress external \
  --cpu 0.5 --memory 1.0Gi \
  --secrets openai-api-key=<YOUR_OPENAI_KEY> \
  --env-vars OPENAI_API_KEY=secretref:openai-api-key USE_REAL_PROVIDERS=osm
```

> Pass `<YOUR_OPENAI_KEY>` at the command line — never commit the real key.
> Azure also needs pull access to the registry; the portal wires this up
> automatically, and via CLI you enable it with the ACR admin user or a managed identity.

### 6. Verify

```bash
curl https://<your-app-url>/version
# expect: {"version": "...", "commit": "<sha>", "backend": "openstreetmap"}
```

If `backend` comes back as `dummy-json`, the `USE_REAL_PROVIDERS=osm` env var is
missing from the active revision — set it and roll a new revision.

### Watching logs

```bash
az containerapp logs show --name "$APP" --resource-group "$RG" --follow
```

A healthy run shows `find_providers backend: openstreetmap`, the model calling
`get_patient_record(...)` and `find_providers(...)`, any structured-output retry
paths, and HTTP 200 responses.

## CI/CD

`.github/workflows/deploy.yml` runs on every push to `main` (and on demand). It is a
single job that gates on tests before it deploys, and verifies the live app after:

1. **Test gate** — set up Python 3.11, install deps, run `python -m pytest` in memory mode, then again
   against Redis + Postgres + PostGIS (including the 3-replica statelessness test).
   A failing test stops the pipeline before any build.
2. **Build** — log in to Azure via OIDC (no stored secrets), then `az acr build` the
   image tagged with both the commit SHA and `latest`.
3. **Deploy** — `az containerapp update` to the new image, passing `GIT_SHA` so the
   running app can report exactly which commit is live.
4. **Smoke test** — poll `/version` until it reports the SHA just deployed (guards
   against a stale image / unrolled revision), then `POST /care` and confirm an
   answer comes back, with generous timeouts and retries for cold starts.

Configure the workflow's Azure resource names at the top (`RESOURCE_GROUP`,
`ACR_NAME`, `CONTAINER_APP`) and provide `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, and
`AZURE_SUBSCRIPTION_ID` as repository secrets.

## Note on the fallback path

When no specialist matches, the design keeps non-specialist options **structurally
labelled** rather than merely asking the model to behave:

- The **dummy backend** returns `match_found: false` with a count and the available
  specialties — no facility names in the miss payload at all.
- The **OSM backend** goes further and surfaces `general_alternatives`: the nearest
  general facilities (with road distance/ETA), each carrying
  `is_specialist_match: false`, because users want nearby options immediately. The
  answer guard harvests those names into `offered` (unverified), so a bare list can
  never be presented as confirmed specialists.
- Cosmetic/single-specialty boutiques (skin, hair, laser, dental, eye, fertility…)
  are filtered out of *general* lists — never from specialist results, and never
  pharmacies — because a hair clinic is noise for an unrelated complaint.

This is on purpose. An earlier version put general facilities directly in the miss
payload and told the model in the system prompt not to present them as specialists.
It did anyway — a dental clinic was recommended for anxiety attacks, with an
invented justification. The lesson, repeated across the codebase: **a prompt
instruction is advisory; a data structure — and a code guard — is enforced.**

## Note on outages

During a real Overpass outage (August 2026), `find_providers` returned an error;
the model treated the error as a miss, escalated the search radius three times
(making the timeouts *more* likely), and told a cardiac patient that no cardiologist
was found — while Sri Jayadeva Institute of Cardiology sat 3.3 km away. A failed
lookup is not an empty lookup, and only the second justifies that sentence.

The fix lives at two layers. **Data layer**: rotate across genuinely independent
mirrors (two of the original three turned out to be the same operator under two
names), cache successful fetches, serve stale data during a total outage, and — if
truly nothing is available — return a **typed** error
(`error_type: upstream_unavailable`) whose hint explicitly forbids the false framing
and says to retry the same radius rather than escalate. **Orchestration layer**: the
graph records `tool_errors` in state, and the guard prefixes any post-outage answer
with "this is NOT a confirmed result".

## What I'd do next (talking points)

- **Async request path**: handlers and the graph are still synchronous, so one
  replica serves at most ~40 concurrent turns (the FastAPI threadpool). Move to
  `ainvoke`, `AsyncPostgresSaver`, `redis.asyncio` and `httpx.AsyncClient`;
  every I/O call already sits behind one class, so the change is contained.
- **Push specialty matching into PostGIS**: precompute specialty / pharmacy /
  hospital columns at import and use a KNN `ORDER BY geom <-> point LIMIT n`
  query, instead of filtering every row in the radius in Python.
- **Cut LLM calls**: a deterministic pre-pass (emergency detection, common
  complaint -> specialty mappings) plus a cache, measured in calls per turn.
- Real clinical records over **FHIR / HL7** instead of remapped Synthea JSON.
- **Eval + observability** (LangSmith/Langfuse) for per-step traces and tokens.
- Verify emergency numbers against an authoritative per-country source.
