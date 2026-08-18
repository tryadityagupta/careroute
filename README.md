# CareRoute — Agentic Provider-Matching Assistant

A small **agentic AI** demo: given a patient and their complaint, an LLM agent
reasons about the needed specialty, retrieves the patient's clinical record,
and finds the nearest matching healthcare providers by geographic proximity.

Built as a learning project to understand the agentic (tool-calling) pattern —
the same shape as a real care-coordination / referral system. It has two
interchangeable agent engines behind one interface: a hand-rolled loop and a
LangGraph port, with behavioural parity proven by an offline test harness.

## What it demonstrates
- **Agentic loop** (reason -> act -> observe -> repeat) — implemented twice:
  raw OpenAI tool-calling (`agent.py`) and a LangGraph `StateGraph`
  (`agent_langgraph.py`), same behaviour, one import swap apart
- **Tool calling / function calling** — the LLM chooses tools; our code executes them
- **Condition -> specialty reasoning** done by the model, not hardcoded
- **Model-driven recovery** — a specialty miss comes back as a structured
  result (`match_found: false` + a hint), and the agent retries with a larger
  search radius or a re-mapped specialty before falling back honestly
- **Deterministic output guard** — a code-level check withholds any provider
  list that no tool confirmed, and refuses to let an upstream outage be
  reported as "no specialists found"
- **Outage resilience** — three independently operated OpenStreetMap mirrors,
  a 7-day on-disk fetch cache (~1 km location cells), and stale-if-error
  serving, so a dead upstream degrades the demo instead of blanking it
- **Offline regression harness** — golden-transcript tests drive the graph
  with a scripted model: no API key, no network, deterministic
- **Proximity search** via the haversine formula (no paid geo API)
- **Clinical-record retrieval** as a tool (the "interchange of records" piece)
- **Guardrails**: a max-steps cap, per-tool error handling, and tool-side
  clamping of model-supplied arguments (search radius)

## Architecture
```
user request
   |
   v
agent engine ------------- gives tool descriptions to -----> gpt-4o-mini
  agent.py (raw loop)                                            |
  agent_langgraph.py (StateGraph)   <- server.py uses this       |
   ^                                                             |
   |  observe (tool result)                   act (tool call)    |
   |                                                             v
   +---------------------- tools.py (get_patient_record, find_providers,
   |                                 find_general_facilities)
   v                                          |
 guard: deterministic answer check            v
 before anything reaches the user     dummy JSON | OSM (+ mirrors + cache)
                                      | Google Places
```
On a specialty miss, `find_providers` returns `{"match_found": false, "hint": ...}`
instead of a silent nearest-anything list — the model reads the hint and loops
back with a larger `radius_m` (or a different specialty) before answering.

## The LangGraph engine

`agent_langgraph.py` is a behaviour-parity port of the hand-rolled loop. The
loop was already a graph; LangGraph just makes the shape explicit:

```
START -> agent --tool calls?--> tools --> agent    (reason-act-observe)
           |--no tool calls--> guard --> END       (deterministic answer check)
           |--budget spent---> stop  --> END       (the max-steps cap)
```

State carries the conversation plus everything the old loop kept in local
variables: `confirmed_providers` (names a successful `find_providers`
returned), `offered_facilities` (anything unverified), `tool_errors` (tools
that FAILED — not the same fact as tools that found nothing), and `llm_calls`
(the step budget, counted in model calls so the cap keeps its meaning).

The guard is the last line of defence, in code, because prompt instructions
alone did not hold: a recommendation list with no confirmed specialist gets
withheld, and an answer produced after a directory outage gets prefixed with
an explicit "this is NOT a confirmed result" warning.

Parity is pinned by `test_agent_langgraph.py`: a scripted stand-in model
replays fixed tool-call sequences ("golden transcripts"), so the tests verify
the graph's behaviour — tracking, guard, step cap, outage handling, tool
schemas — with no API key and no network.

`agent.py` is kept on purpose as the reference implementation: the pair plus
the harness documents the migration rather than hiding it.

## Setup (all backends)

Python 3.10+.

```bash
pip install -r requirements.txt
```

Versions are pinned: the LangChain ecosystem moves fast (the prebuilt
`create_react_agent` was deprecated in favour of `create_agent` within
months), so unpinned installs rot.

Create a `.env` file next to the code:

```
OPENAI_API_KEY=sk-...
# optional — pick a provider backend (see below):
# USE_REAL_PROVIDERS=osm
```

## Choosing a provider backend

`find_providers` has three interchangeable implementations behind one
signature. `USE_REAL_PROVIDERS` selects one **once, at import time** — set it
before starting, and restart the server after changing it. On startup the
terminal prints which one is live:

```
[tools] find_providers backend: openstreetmap
```

| Backend | `USE_REAL_PROVIDERS` | Data | Needs |
|---|---|---|---|
| Dummy JSON | *(unset)* | 10 mock Bengaluru providers | nothing (offline) |
| OpenStreetMap | `osm` | real facilities via Overpass API | internet for the first fetch per area; cached for 7 days after |
| Google Places | `google` | real providers via Places API | `GOOGLE_MAPS_API_KEY` + billing |

### 1) Dummy backend (default — works offline)

```bash
# CLI demo (current engine)
python agent_langgraph.py

# CLI demo (reference implementation)
python agent.py

# Web demo -> open http://localhost:8000
uvicorn server:app --reload
```

### 2) OpenStreetMap backend (real facilities, free, no key)

PowerShell (Windows):
```powershell
$env:USE_REAL_PROVIDERS="osm"; uvicorn server:app --reload
```

bash / zsh (macOS, Linux):
```bash
USE_REAL_PROVIDERS=osm uvicorn server:app --reload
```

Or simply add `USE_REAL_PROVIDERS=osm` to `.env` and run
`uvicorn server:app --reload` — `tools.py` loads `.env` before reading the flag.

The OSM backend rotates across three independently operated public Overpass
instances (FOSSGIS, VK Maps, Private.coffee) and caches every successful
fetch to `data/osm_cache.json` (gitignored). Running `python osm.py` once
while a mirror is up pre-warms the cache for the demo area, which makes the
web demo outage-proof for that area for a week.

### 3) Google Places backend (real providers, key + billing required)

Add to `.env`:
```
USE_REAL_PROVIDERS=google
GOOGLE_MAPS_API_KEY=AIza...
```
then:
```bash
uvicorn server:app --reload
```

PowerShell one-liner without `.env`:
```powershell
$env:USE_REAL_PROVIDERS="google"; $env:GOOGLE_MAPS_API_KEY="AIza..."; uvicorn server:app --reload
```

## Tests (no LLM key needed except where noted)

Run these with `USE_REAL_PROVIDERS` unset unless noted:

```bash
python test_agent_langgraph.py   # offline harness for the LangGraph engine:
                                 # scripted model, real dummy tools; verifies
                                 # guard, tracking, step cap, outage handling,
                                 # and tool-schema fidelity (6 tests)
python tools.py                  # deterministic tools: record lookup, haversine
                                 # ranking, and BOTH structured-miss paths
python osm.py                    # live Overpass query near Koramangala (needs
                                 # internet; also pre-warms the fetch cache)
python mocktest.py               # exercises the OLD loop with a scripted model
```

## What I'd do for production (talking points)
- **Persistence / multi-turn**: add a LangGraph checkpointer so conversations
  survive across requests (`thread_id` per user)
- **Cut LLM calls further**: a semantic symptom -> specialty cache (the
  provider-fetch side is already cached deterministically in `osm.py`)
- Real provider DB with a **geospatial index** (e.g. PostGIS) instead of a
  per-request scan
- Real clinical records over **FHIR / HL7** APIs instead of mock JSON
- **Eval + observability** (LangSmith/Langfuse) to trace each tool call and
  log token usage per step
- Return copies / immutable reads under concurrency; auth & PHI access controls

## Note on the fallback path

When no specialist matches, `find_providers` returns `match_found: false` with a
**count** of nearby general facilities — never their names. Offering
non-specialist options requires a deliberate call to `find_general_facilities`,
whose results are labelled `is_specialist_match: false`.

This is on purpose. An earlier version put the general facilities directly in
the miss payload and told the model in the system prompt not to present them as
specialists. It did anyway — a dental clinic was recommended for anxiety
attacks, with an invented justification. Withholding the names makes the failure
structural rather than advisory.

## Note on outages

During a real Overpass outage (August 2026), `find_providers` returned an
error; the model treated the error as a miss, escalated the search radius
three times (making the timeouts *more* likely), and told a cardiac patient
that no cardiologist was found — while Sri Jayadeva Institute of Cardiology
sat 3.3 km away. A failed lookup is not an empty lookup, and only the second
justifies that sentence.

The fix lives at two layers. Data layer: rotate across genuinely independent
mirrors (two of the original three turned out to be the same operator under
two names), cache successful fetches, serve stale data during a total outage,
and — if truly nothing is available — return a **typed** error
(`error_type: upstream_unavailable`) whose hint explicitly forbids the false
framing. Orchestration layer: the graph records `tool_errors` in state, and
the guard prefixes any post-outage answer with "this is NOT a confirmed
result". Same lesson as the fallback path: a prompt instruction is advisory;
a data structure is enforced.