# CareRoute — Agentic Provider-Matching Assistant

A small **agentic AI** demo: given a patient and their complaint, an LLM agent
reasons about the needed specialty, retrieves the patient's clinical record,
and finds the nearest matching healthcare providers by geographic proximity.

Built as a weekend learning project to understand the agentic (tool-calling)
pattern — the same shape as a real care-coordination / referral system.

## What it demonstrates
- **Agentic loop** (reason -> act -> observe -> repeat) on raw OpenAI tool-calling
- **Tool calling / function calling** — the LLM chooses tools; our code executes them
- **Condition -> specialty reasoning** done by the model, not hardcoded
- **Model-driven recovery** — a specialty miss comes back as a structured
  result (`match_found: false` + a hint), and the agent retries with a larger
  search radius or a re-mapped specialty before falling back honestly
- **Proximity search** via the haversine formula (no paid geo API)
- **Clinical-record retrieval** as a tool (the "interchange of records" piece)
- **Guardrails**: a max-steps cap, per-tool error handling, and tool-side
  clamping of model-supplied arguments (search radius)

## Architecture
```
user request
   |
   v
agent loop (agent.py) ---- gives tool descriptions to ----> gpt-4o-mini
   ^                                                            |
   |  observe (tool result)                  act (tool call)    |
   |                                                            v
   +---------------------- tools.py (get_patient_record, find_providers)
                                   |
                                   v
                          data/  providers.json, patients.json
```
On a specialty miss, `find_providers` returns `{"match_found": false, "hint": ...}`
instead of a silent nearest-anything list — the model reads the hint and loops
back with a larger `radius_m` (or a different specialty) before answering.

## Setup (all backends)

Python 3.10+.

```bash
pip install -r requirements.txt
```

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
| OpenStreetMap | `osm` | real facilities via Overpass API | internet only |
| Google Places | `google` | real providers via Places API | `GOOGLE_MAPS_API_KEY` + billing |

### 1) Dummy backend (default — works offline)

```bash
# CLI demo
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

## Tests (no LLM, no OpenAI key)

Run these with `USE_REAL_PROVIDERS` unset unless noted:

```bash
python tools.py       # deterministic tools: record lookup, haversine ranking,
                      # and BOTH structured-miss paths (unknown specialty,
                      # radius too small)
python osm.py         # live Overpass query near Koramangala (needs internet)
python mocktest.py    # exercises the agent LOOP with a scripted fake model
```

## What I'd do for production (talking points)
- Orchestrate with **LangGraph** for explicit state/branching instead of a hand-rolled loop
- Real provider DB with a **geospatial index** (e.g. PostGIS) instead of a JSON scan
- Real clinical records over **FHIR / HL7** APIs instead of mock JSON
- **Guardrails + eval** (was the right specialty chosen? did the retry path fire
  when it should?) and **observability** (LangSmith/Langfuse) to trace each tool
  call and log token usage per step
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