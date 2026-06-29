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
- **Proximity search** via the haversine formula (no paid geo API)
- **Clinical-record retrieval** as a tool (the "interchange of records" piece)
- A **max-steps guardrail** and per-tool **error handling**

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

## Run it
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # or put it in a .env and load it
python agent.py
```

Test the tools alone (no LLM, no key needed):
```bash
python tools.py
```

Test the agent LOOP without spending tokens (mocked model):
```bash
OPENAI_API_KEY=sk-mock python mock_test.py
```

## What I'd do for production (talking points)
- Orchestrate with **LangGraph** for explicit state/branching instead of a hand-rolled loop
- Real provider DB with a **geospatial index** (e.g. PostGIS) instead of a JSON scan
- Real clinical records over **FHIR / HL7** APIs instead of mock JSON
- **Guardrails + eval** (was the right specialty chosen? were results correct?) and **observability** (LangSmith/Langfuse) to trace each tool call
- Return copies / immutable reads under concurrency; auth & PHI access controls