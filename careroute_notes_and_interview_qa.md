# CareRoute — Notebook Notes & Interview Q&A

Notes for the agentic project, mapped to the role the director described
(agentic AI, provider matching by proximity & condition, clinical-record interchange).

---

## PART 1 — CONCEPT NOTES

### Agent vs RAG (know this cold — your #1 framing)
- **RAG** = retrieve context, generate **one** answer. "Look it up, then answer."
- **Agent** = reason in a **loop**, decide which **tools** to call, call them,
  observe results, repeat until the task is done. "Figure out the steps and take them."
- An agent often **uses RAG as one of its tools**. RAG is a building block; agentic is the orchestration.
- One-liner: *"RAG answers from retrieved text; an agent decides and acts in a loop using tools."*

### The agentic loop (this is what agent.py IS)
REASON -> ACT -> OBSERVE -> (repeat) -> ANSWER
- **Reason**: model decides what's needed next (e.g. "chest pain means Cardiology").
- **Act**: model emits a *tool call* — a name + JSON arguments.
- **Observe**: our code runs the tool, feeds the result back into the conversation.
- Loop continues until the model returns a normal message (no tool call) = final answer.
- Often called **ReAct** (Reasoning + Acting).

### Tool calling / function calling (the core mechanic)
- We send the model a list of **tool schemas** (name, description, parameter JSON schema).
- The model never runs code or sees the DB. It just *requests* a call: `find_providers(specialty=..., lat=..., lng=...)`.
- Our loop maps that name to a real Python function (via `TOOL_REGISTRY`), runs it, returns the result.
- **Tool descriptions are prompt engineering** — good descriptions = correct tool choice.

### Why each CareRoute piece exists (maps to her keywords)
- "agentic AI" -> the reason/act/observe loop in `agent.py`.
- "depending on condition" -> the model reasons condition -> specialty (not hardcoded).
- "fetch providers by proximity" -> `_haversine_km` ranks providers nearest-first.
- "interchange of clinical records" -> `get_patient_record` tool retrieves the record.

### Guardrails in an agent
- **max_steps cap**: stops an agent looping forever / burning tokens. (We set max_steps=5.)
- **Per-tool try/except**: a tool crash returns an error dict, not a dead agent.
- In production: also validate arguments, restrict which tools are allowed, log every call.

### Honest production upgrades (senior signal)
- **LangGraph** for explicit, stateful orchestration instead of a hand-rolled loop.
- **Geospatial index** (PostGIS) instead of scanning JSON; real provider DB.
- **FHIR / HL7** APIs for real clinical-record interchange (these are THE healthcare data standards).
- **Eval** (did it pick the right specialty? right providers?) + **observability** (LangSmith/Langfuse trace of each tool call).
- PHI access controls / auth — it's patient data.

---

## PART 2 — INTERVIEW Q&A (with follow-ups)

> Format: **Q** -> your answer -> **[follow-up]** the deeper probe to be ready for.

**Q: Walk me through what this does.**
A: Given a patient and a complaint, an LLM agent reasons which specialty is needed,
calls a tool to fetch the patient's clinical record for their location, calls a
second tool to find the nearest providers of that specialty by distance, and
returns a ranked recommendation. It's built on raw OpenAI tool-calling so the
reason-act-observe loop is explicit.
- **[follow-up] Why raw tool-calling and not LangChain/LangGraph?** So I could
  understand the loop mechanic directly; in production I'd use LangGraph for
  stateful orchestration.

**Q: What's the difference between this and your RAG chatbot?**
A: The RAG bot retrieves context and generates one answer. This *reasons in a loop*
and *acts* by calling tools, observing results between steps. An agent can even
use RAG as one of its tools — RAG is a building block, agentic is the orchestration.
- **[follow-up] When would you NOT use an agent?** When the task is single-step
  Q&A — an agent adds latency, cost, and failure modes you don't need. Use the
  simplest thing that works.

**Q: How does tool calling actually work under the hood?**
A: I send the model tool schemas — name, description, parameter JSON schema. The
model decides whether to call one and returns the name plus JSON arguments. My
loop maps the name to a real function via a registry, runs it, and feeds the
result back as a 'tool' message. The model never executes code itself.
- **[follow-up] How does the model know which tool to pick?** From the
  descriptions — they're effectively prompts. I wrote them to guide ordering
  ("call this first / after").
- **[follow-up] What if it calls a tool that doesn't exist or with bad args?**
  Registry lookup returns None -> I return an error dict; and each call is wrapped
  in try/except so a failure becomes an observation, not a crash.

**Q: How do you do the proximity / "nearest" part?**
A: The haversine formula — great-circle distance over the Earth's curvature from
the patient's lat/lng to each provider's, then sort ascending. No paid geo API.
- **[follow-up] Does this scale?** No — it's a linear scan over all providers per
  request. At scale I'd use a geospatial index (PostGIS, or a geohash/R-tree) so
  the nearest-neighbour lookup isn't O(n) every time.

**Q: How does the agent decide the specialty from the complaint?**
A: The model reasons it ("chest pain -> Cardiology") — it's not hardcoded. That's
the agentic part: the reasoning lives in the model, guided by the system prompt.
- **[follow-up] Isn't that risky in healthcare?** Yes — for real clinical use
  you'd constrain it to a validated mapping, add human review, and never let an
  LLM make an unguarded triage decision. This is a demo of the pattern, not a
  medical device.

**Q: What stops the agent from looping forever?**
A: A max_steps cap (5). If it hasn't produced a final answer by then, the loop
exits with a clear message. It also bounds token cost.
- **[follow-up] What would you add for robustness?** Argument validation, an
  allow-list of tools, ret/timeout on tool calls, and logging/tracing of every step.

**Q: How would you take this to production?**
A: LangGraph for orchestration, a real provider DB with a geospatial index, FHIR/HL7
for clinical records, guardrails + eval on specialty/result correctness, observability
to trace tool calls, and PHI access controls.
- **[follow-up] How would you evaluate it?** Two layers, like RAG: did it pick the
  right specialty (classification accuracy on labelled cases), and were the
  returned providers actually the nearest valid ones (deterministic check).

**Q: Where could this break?**
A: Wrong specialty reasoning; a patient/provider missing from data (handled with
error dicts); tool failure (caught); and the linear proximity scan at scale.
Being able to name your own failure modes is the point.

---

## PART 3 — THE 30-SECOND PITCH (rehearse out loud)
"After we spoke about agentic provider matching, I built a small version over the
weekend to understand the pattern. It's an LLM agent on raw tool-calling: it
reasons the specialty from the complaint, pulls the patient's record for their
location, and finds the nearest matching providers by haversine distance —
reason, act, observe, in a loop with a max-step guardrail. It's a demo of the
pattern, not production; for that I'd move to LangGraph, a geospatial DB, and
FHIR for records."