"""
agent_langgraph.py — CareRoute's agent, ported from the hand-rolled loop to LangGraph.

The loop in agent.py was already a graph; it just hid its shape inside a
`for` loop and three local variables. This file makes the shape explicit:

        START
          |
          v
    +-> agent ---(tool calls, budget left)---> tools ---+
    |                                                   |
    +---------------------------------------------------+
          |
          |--(no tool calls)---> guard --> END    deterministic answer check
          |--(budget spent)----> stop  --> END    the old max_steps cap

Everything the old loop kept in local variables (`confirmed`, `offered`,
`step`) now lives in one typed state object that every node reads and writes.
That is the trade LangGraph offers: explicit state and explicit control flow —
and in exchange, every behaviour you care about must be wired in on purpose.
Nothing survives the migration just because the old loop had it.

Behaviour parity with agent.py, item by item:
    max_steps=8 model calls  ->  llm_calls counter + `stop` node
    confirmed/offered sets   ->  state keys, updated by the tools node
    _guard_answer rewrite    ->  `guard` node (same regex, same strings)
    step logging             ->  same prints, now emitted from inside nodes
    tool schemas             ->  the old TOOLS JSON descriptions, moved
                                 verbatim into docstrings + Annotated hints

server.py switches engines by changing one line:
    from agent_langgraph import run_agent
"""

import threading as _threading  # used only by the lazy initialiser below
import json
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import tools as careroute_tools

load_dotenv()

MAX_LLM_CALLS = 8  # same budget as agent.py's max_steps. The counter counts
# MODEL calls, not graph ticks, so the cap keeps its
# original meaning across engines.


# ---------------------------------------------------------------------------
# 1) STATE — the one object every node shares.
#
# `Annotated[list, add_messages]` attaches a *reducer* to the messages key: a
# rule for how a node's return value merges into existing state. add_messages
# appends new messages, replaces by matching id, and honours RemoveMessage
# deletions. The other keys have no reducer, so whatever a node returns
# REPLACES the old value — which is why the tools node merges the sets itself
# before returning them.
#
# (Sets are fine in memory. If you later add a checkpointer — LangGraph's
# persistence layer — and its serializer complains, switch to sorted lists.)
# ---------------------------------------------------------------------------
class CareRouteState(TypedDict):
    messages: Annotated[list, add_messages]
    confirmed_providers: set[str]  # names a SUCCESSFUL find_providers returned
    offered_facilities: set[str]   # names from fallback / unverified results
    tool_errors: list[str]         # tools that FAILED (not: found nothing)
    llm_calls: int                 # model calls so far (the old `step`)
    is_emergency: bool             # get_emergency_help fired -> skip specialist guard


SYSTEM_PROMPT = """
You are CareRoute, a clinical care-coordination assistant.
Given a patient and their complaint, your job is to recommend the nearest
appropriate healthcare providers.

Reason step by step:
0. FIRST, decide if this is a medical EMERGENCY — seizure, stroke signs (face
   droop, slurred speech, one-sided weakness), major trauma or a serious
   accident, heavy or uncontrolled bleeding, chest pain with cardiac features,
   fainting or unconsciousness, or trouble breathing. If it is, call
   get_emergency_help, and make your FIRST sentence tell the user to call the
   returned emergency number NOW (or go to the nearest emergency department).
   You may then list the nearest hospitals it returned. Do this before — or
   instead of — any specialist search; speed matters more than specialty here.
1. Decide which medical SPECIALTY the complaint requires (e.g. chest pain -> Cardiology).
   Choose the LEAST specific specialty that still fits. The provider directory is
   crowd-sourced (OpenStreetMap) and tags narrow specialties sparsely, so an
   over-specific choice (e.g. Podiatry for a toe splinter) often matches NOTHING
   anywhere. For minor or general complaints — small cuts, splinters, fever,
   general aches, minor bleeding — use "General Medicine", or go straight to
   find_general_facilities. Reserve narrow specialties for clearly specialist
   needs (Cardiology for chest pain, Dermatology for a rash).
2. Use get_patient_record to fetch the patient's location and history.
3. Use find_providers to get the nearest matching specialists.
4. Give a short, clear recommendation naming the providers and their distances,
   and briefly note any relevant item from the patient's history.
   If a provider matched only via its OSM speciality tag (matched_via =
   "speciality_tag"), say what kind of facility it actually is — e.g. "a
   multi-speciality clinic that lists psychiatry" — so the user can judge.

Location and details from the user's own words:
- The patient's stored coordinates come from the browser and may be wrong, or
  the user may name a DIFFERENT location (e.g. "she is in Guwahati, not
  Bangalore"). When the user names a place, call geocode_place to resolve it,
  run EVERY search (find_providers / find_general_facilities / find_pharmacies)
  with those coordinates, and call update_patient_record(lat, lng, area=<place>)
  so later turns stay there. NEVER assume the stored point is in the city the
  user named, and NEVER state a result is in a specific city or locality unless
  you geocoded it — give distances and at most "near <the place you searched>".
- If the user states the patient's NAME or MEDICATIONS in their message, call
  update_patient_record to save them onto the record.

Obtaining a medicine (not a diagnosis): if the user wants to BUY or pick up a
medicine or over-the-counter drug — painkillers, antacids, ORS, cold medicine,
etc. — the right provider is a PHARMACY. Call find_pharmacies (after
get_patient_record for the location) and list the nearest ones with distance.
You are ROUTING to a provider, not prescribing: never recommend a specific
medicine, dose, or brand, and never present a hospital or clinic as a pharmacy.
If find_pharmacies returns match_found=false, say no pharmacy was found in the
map data nearby.

Each message is its own request: in a conversation a new turn may add detail to
the earlier complaint OR raise a NEW need (e.g. "now I need painkillers"). Run
the search that fits THIS turn — do not just re-read the record and repeat the
previous list.


Recovering when find_providers returns match_found=false:
- The miss payload includes general_alternatives: the nearest GENERAL facilities,
  already labelled non-specialist, with drive distance/time. Present these to the
  user right away as convenient nearby options. You MAY ALSO call find_providers
  again with a larger radius_m (double it, up to 30000; at most twice) to look for
  the actual specialist further out, then let the USER choose between a nearby
  general facility and a farther specialist. Never describe general_alternatives
  as specialists.
- The specialty does not exist in the directory: pick the most clinically
  appropriate option from available_specialties and call find_providers again.
- If find_providers returns error_type=upstream_unavailable, the directory is
  unreachable: do NOT change the radius. You may retry the SAME call once; if it
  still fails, tell the user the directory is temporarily unreachable and to try
  again shortly, and direct them to emergency care for urgent symptoms.

Hard rule: a facility is a specialist match ONLY if find_providers returned it
in a success list. Never call anything else a specialist, and never invent a
clinical justification for a facility whose specialty you do not know.

Only use the tools provided. If a tool returns an error, explain the problem.
"""


# ---------------------------------------------------------------------------
# 2) TOOLS — thin adapters around tools.py.
#
# The domain logic stays framework-free in tools.py; these wrappers only
# translate it into LangChain's tool format. With @tool, the DOCSTRING is the
# description the model reads, and Annotated[...] metadata becomes the
# per-parameter descriptions — so the old TOOLS JSON schemas are reproduced
# here word for word. This matters: drop the "Call this FIRST" / "double
# radius_m on retry, up to 30000" guidance and the model gets measurably
# worse, because it acts on what it is told, not on what tools.py actually
# does. A one-line docstring is a silent behaviour regression.
# ---------------------------------------------------------------------------
@tool
def get_patient_record(
    patient_id: Annotated[str, "The patient's ID, e.g. 'P001'"],
) -> dict:
    """Retrieve a patient's clinical record (location + history) by patient
    ID. Call this FIRST to get the patient's coordinates before searching for
    providers."""
    return careroute_tools.get_patient_record(patient_id)


@tool
def geocode_place(
    place: Annotated[str, "A place name to resolve, e.g. 'Guwahati'"],
) -> dict:
    """Resolve a place NAME to coordinates. Call this whenever the user gives a
    location by name (e.g. "she is in Guwahati") rather than trusting the
    patient's stored coordinates. Feed the returned lat/lng into the search
    tools so the search happens THERE. Never guess coordinates, and never claim
    a result is in a city you did not resolve with this tool."""
    return careroute_tools.geocode_place(place)


@tool
def update_patient_record(
    patient_id: Annotated[str, "The patient's ID"],
    name: Annotated[str | None, "Patient name, if the user stated it"] = None,
    medications: Annotated[str | None,
                           "Comma-separated meds the user mentioned"] = None,
    lat: Annotated[float | None, "New latitude, from geocode_place"] = None,
    lng: Annotated[float | None, "New longitude, from geocode_place"] = None,
    area: Annotated[str | None,
                    "Human name of the location, e.g. 'Guwahati'"] = None,
) -> dict:
    """Persist details the user states onto the patient record: their NAME,
    MEDICATIONS, or a corrected LOCATION. For a named location, call
    geocode_place first, then pass its lat/lng here plus area=<place>. Only the
    fields you pass change."""
    return careroute_tools.update_patient_record(
        patient_id=patient_id, name=name, medications=medications,
        lat=lat, lng=lng, area=area,
    )


@tool
def find_providers(
    specialty: Annotated[str, "Medical specialty, e.g. 'Cardiology', 'Orthopedics'"],
    patient_lat: Annotated[float, "Patient latitude"],
    patient_lng: Annotated[float, "Patient longitude"],
    k: Annotated[int, "How many providers to return (default 3)"] = 3,
    radius_m: Annotated[int, (
        "Search radius in metres (default 8000). If no match was found, "
        "double it on retry, up to a maximum of 30000."
    )] = 8000,
) -> list | dict:
    """Find the k nearest providers of a given medical specialty to the
    patient's location. Call this AFTER you know the patient's coordinates
    and have decided the specialty the condition requires. If it returns
    match_found=false, follow the hint in the result: retry with a larger
    radius_m, or switch to one of the available_specialties."""
    return careroute_tools.find_providers(
        specialty=specialty,
        patient_lat=patient_lat,
        patient_lng=patient_lng,
        k=k,
        radius_m=radius_m,
    )


@tool
def find_general_facilities(
    patient_lat: Annotated[float, "Patient latitude"],
    patient_lng: Annotated[float, "Patient longitude"],
    k: Annotated[int, "How many facilities to return (default 3)"] = 3,
    radius_m: Annotated[int, "Search radius in metres (default 8000)"] = 8000,
) -> dict:
    """Nearest healthcare facilities of ANY type — these are NOT specialists.
    Only call this AFTER find_providers has returned match_found=false and
    you have exhausted your radius retries. Everything it returns must be
    presented as a general option, never as a specialist match."""
    return careroute_tools.find_general_facilities(
        patient_lat=patient_lat,
        patient_lng=patient_lng,
        k=k,
        radius_m=radius_m,
    )


@tool
def find_pharmacies(
    patient_lat: Annotated[float, "Patient latitude"],
    patient_lng: Annotated[float, "Patient longitude"],
    k: Annotated[int, "How many pharmacies to return (default 3)"] = 3,
    radius_m: Annotated[int, "Search radius in metres (default 8000)"] = 8000,
) -> dict:
    """Nearest PHARMACIES / chemists — where a user goes to OBTAIN medicines.
    Call this when the user wants to buy or pick up a medicine or OTC drug
    (painkillers, antacids, ORS, cold medicine, etc.). You are routing them to a
    provider, NOT prescribing: do not recommend a specific medicine, dose, or
    brand. Returns match_found=false when no pharmacy is mapped nearby — if so,
    say that plainly and never substitute a hospital or clinic for a pharmacy."""
    return careroute_tools.find_pharmacies(
        patient_lat=patient_lat,
        patient_lng=patient_lng,
        k=k,
        radius_m=radius_m,
    )


@tool
def get_emergency_help(
    patient_lat: Annotated[float, "Patient latitude"],
    patient_lng: Annotated[float, "Patient longitude"],
) -> dict:
    """Return the LOCAL emergency number to call NOW plus the nearest hospitals
    (which have emergency departments). Call this FIRST for any medical
    emergency — seizure, stroke signs, major trauma or a serious accident, heavy
    bleeding, chest pain with cardiac features, fainting, or trouble breathing —
    before any specialty search, and lead the answer with the number."""
    return careroute_tools.get_emergency_help(
        patient_lat=patient_lat,
        patient_lng=patient_lng,
    )


TOOLS = [get_patient_record, update_patient_record, geocode_place,
         find_providers, find_general_facilities, find_pharmacies,
         get_emergency_help]

llm = ChatOpenAI(model="gpt-4o-mini")  # no temperature set, matching agent.py.
llm_with_tools = llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# 3) OUTPUT GUARD + LOGGING HELPERS — copied verbatim from agent.py, on
# purpose. agent.py is scheduled for deletion once this port passes the
# regression harness; importing from it would keep it alive forever.
# Temporary duplication is the price of a clean cutover.
# ---------------------------------------------------------------------------
_LIST_ITEM = re.compile(r"^\s*\d+[.)]\s+\S", re.M)


def _summarize(result) -> str:
    """One-line summary of a tool result, for the step log."""
    if isinstance(result, list):
        def _one(r):
            n = str(r.get("name", "?"))[:40]
            via = r.get("matched_via")
            return f"{n} [{via}]" if via else n
        names = ", ".join(_one(r) for r in result[:5] if isinstance(r, dict))
        return f"OK list[{len(result)}]: {names}"
    if isinstance(result, dict):
        if "error" in result:
            return f"ERROR: {result['error']}"
        if result.get("match_found") is False:
            return f"MISS: {result.get('reason')}"
        if "facilities" in result:
            names = ", ".join(str(f.get("name", "?"))[:40]
                              for f in result["facilities"][:5])
            return f"FALLBACK[{len(result['facilities'])}]: {names}"
    return f"OK: {str(result)[:120]}"


def _names_in(result) -> set:
    """Facility names present in a tool result, whatever its shape."""
    items = []
    if isinstance(result, list):
        items = result
    elif isinstance(result, dict):
        # 'facilities' (find_general_facilities) and 'general_alternatives' (a
        # find_providers miss) both hold unverified names the answer may
        # mention — harvest both so the guard counts them as 'offered'.
        for _key in ("facilities", "general_alternatives", "nearest_hospitals"):
            if isinstance(result.get(_key), list):
                items += result[_key]
    return {i["name"] for i in items if isinstance(i, dict) and i.get("name")}


def _guard_answer(answer: str, confirmed: set, offered: set,
                  failed: list | tuple = ()) -> str:
    """Deterministic last line of defence.

    The model has twice presented facilities as specialists that no tool
    confirmed. Prompt rules did not hold and withholding names did not hold,
    so this check runs in code: if the answer lists providers but no tool
    ever returned a confirmed specialist match, the list cannot be trusted
    and is replaced. Code beats prompt.
    """
    if not answer or not _LIST_ITEM.search(answer):
        return answer                      # not a recommendation list
    if confirmed:
        return answer                      # a tool really did confirm matches
    if failed:
        # An empty `confirmed` set means one of two very different things:
        # nothing matched, or the lookup never ran. Only the first justifies
        # telling a patient that no specialist is nearby.
        return ("The provider directory could not be reached, so this is NOT a "
                "confirmed result \u2014 matching specialists may well exist nearby. "
                "Please try again shortly, and seek emergency care now if your "
                "symptoms are severe.\n\n" + answer)
    if offered:                            # only unlabelled fallbacks exist
        return ("No verified specialist match was found nearby for this "
                "condition.\n\n" + answer)
    return ("No verified specialist match was found nearby for this condition, "
            "so I can't recommend specific providers. Consider widening the "
            "search area, or seeing a general physician who can refer you.\n\n"
            "(The model attempted to list providers that no tool confirmed; "
            "that response was withheld.)")


def _payload(msg: ToolMessage):
    """ToolNode JSON-encodes dict/list tool results into the message content;
    decode them back so tracking sees the same shapes the old loop saw."""
    if not isinstance(msg.content, str):
        return msg.content
    try:
        return json.loads(msg.content)
    except (ValueError, TypeError):
        # e.g. ToolNode's own "Error: ..." string when a tool raised —
        # same shape the old loop used for failures.
        return {"error": msg.content}


def _text(message) -> str:
    """Message content as plain text (v1 models can return content blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


# ---------------------------------------------------------------------------
# 4) NODES — each is a plain function: full state in, partial update out.
# ---------------------------------------------------------------------------
def agent(state: CareRouteState):
    """REASON: ask the model what to do next.

    The system prompt is injected here rather than stored in state, so every
    caller (CLI, server.py, tests) gets it for free and can never forget it —
    and a prompt update never invalidates a saved conversation.
    """
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]

    response = llm_with_tools.invoke(messages)

    n = state.get("llm_calls", 0) + 1
    for call in response.tool_calls:
        print(f" [call {n}] model called: {call['name']}({call['args']})")
    return {"messages": [response], "llm_calls": n}


_tool_node = ToolNode(TOOLS)  # executes requested tools, catches exceptions,
# returns each result as a ToolMessage


def tools(state: CareRouteState):
    """ACT + OBSERVE: run the requested tools, then do the bookkeeping the
    old loop did inline — sort every returned facility name into `confirmed`
    (a success list from find_providers) or `offered` (anything else).
    The guard's verdict is only as good as this classification.
    """
    result = _tool_node.invoke(state)

    confirmed = set(state.get("confirmed_providers") or ())
    offered = set(state.get("offered_facilities") or ())
    errors = list(state.get("tool_errors") or ())
    is_emergency = bool(state.get("is_emergency"))
    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        payload = _payload(msg)
        print(f"          -> {_summarize(payload)}")
        if isinstance(payload, dict) and payload.get("emergency"):
            is_emergency = True               # emergency answers skip the guard
        if isinstance(payload, dict) and "error" in payload:
            errors.append(f"{msg.name}: {payload['error']}")
        elif msg.name == "find_providers" and isinstance(payload, list):
            confirmed |= _names_in(payload)   # tool-confirmed specialists
        else:
            offered |= _names_in(payload)     # everything else is unverified

    return {
        "messages": result["messages"],
        "confirmed_providers": confirmed,
        "offered_facilities": offered,
        "tool_errors": errors,
        "is_emergency": is_emergency,
    }


def guard(state: CareRouteState):
    """Deterministic answer check — the same logic as agent.py.

    If the answer must change, the model's message is REMOVED (add_messages
    honours RemoveMessage-by-id) and the guarded text is appended as the
    final AIMessage, so the transcript never keeps the untrusted version.
    """
    final = state["messages"][-1]
    answer = _text(final)
    # An emergency answer is a call-for-help + hospitals, not a specialist
    # recommendation, so the specialist guard must not rewrite it.
    if state.get("is_emergency"):
        return {}
    guarded = _guard_answer(
        answer,
        state.get("confirmed_providers") or set(),
        state.get("offered_facilities") or set(),
        state.get("tool_errors") or [],
    )
    if guarded == answer:
        return {}

    swap = [AIMessage(content=guarded)]
    if final.id:
        swap.insert(0, RemoveMessage(id=final.id))
    return {"messages": swap}


def stop(state: CareRouteState):
    """Budget exhausted mid-plan (the old `return "Stopped: ..."` branch).

    The dangling assistant message still REQUESTS tools; leaving it in place
    corrupts the transcript for any future turn (OpenAI rejects a tool call
    that has no tool result after it). Remove it and end with a plain
    statement instead.
    """
    dangling = state["messages"][-1]
    swap = [AIMessage(content=(
        "Stopped: reached the maximum number of steps without a final answer."
    ))]
    if dangling.id:
        swap.insert(0, RemoveMessage(id=dangling.id))
    return {"messages": swap}


# ---------------------------------------------------------------------------
# 5) CONTROL FLOW — the old loop's if/else, spelled out as edges.
#
# langgraph.prebuilt.tools_condition only knows two exits (tools / END).
# This loop has three, so the router is written by hand: four lines, and
# every exit is visible in the graph instead of implied by loop structure.
# ---------------------------------------------------------------------------
def route_after_agent(state: CareRouteState) -> str:
    last = state["messages"][-1]
    if not last.tool_calls:
        return "guard"                    # model is done -> validate answer
    if state["llm_calls"] >= MAX_LLM_CALLS:
        return "stop"                     # wants more tools, out of budget
    return "tools"                        # normal reason-act-observe step


builder = StateGraph(CareRouteState)
builder.add_node("agent", agent)
builder.add_node("tools", tools)
builder.add_node("guard", guard)
builder.add_node("stop", stop)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {"tools": "tools", "guard": "guard", "stop": "stop"},
)
builder.add_edge("tools", "agent")
builder.add_edge("guard", END)
builder.add_edge("stop", END)

app = builder.compile()


# ---------------------------------------------------------------------------
# 6) PUBLIC API — same signature as agent.py, so server.py migrates by
# changing one import. Keep both engines importable until the regression
# harness says their behaviour matches; then delete agent.py.
# ---------------------------------------------------------------------------
def run_agent(user_request: str) -> str:
    final_state = app.invoke({
        "messages": [HumanMessage(content=user_request)],
        "confirmed_providers": set(),
        "offered_facilities": set(),
        "tool_errors": [],
        "llm_calls": 0,
    })
    return _text(final_state["messages"][-1])


# ---------------------------------------------------------------------------
# 7) MULTI-TURN API — the SAME graph, compiled with a checkpointer so a
# conversation can span several HTTP requests. Everything above (app, run_agent,
# the state, the nodes) is left EXACTLY as it was, so the offline test harness
# and single-shot callers are untouched — this section is purely additive.
#
# HOW IT WORKS
#   * conversation_app is the identical graph compiled with a MemorySaver.
#   * Each conversation has a thread_id; the checkpointer stores that thread's
#     full state (transcript + tracking sets) between calls, so a later turn
#     sees the earlier turns and the model has real context.
#   * The agent node injects the system prompt fresh every turn and never saves
#     it into state, so a prompt change can't invalidate a live conversation.
#
# PER-TURN vs PER-CONVERSATION STATE (the one subtlety worth explaining):
#   * llm_calls, tool_errors, is_emergency are RESET at the start of each turn —
#     the step budget, "did a tool fail", and "is this an emergency" are about
#     THIS turn, not the whole conversation.
#   * confirmed_providers / offered_facilities are NOT reset — a provider name
#     backed by a real earlier tool call stays trusted, so re-mentioning it in a
#     later answer is not treated as a hallucination by the guard.
#
# SCALING CAVEAT (interview point): MemorySaver keeps threads in process memory,
# so conversations live on ONE replica and are lost on restart. For multi-replica
# or durable history, swap MemorySaver for a DB-backed saver (SqliteSaver /
# PostgresSaver) — same graph, different checkpointer.
# ---------------------------------------------------------------------------

_conversation_app = None
_conv_lock = _threading.Lock()


def _get_conversation_app():
    """Compile the graph WITH a checkpointer once, on first use.

    Lazy on purpose: importing this module (as the offline test harness does)
    must never depend on the checkpointer import resolving, so the deploy gate
    can't be broken by a langgraph whose import path differs. Double-checked
    locking keeps two concurrent first-callers from building two savers.
    """
    global _conversation_app
    if _conversation_app is None:
        with _conv_lock:
            if _conversation_app is None:
                try:
                    from langgraph.checkpoint.memory import MemorySaver
                except ImportError:  # older/newer layout
                    from langgraph.checkpoint import MemorySaver  # type: ignore
                _conversation_app = builder.compile(checkpointer=MemorySaver())
    return _conversation_app


def continue_conversation(user_message: str, thread_id: str) -> str:
    """Run ONE turn of a multi-turn conversation identified by thread_id.

    First turn for a thread: seed the full initial state. Later turns: pass only
    the new message plus the per-turn resets, so the checkpointer's accumulated
    state (transcript + confirmed providers) is preserved and built upon.
    """
    conv = _get_conversation_app()
    config = {"configurable": {"thread_id": thread_id}}

    # A brand-new thread has empty .values; an existing one has state to build on.
    try:
        existing = bool(conv.get_state(config).values)
    except Exception:
        existing = False

    turn = {
        "messages": [HumanMessage(content=user_message)],
        "llm_calls": 0,        # fresh step budget each turn
        "tool_errors": [],     # only this turn's tool failures matter
        "is_emergency": False,  # re-decide emergency per turn
    }
    if not existing:
        # New conversation — initialise the accumulating trackers too.
        turn["confirmed_providers"] = set()
        turn["offered_facilities"] = set()

    final_state = conv.invoke(turn, config=config)
    return _text(final_state["messages"][-1])


if __name__ == "__main__":
    request = "Patient P001 is experiencing chest pain. Find the nearest specialists."
    print(f"USER: {request}\n")
    answer = run_agent(request)
    print(f"\nCAREROUTE:\n{answer}")
