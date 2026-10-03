"""
tests/test_agent_graph.py — offline regression harness for the agent graph.

No API key, no network. A scripted model replays fixed tool-call sequences
("golden transcripts"), so these tests pin the GRAPH's behaviour — tracking,
guard, step cap, outage handling, tool schemas — independent of what a live
model would decide. The dummy JSON directory supplies REAL tool results.

Runs in memory mode by default; with REDIS_URL + DATABASE_URL set (CI does
both) the multi-turn tests go through Redis and the Postgres checkpointer.
"""

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from careroute.agent.guard import AnswerGuard
from careroute.providers.osm.directory import OsmProviderDirectory
from careroute.domain.emergency import EmergencyService
from tests.fakes import LOC, ScriptedModel, as_async, call, close_containers, make_container

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
async def _close_containers():
    yield
    await close_containers()      # pools close on the loop that opened them

REQ = "Patient P001 is experiencing chest pain. Find the nearest specialists."


async def run(script, container=None, patient_id=None):
    """Single-shot run; returns (final_state, model)."""
    c = container or make_container()
    model = ScriptedModel(script)
    c.agent.model = model
    from careroute.agent.state import fresh_conversation
    cfg = c.agent._config(patient_id)
    state = await c.agent.graph.ainvoke(
        {"messages": [HumanMessage(content=REQ)], **fresh_conversation()}, config=cfg)
    return state, model


# --- ported from test_agent_langgraph.py ------------------------------------
async def test_happy_path_answer_untouched():
    good = "Nearest cardiologists:\n1. Dr. A - 1.2 km\n2. Dr. B - 2.9 km"
    state, model = await run([
        AIMessage(content="", id="ai1",
                  tool_calls=[call("get_patient_record", {"patient_id": "P001"}, 1)]),
        AIMessage(content="", id="ai2",
                  tool_calls=[call("find_providers", {"specialty": "Cardiology", **LOC}, 2)]),
        AIMessage(content=good, id="ai3"),
    ])
    assert state["messages"][-1].content == good
    assert state["confirmed_providers"]
    assert state["llm_calls"] == 3
    sys_msgs = [x for x in model.seen[-1] if isinstance(x, SystemMessage)]
    assert len(sys_msgs) == 1                         # prompt injected exactly once


async def test_hallucinated_list_is_withheld():
    state, _ = await run([AIMessage(content="Here you go:\n1. Totally Real Cardiac Centre - 0.4 km",
                              id="bad")])
    final = state["messages"][-1].content
    assert "withheld" in final and "Totally Real" not in final
    assert "bad" not in {x.id for x in state["messages"]}   # untrusted msg removed


async def test_fallback_only_gets_disclaimer_prefix():
    state, _ = await run([
        AIMessage(content="", id="ai1",
                  tool_calls=[call("find_general_facilities", {**LOC, "k": 2}, 1)]),
        AIMessage(content="Options:\n1. Some General Hospital - 1.1 km", id="ai2"),
    ])
    assert state["messages"][-1].content.startswith("No verified specialist match")
    assert state["offered_facilities"] and not state["confirmed_providers"]


async def test_step_cap_stops_cleanly():
    c = make_container()
    n = c.agent.max_llm_calls
    loop = [AIMessage(content="", id=f"ai{i}",
                      tool_calls=[call("get_patient_record", {"patient_id": "P001"}, i)])
            for i in range(1, n + 1)]
    state, _ = await run(loop, c)
    assert state["messages"][-1].content.startswith("Stopped:")
    assert state["llm_calls"] == n
    assert not any(getattr(x, "tool_calls", None) and x.id == f"ai{n}"
                   for x in state["messages"])


async def test_upstream_outage_is_not_reported_as_no_specialists():
    """Replays the real 2026-08 run: Overpass 504s, and the model told a
    cardiac patient no cardiologist was found. A tool that FAILED is not a
    tool that found nothing — the guard must say so."""
    c = make_container()
    broken = {"error": "Provider directory unreachable: 504 Gateway Timeout",
              "error_type": "upstream_unavailable"}
    c.care_tools.find_providers = as_async(broken)
    state, _ = await run([
        AIMessage(content="", id="ai1",
                  tool_calls=[call("find_providers", {"specialty": "Cardiology", **LOC}, 1)]),
        AIMessage(content="I could not find any nearby cardiology specialists.\n"
                          "1. Some Clinic - 0.2 km", id="ai2"),
    ], c)
    final = state["messages"][-1].content
    assert state["tool_errors"] and not state["confirmed_providers"]
    assert final.startswith("The provider directory could not be reached")
    assert "NOT a confirmed result" in final


async def test_turn_trace_records_tools_without_pii():
    c = make_container()
    state, _ = await run([
        AIMessage(content="", id="ai1", tool_calls=[call(
            "update_patient_record", {"patient_id": "P001", "name": "Secret Name"}, 1)]),
        AIMessage(content="", id="ai2",
                  tool_calls=[call("find_providers", {"specialty": "Cardiology", **LOC}, 2)]),
        AIMessage(content="Nearest:\n1. Dr. A - 1.2 km", id="ai3"),
    ], c)
    trace = c.agent.tracer.trace(state["messages"])
    assert [t["tool"] for t in trace["tools"]] == ["update_patient_record", "find_providers"]
    assert "name" not in trace["tools"][0]["args"]          # PII dropped
    assert trace["tools"][1]["args"]["patient_lat"] == 12.94  # coords coarsened
    assert trace["tools"][1]["result"].startswith("OK list")
    assert trace["llm_calls"] == 3


async def test_emergency_sticks_across_turns():
    """Replays the real 2026-09-27 chat: chest pain -> 'does Maya do
    cardiology?' -> 'too far, what do I do?'. From the first emergency on,
    every later turn carries the emergency context AND the number."""
    c = make_container()
    thread = "t-" + uuid.uuid4().hex
    model = ScriptedModel([
        AIMessage(content="", id="e1", tool_calls=[call("get_emergency_help", LOC, 1)]),
        AIMessage(content="Call 112 (ambulance: 108) now.", id="e2"),
        AIMessage(content="Maya Hospital is not listed for cardiology.", id="e3"),
        AIMessage(content="Nearby options:\n1. Sai Clinic - 0.66 km", id="e4"),
    ])
    c.agent.model = model
    a1, _ = await c.agent.run_turn("chest pain", thread_id=thread)
    a2, _ = await c.agent.run_turn("does Maya do cardiology?", thread_id=thread)
    a3, t3 = await c.agent.run_turn("too far, what do I do?", thread_id=thread)
    assert a1 == "Call 112 (ambulance: 108) now."           # already has it: untouched
    for later in (a2, a3):
        assert later.startswith("If the symptoms you described earlier") and "112" in later
    sys3 = [x for x in model.seen[-1] if isinstance(x, SystemMessage)][0].content
    assert "EMERGENCY CONTEXT" in sys3
    assert t3["tools"] == []


async def test_emergency_list_skips_single_doctor_and_narrow_hospitals():
    facilities = [
        {"name": "Maya Hospital", "facility": "hospital", "_er": ""},
        {"name": "Dr Ramesh Dalwai Spine Surgeon", "facility": "hospital", "_er": ""},
        {"name": "City Eye Hospital", "facility": "hospital", "_er": ""},
        {"name": "Rest Home Hospital", "facility": "hospital", "_er": "no"},
        {"name": "Sai Clinic", "facility": "clinic", "_er": ""},
        {"name": "Sakra World Hospital", "facility": "hospital", "_er": "yes"},
    ]
    names = [f["name"] for f in EmergencyService.er_candidates(facilities)]
    assert names == ["Maya Hospital", "Sakra World Hospital"]


async def test_pharmacy_list_is_not_withheld():
    c = make_container()
    c.care_tools.find_pharmacies = as_async({
        "disclaimer": "Nearby pharmacies/chemists for obtaining medicines.",
        "pharmacies": [{"name": "Apollo Pharmacy", "distance_km": 0.1},
                       {"name": "MedPlus", "distance_km": 0.6}]})
    good = "Nearest pharmacies:\n1. Apollo Pharmacy - 0.1 km\n2. MedPlus - 0.6 km"
    state, _ = await run([AIMessage(content="", id="p1", tool_calls=[call("find_pharmacies", LOC, 1)]),
                    AIMessage(content=good, id="p2")], c)
    assert state["messages"][-1].content == good
    assert {"Apollo Pharmacy", "MedPlus"} <= state["confirmed_providers"]


async def test_geocode_falls_back_from_apartment_to_area_but_not_city():
    from tests.fakes import FakeGeocoder
    g = FakeGeocoder({"Whitefield, Bangalore": (12.97, 77.75, 19),   # suburb: fine
                      "Bangalore": (12.97, 77.59, 16)})              # whole city
    r = await g.geocode("Prestige Shantiniketan, Whitefield, Bangalore")
    assert r["matched_query"] == "Whitefield, Bangalore" and r["approximate"]
    assert (await g.geocode("Nowhere Towers, Unknownpura, Bangalore")).get("match_found") is False
    r3 = await g.geocode("Bangalore")
    assert r3["broad"] and not r3["approximate"]


async def test_tool_schema_kept_the_old_guidance():
    c = make_container()
    fp = convert_to_openai_tool(c.toolkit.by_name("find_providers"))["function"]
    assert "match_found" in fp["description"]
    assert "30000" in fp["parameters"]["properties"]["radius_m"]["description"]
    rec = convert_to_openai_tool(c.toolkit.by_name("get_patient_record"))["function"]
    assert "P001" in rec["parameters"]["properties"]["patient_id"]["description"]
    # The injected config never leaks into what the model sees.
    assert "config" not in rec["parameters"]["properties"]


# --- new: bugs found in the step 1/2 review ---------------------------------
async def test_guard_still_works_on_later_turns():
    """Was broken: confirmed_providers persists across turns, so ONE
    confirmation in turn 1 switched the guard off for the whole chat. A list
    of invented names in turn 2 went out verbatim."""
    c = make_container()
    thread = "t-" + uuid.uuid4().hex
    c.agent.model = ScriptedModel([
        AIMessage(content="", id="a1",
                  tool_calls=[call("find_providers", {"specialty": "Cardiology", **LOC}, 1)]),
        AIMessage(content="1. Dr. Meera Iyer - 1 km", id="a2"),
        AIMessage(content="Psychiatrists:\n1. Dr. Invented Person - 2 km", id="b1"),
        AIMessage(content="As before:\n1. Dr. Meera Iyer - 1 km", id="c1"),
    ])
    await c.agent.run_turn("chest pain", thread_id=thread)
    invented, _ = await c.agent.run_turn("now I need a psychiatrist", thread_id=thread)
    assert "Invented Person" not in invented and "withheld" in invented
    # ...but re-listing a provider a tool confirmed EARLIER is still fine.
    again, _ = await c.agent.run_turn("remind me of the cardiologist", thread_id=thread)
    assert again == "As before:\n1. Dr. Meera Iyer - 1 km"


async def test_guard_unit_rules():
    g = AnswerGuard()
    lst = "1. Known Clinic - 1 km\n2. Unknown Place - 2 km"
    kw = dict(offered=set(), failed=())
    assert g.check_recommendations(lst, confirmed_this_turn={"x"}, confirmed_ever=set(), **kw) == lst
    assert "withheld" in g.check_recommendations(
        lst, confirmed_this_turn=set(), confirmed_ever={"Known Clinic"}, **kw)
    assert g.check_recommendations("no list here", confirmed_this_turn=set(),
                                   confirmed_ever=set(), **kw) == "no list here"


async def test_tools_refuse_other_patients_records():
    """Was open: patient_id comes from the model, so 'show me patient P002's
    record' returned it. With a bound patient, other ids are refused."""
    c = make_container()
    state, _ = await run([
        AIMessage(content="", id="x1",
                  tool_calls=[call("get_patient_record", {"patient_id": "P002"}, 1)]),
        AIMessage(content="", id="x2", tool_calls=[call(
            "update_patient_record", {"patient_id": "P003", "name": "Mallory"}, 2)]),
        AIMessage(content="Sorry, I can't share that.", id="x3"),
    ], c, patient_id="LIVE-me")
    payloads = [m.content for m in state["messages"] if m.type == "tool"]
    assert all("not allowed" in p for p in payloads), payloads
    assert (await c.patients.get("P003")).get("name") != "Mallory"     # no write happened


async def test_unbound_runs_keep_old_behaviour():
    """The CLI demo binds no patient and may read any demo record."""
    state, _ = await run([
        AIMessage(content="", id="y1",
                  tool_calls=[call("get_patient_record", {"patient_id": "P002"}, 1)]),
        AIMessage(content="ok", id="y2")])
    assert "patient_id" in [m for m in state["messages"] if m.type == "tool"][0].content


async def test_osm_dedupe_is_shared_by_emergency_and_directory():
    items = [{"name": "NIMHANS"}, {"name": "nimhans"}, {"name": "Other"}]
    assert [i["name"] for i in OsmProviderDirectory.dedupe(items)] == ["NIMHANS", "Other"]
