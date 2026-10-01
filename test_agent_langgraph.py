"""
test_agent_langgraph.py — offline regression harness for the LangGraph port.

No API key, no network. A scripted stand-in replaces the model and replays
fixed tool-call sequences ("golden transcripts"), so these tests pin down the
GRAPH's behaviour — tracking, guard, step cap, prompt injection — independent
of whatever a live model happens to decide. The deterministic dummy backend
in tools.py supplies the real tool results.

Run:
    python test_agent_langgraph.py
"""

import os

# Must happen BEFORE importing agent_langgraph: ChatOpenAI wants a key at
# construction (never used — the model is replaced), and the dummy backend is
# forced so results are deterministic regardless of your .env.
os.environ.setdefault("OPENAI_API_KEY", "offline-test")
os.environ["USE_REAL_PROVIDERS"] = ""
# Shared mode (REDIS_URL set): give every run its own Redis namespace. Redis
# state outlives the process — unlike the in-memory stores — so without this a
# previous run's emptied rate-limit bucket makes this run fail with 429s.
import uuid  # noqa: E402
os.environ["CAREROUTE_KEY_PREFIX"] = f"careroute-test-{uuid.uuid4().hex[:8]}"

from langchain_core.messages import AIMessage, SystemMessage  # noqa: E402

import agent_langgraph as m  # noqa: E402


class ScriptedModel:
    """Duck-types llm_with_tools: .invoke() pops the next scripted reply and
    records what the graph sent, so tests can also inspect the prompt."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.script.pop(0)


def call(name, args, i):
    return {"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"}


LOC = {"patient_lat": 12.9352, "patient_lng": 77.6245}  # P001, Koramangala
REQ = "Patient P001 is experiencing chest pain. Find the nearest specialists."


def run(script):
    model = ScriptedModel(script)
    m.llm_with_tools = model  # swap the engine's brain for a script
    state = m.app.invoke({
        "messages": [m.HumanMessage(content=REQ)],
        "confirmed_providers": set(),
        "offered_facilities": set(),
        "llm_calls": 0,
    })
    return state, model


def test_happy_path_answer_untouched():
    good = "Nearest cardiologists:\n1. Dr. A - 1.2 km\n2. Dr. B - 2.9 km"
    state, model = run([
        AIMessage(content="", id="ai1",
                  tool_calls=[call("get_patient_record", {"patient_id": "P001"}, 1)]),
        AIMessage(content="", id="ai2",
                  tool_calls=[call("find_providers",
                                   {"specialty": "Cardiology", **LOC}, 2)]),
        AIMessage(content=good, id="ai3"),
    ])
    # guard passed it through
    assert state["messages"][-1].content == good
    # filled from real tool output
    assert state["confirmed_providers"]
    assert state["llm_calls"] == 3
    sys_msgs = [x for x in model.seen[-1] if isinstance(x, SystemMessage)]
    # prompt injected exactly once
    assert len(sys_msgs) == 1


def test_hallucinated_list_is_withheld():
    state, _ = run([
        AIMessage(content="Here you go:\n1. Totally Real Cardiac Centre - 0.4 km",
                  id="bad"),
    ])
    final = state["messages"][-1].content
    assert "withheld" in final
    assert "Totally Real" not in final
    # untrusted msg removed
    assert "bad" not in {x.id for x in state["messages"]}


def test_fallback_only_gets_disclaimer_prefix():
    state, _ = run([
        AIMessage(content="", id="ai1",
                  tool_calls=[call("find_general_facilities", {**LOC, "k": 2}, 1)]),
        AIMessage(content="Options:\n1. Some General Hospital - 1.1 km", id="ai2"),
    ])
    final = state["messages"][-1].content
    assert final.startswith("No verified specialist match")
    assert state["offered_facilities"] and not state["confirmed_providers"]


def test_step_cap_stops_cleanly():
    loop = [
        AIMessage(content="", id=f"ai{i}",
                  tool_calls=[call("get_patient_record", {"patient_id": "P001"}, i)])
        for i in range(1, m.MAX_LLM_CALLS + 1)
    ]
    state, _ = run(loop)
    final = state["messages"][-1]
    assert final.content.startswith("Stopped:")
    assert state["llm_calls"] == m.MAX_LLM_CALLS
    # the dangling tool request was removed, so the transcript stays valid
    assert not any(getattr(x, "tool_calls", None) and x.id == f"ai{m.MAX_LLM_CALLS}"
                   for x in state["messages"])


def test_upstream_outage_is_not_reported_as_no_specialists():
    """Replays the real 2026-08 run: Overpass 504s, the model gives up and
    tells a cardiac patient no cardiologist was found. A tool that FAILED is
    not a tool that found nothing — the guard must say so."""
    import tools as t
    broken = {"error": "Provider directory unreachable: 504 Gateway Timeout",
              "error_type": "upstream_unavailable"}
    real = t.find_providers
    t.find_providers = lambda **kw: broken   # simulate every mirror down
    try:
        state, _ = run([
            AIMessage(content="", id="ai1",
                      tool_calls=[call("find_providers",
                                       {"specialty": "Cardiology", **LOC}, 1)]),
            AIMessage(content="I could not find any nearby cardiology "
                              "specialists.\n1. Some Clinic - 0.2 km", id="ai2"),
        ])
    finally:
        t.find_providers = real

    final = state["messages"][-1].content
    assert state["tool_errors"], "the failure was never recorded"
    assert not state["confirmed_providers"]
    assert final.startswith("The provider directory could not be reached")
    assert "NOT a confirmed result" in final


def test_turn_trace_records_tools_without_pii():
    state, _ = run([
        AIMessage(content="", id="ai1",
                  tool_calls=[call("update_patient_record",
                                   {"patient_id": "P001", "name": "Secret Name"}, 1)]),
        AIMessage(content="", id="ai2",
                  tool_calls=[call("find_providers",
                                   {"specialty": "Cardiology", **LOC}, 2)]),
        AIMessage(content="Nearest:\n1. Dr. A - 1.2 km", id="ai3"),
    ])
    trace = m._turn_trace(state["messages"])
    assert [t["tool"] for t in trace["tools"]] == ["update_patient_record",
                                                   "find_providers"]
    assert "name" not in trace["tools"][0]["args"]           # PII dropped
    # coords coarsened
    assert trace["tools"][1]["args"]["patient_lat"] == 12.94
    assert trace["tools"][1]["result"].startswith("OK list")
    assert trace["llm_calls"] == 3


def test_emergency_sticks_across_turns():
    """Replays the real 2026-09-27 chat: chest pain (emergency) -> "does Maya
    Hospital do cardiology?" -> "too far, traffic, what do I do?". On turn 3
    the model suggested a small clinic and never mentioned 112/108. From the
    first emergency on, every later turn must (a) carry the emergency context
    in the prompt and (b) contain the emergency number, enforced in code."""
    import uuid
    import emergency
    import osm
    real_fetch, real_cc = osm._fetch_nearby, emergency._country_code
    osm._fetch_nearby = lambda *a, **k: ([], None)   # no network in tests
    emergency._country_code = lambda lat, lng: "IN"
    thread = "t-" + uuid.uuid4().hex
    try:
        model = ScriptedModel([
            AIMessage(content="", id="e1",
                      tool_calls=[call("get_emergency_help", LOC, 1)]),
            AIMessage(content="Call 112 (ambulance: 108) now.", id="e2"),
            AIMessage(
                content="Maya Hospital is not listed for cardiology.", id="e3"),
            AIMessage(
                content="Nearby options:\n1. Sai Clinic - 0.66 km", id="e4"),
        ])
        m.llm_with_tools = model
        a1, _ = m.continue_conversation_traced("chest pain", thread)
        a2, _ = m.continue_conversation_traced(
            "does Maya do cardiology?", thread)
        a3, t3 = m.continue_conversation_traced(
            "too far, what do I do?", thread)
    finally:
        osm._fetch_nearby, emergency._country_code = real_fetch, real_cc

    assert a1 == "Call 112 (ambulance: 108) now."   # already has it: untouched
    for later in (a2, a3):
        assert later.startswith("If the symptoms you described earlier")
        assert "112" in later
    sys3 = [x for x in model.seen[-1]
            if isinstance(x, SystemMessage)][0].content
    assert "EMERGENCY CONTEXT" in sys3               # prompt carries it on turn 3
    assert t3["tools"] == []


def test_emergency_list_skips_single_doctor_and_narrow_hospitals():
    import emergency
    facilities = [
        {"name": "Maya Hospital", "facility": "hospital", "_er": ""},
        {"name": "Dr Ramesh Dalwai Spine Surgeon",
            "facility": "hospital", "_er": ""},
        {"name": "City Eye Hospital", "facility": "hospital", "_er": ""},
        {"name": "Rest Home Hospital", "facility": "hospital", "_er": "no"},
        {"name": "Sai Clinic", "facility": "clinic", "_er": ""},
        {"name": "Sakra World Hospital", "facility": "hospital", "_er": "yes"},
    ]
    names = [f["name"] for f in emergency._er_candidates(facilities)]
    assert names == ["Maya Hospital", "Sakra World Hospital"]


def _fake_nominatim(known):
    """known: {query: (lat, lng, place_rank)} — anything else is not found."""
    def fake(query):
        if query in known:
            lat, lng, rank = known[query]
            return [{"lat": str(lat), "lon": str(lng), "place_rank": rank,
                     "display_name": f"{query}, Bengaluru, Karnataka, India"}]
        return []
    return fake


def test_geocode_falls_back_from_apartment_to_area_but_not_city():
    import tools as t
    real = t._nominatim
    t._GEOCODE_CACHE.clear()
    t._nominatim = _fake_nominatim({
        "Whitefield, Bangalore": (12.97, 77.75, 19),   # suburb: fine
        "Bangalore": (12.97, 77.59, 16),               # whole city
    })
    try:
        g = t.geocode_place("Prestige Shantiniketan, Whitefield, Bangalore")
        assert g["matched_query"] == "Whitefield, Bangalore" and g["approximate"]
        # apartment AND area unknown: must NOT silently fall back to the city
        g2 = t.geocode_place("Nowhere Towers, Unknownpura, Bangalore")
        assert g2.get("match_found") is False
        # but a user who types just the city gets it, flagged as broad
        g3 = t.geocode_place("Bangalore")
        assert g3["broad"] and not g3["approximate"]
    finally:
        t._nominatim = real
        t._GEOCODE_CACHE.clear()


def test_chat_location_from_gps_or_from_the_message():
    """No location box on the page: GPS on the first message, otherwise the
    user names the place in chat and the agent geocodes it."""
    from fastapi.testclient import TestClient
    import server
    import tools as t
    real = t._nominatim
    t._GEOCODE_CACHE.clear()
    t._nominatim = _fake_nominatim(
        {"HSR Layout, Bengaluru": (12.91, 77.64, 20)})
    try:
        c = TestClient(server.app)

        # 1) GPS blocked, no place in the message: allowed; the model is told
        #    the location is UNKNOWN (and, per the prompt, asks for it).
        model = ScriptedModel([AIMessage(content="Where are you?", id="q1")])
        m.llm_with_tools = model
        r = c.post("/chat", json={"message": "knee pain"})
        assert r.status_code == 200 and r.json()["location"] is None
        sent = model.seen[0][-1].content
        assert "location=UNKNOWN" in sent
        sid = r.json()["session_id"]
        pid = server.sessions.get_session(sid)["patient_id"]

        # 2) The user names the place in chat: the agent geocodes it and saves
        #    it, and the page is told where the search now happens.
        m.llm_with_tools = ScriptedModel([
            AIMessage(content="", id="g1", tool_calls=[call(
                "geocode_place", {"place": "HSR Layout, Bengaluru"}, 1)]),
            AIMessage(content="", id="g2", tool_calls=[call(
                "update_patient_record", {"patient_id": pid, "lat": 12.91,
                                          "lng": 77.64, "area": "HSR Layout, Bengaluru"}, 2)]),
            AIMessage(content="Searching near HSR Layout.", id="g3"),
        ])
        r2 = c.post("/chat", json={"message": "I'm in HSR Layout, Bengaluru",
                                   "session_id": sid}).json()
        assert t.get_patient_record(pid)["lat"] == 12.91
        assert r2["location"] == {
            "label": "HSR Layout, Bengaluru", "source": "chat"}

        # 3) GPS on the first message; a follow-up without coordinates must
        #    not move the patient.
        m.llm_with_tools = ScriptedModel([AIMessage(content="ok", id="p1"),
                                          AIMessage(content="ok", id="p2")])
        r3 = c.post("/chat", json={"message": "fever", "lat": 12.97,
                                   "lng": 77.64}).json()
        pid3 = server.sessions.get_session(r3["session_id"])["patient_id"]
        c.post("/chat", json={"message": "and a cough",
                              "session_id": r3["session_id"]})
        assert t.get_patient_record(pid3)["lat"] == 12.97
    finally:
        t._nominatim = real
        t._GEOCODE_CACHE.clear()


def test_tool_schema_kept_the_old_guidance():
    from langchain_core.utils.function_calling import convert_to_openai_tool
    fp = convert_to_openai_tool(m.find_providers)["function"]
    # recovery guidance survived
    assert "match_found" in fp["description"]
    props = fp["parameters"]["properties"]
    assert "30000" in props["radius_m"]["description"]  # param hints survived
    assert "P001" in convert_to_openai_tool(
        m.get_patient_record)["function"]["parameters"][
        "properties"]["patient_id"]["description"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\nAll {len(tests)} green — graph behaviour matches agent.py.")
