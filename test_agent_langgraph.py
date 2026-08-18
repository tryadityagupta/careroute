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
