"""
tests/test_mock_llm.py — the mock model behaves like a careful tool-calling model.

Three layers:
  * AgentPolicy.decide(): the pure decision function, on hand-built transcripts.
  * HTTP contract: responses parse as OpenAI chat completions; faults switch live.
  * End to end (needs GEO_TEST_DATABASE_URL + osm2pgsql): the real app talking to
    the mock over HTTP, against the PostGIS fixture. Counts model calls per
    turn — step 5's baseline metric.
"""

import json
import os
import shutil

import httpx
import pytest
from fastapi.testclient import TestClient

from mock_llm.app import MockConfig, create_app
from mock_llm.policy import AgentPolicy
from tests import procs

TOOLS = {"get_patient_record", "update_patient_record", "geocode_place",
         "find_providers", "find_general_facilities", "find_pharmacies", "get_emergency_help"}
KNOWN = "lat=12.9352, lng=77.6245, area=Koramangala"
UNKNOWN = "location=UNKNOWN (browser location was not shared)"
policy = AgentPolicy()


def user(text, where=KNOWN, pid="LIVE-abc"):
    return {"role": "user", "content": (
        f"Patient {pid} [record: name=Live user, {where}, current_medications=none] "
        f"reports these symptoms: {text}. Find the nearest appropriate specialists.")}


def step(name, args, result, i):
    cid = f"call_{i}"
    return [{"role": "assistant", "content": None, "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": cid, "content": json.dumps(result)}]


def decide(*msgs, tools=TOOLS):
    return policy.decide(list(msgs), tools)


# --- decide(): the model's judgement ------------------------------------------
def test_emergency_calls_emergency_tool_first():
    kind, calls = decide(user("crushing chest pain"))
    assert kind == "tools" and calls[0][0] == "get_emergency_help"


def test_emergency_answer_leads_with_number_and_uses_only_tool_names():
    res = {"emergency": True, "emergency_number": "112 (ambulance: 108)",
           "nearest_hospitals": [{"name": "Sakra", "distance_km": 0.4}]}
    kind, text = decide(user("seizure"), *step("get_emergency_help", {}, res, 1))
    assert kind == "answer" and text.startswith("This may be a medical emergency. Call 112")
    assert "1. Sakra" in text


def test_emergency_without_location_still_gives_number():
    kind, text = decide(user("he fainted", where=UNKNOWN))
    assert kind == "answer" and "112" in text


def test_widens_radius_twice_then_answers_with_alternatives():
    miss = {"match_found": False, "general_alternatives": [{"name": "City Clinic", "distance_km": 1.0}]}
    msgs, radii = [user("stiff joints, maybe arthritis")], []
    for i in range(1, 5):
        kind, out = decide(*msgs)
        if kind == "answer":
            break
        name, args = out[-1]
        assert name == "find_providers" and args["specialty"] == "Rheumatology"
        radii.append(args["radius_m"])
        msgs += step(name, args, miss, i)
    assert radii == [8000, 16000, 30000]
    assert "not specialists" in out and "1. City Clinic" in out


def test_directory_error_retries_once_then_never_lists():
    err = {"error": "Provider directory unreachable", "error_type": "upstream_unavailable"}
    msgs = [user("palpitations")]
    kind, out = decide(*msgs)
    msgs += step(*out[0], err, 1)
    kind, out = decide(*msgs)
    assert kind == "tools" and out[0][1]["radius_m"] == 8000          # same call
    msgs += step(*out[0], err, 2)
    kind, text = decide(*msgs)
    assert kind == "answer" and "unreachable" in text and "1." not in text


def test_pharmacy_not_clinic():
    assert decide(user("need to buy ORS"))[1][0][0] == "find_pharmacies"


def test_named_place_geocoded_then_saved_and_searched_together():
    msg = user("I'm staying in Hubballi and have a migraine", where=UNKNOWN)
    kind, calls = decide(msg)
    assert calls == [("geocode_place", {"place": "Hubballi"})]
    geo = {"lat": 15.36, "lng": 75.12, "display_name": "Hubballi"}
    kind, calls = decide(msg, *step("geocode_place", {"place": "Hubballi"}, geo, 1))
    assert [c[0] for c in calls] == ["update_patient_record", "find_providers"]
    assert calls[1][1]["patient_lat"] == 15.36 and calls[1][1]["specialty"] == "Neurology"


def test_geocode_failure_is_not_reported_as_unknown_place():
    err = {"error": "Geocoding failed: connection refused"}
    kind, text = decide(user("I'm in Mysuru with a fever", where=UNKNOWN),
                        *step("geocode_place", {"place": "Mysuru"}, err, 1))
    assert "can't look up" in text and "couldn't find" not in text


def test_no_location_asks_for_it():
    kind, text = decide(user("my knee hurts", where=UNKNOWN))
    assert kind == "answer" and "Where are you" in text


def test_child_maps_to_pediatrics():
    assert decide(user("my 4 year old son has a rash"))[1][0][1]["specialty"] == "Pediatrics"


def test_only_tools_the_request_offers():
    kind, out = decide(user("I'm in Hubballi, knee pain", where=UNKNOWN),
                       tools=TOOLS - {"geocode_place"})
    assert kind == "answer" and "Where are you" in out


# --- HTTP contract --------------------------------------------------------------
def _client(mode="agent"):
    return TestClient(create_app(MockConfig(mode=mode, latency_min_s=0, latency_max_s=0)))


def _body(*msgs):
    return {"model": "gpt-4o-mini", "messages": list(msgs),
            "tools": [{"type": "function", "function": {"name": n}} for n in TOOLS]}


def test_tool_call_response_is_valid_openai():
    r = _client().post("/v1/chat/completions", json=_body(user("palpitations")))
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc["type"] == "function" and tc["id"].startswith("call_")
    assert json.loads(tc["function"]["arguments"])["specialty"] == "Cardiology"
    assert r.json()["usage"]["total_tokens"] > 0


def test_faults_switch_at_runtime():
    c = _client()
    for mode, code in (("down", 503), ("ratelimit", 429), ("ok", 200)):
        c.post("/admin/fault", json={"mode": mode})
        assert c.post("/v1/chat/completions", json=_body(user("x"))).status_code == code
    s = c.get("/admin/stats").json()
    assert s["faults_503"] == 1 and s["faults_429"] == 1 and s["requests"] == 3


def test_echo_mode_still_supported():
    r = _client("echo").post("/v1/chat/completions", json=_body(user("a"), user("b")))
    assert r.json()["choices"][0]["message"]["content"].startswith("turns_seen=2")


# --- end to end: real app + mock + PostGIS fixture -----------------------------
DSN = os.getenv("GEO_TEST_DATABASE_URL", "").strip()


@pytest.mark.skipif(not DSN or not shutil.which("osm2pgsql"),
                    reason="needs GEO_TEST_DATABASE_URL and osm2pgsql")
def test_full_stack_turns_and_call_counts():
    from tests.test_geo import import_fixture
    import_fixture(DSN)
    mport, aport = procs.free_port(), procs.free_port()
    env = procs.base_env(
        OPENAI_API_KEY="mock", OPENAI_BASE_URL=f"http://127.0.0.1:{mport}/v1",
        MOCK_LLM_MODE="agent", MOCK_LLM_LATENCY_MIN_S="0.01", MOCK_LLM_LATENCY_MAX_S="0.02",
        USE_REAL_PROVIDERS="osm", OSM_SOURCE="postgis", GEO_DATABASE_URL=DSN,
        OSRM_BASE_URL="http://127.0.0.1:9",           # down: straight-line
        NOMINATIM_URL="http://127.0.0.1:9",           # down: geocode fails
        REDIS_URL="", DATABASE_URL="", CAREROUTE_REQUIRE_SHARED_STATE="0",
        CAREROUTE_RATE_PER_MIN="1000", CAREROUTE_BURST="1000")
    mock = procs.start(procs.MOCK, mport, env)
    app = procs.start(procs.APP, aport, env, probe="/version")
    try:
        def turn(msg, **loc):
            before = httpx.get(f"http://127.0.0.1:{mport}/admin/stats").json()
            r = httpx.post(f"http://127.0.0.1:{aport}/chat", timeout=60, json={"message": msg, **loc})
            after = httpx.get(f"http://127.0.0.1:{mport}/admin/stats").json()
            assert r.status_code == 200, r.text
            return r.json()["answer"], after["requests"] - before.get("requests", 0)

        here = {"lat": 12.9352, "lng": 77.6245}
        a, n = turn("I keep getting palpitations", **here)
        assert "1. Heart Care Clinic" in a and n == 2
        a, n = turn("stiff joints every morning, maybe arthritis", **here)
        assert "not specialists" in a and n == 4                     # 8 -> 16 -> 30 km
        a, n = turn("sudden chest pain spreading to my left arm", **here)
        assert a.startswith("This may be a medical emergency. Call 112") and n == 2
        assert "St. John's Medical College Hospital" in a
        a, n = turn("I need to buy paracetamol", **here)
        assert "1. Apollo Pharmacy" in a and "withheld" not in a
        a, n = turn("I'm in Mysuru and have a bad headache")
        assert "can't look up place names" in a and n == 2
    finally:
        procs.stop(app, mock)
