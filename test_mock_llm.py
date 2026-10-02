"""
test_mock_llm.py — the mock model behaves like a careful tool-calling model.

Three layers:
  * decide(): the pure decision function, on hand-built OpenAI transcripts.
  * HTTP contract: responses parse as OpenAI chat completions (tool_calls with
    JSON-string arguments, finish_reason, usage) and faults switch at runtime.
  * End to end (needs GEO_TEST_DATABASE_URL + osm2pgsql, like test_geo.py):
    the real server.py, talking to the mock over HTTP, against the PostGIS
    fixture. Counts model calls per turn, which is step 5's baseline metric.

    python -m pytest -q test_mock_llm.py
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

import mock_llm

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = {"get_patient_record", "update_patient_record", "geocode_place",
         "find_providers", "find_general_facilities", "find_pharmacies",
         "get_emergency_help"}
KNOWN = "lat=12.9352, lng=77.6245, area=Koramangala"
UNKNOWN = "location=UNKNOWN (browser location was not shared)"


def user(text, where=KNOWN, pid="LIVE-abc"):
    return {"role": "user", "content": (
        f"Patient {pid} [record: name=Live user, {where}, "
        f"current_medications=none] reports these symptoms: {text}. "
        "Find the nearest appropriate specialists.")}


def step(name, args, result, i):
    """The assistant tool call + its tool result, as OpenAI messages."""
    cid = f"call_{i}"
    return [{"role": "assistant", "content": None, "tool_calls": [
        {"id": cid, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": cid, "content": json.dumps(result)}]


def decide(*msgs):
    return mock_llm.decide(list(msgs), TOOLS)


# --- decide(): the model's judgement --------------------------------------
def test_emergency_calls_emergency_tool_first():
    kind, calls = decide(user("crushing chest pain"))
    assert kind == "tools" and calls[0][0] == "get_emergency_help"


def test_emergency_answer_leads_with_number_and_uses_only_tool_names():
    res = {"emergency": True, "emergency_number": "112 (ambulance: 108)",
           "nearest_hospitals": [{"name": "Sakra", "distance_km": 0.4}]}
    kind, text = decide(user("seizure"),
                        *step("get_emergency_help", {}, res, 1))
    assert kind == "answer" and text.startswith(
        "This may be a medical emergency. Call 112")
    assert "1. Sakra" in text


def test_emergency_without_location_still_gives_number():
    kind, text = decide(user("he fainted", where=UNKNOWN))
    assert kind == "answer" and "112" in text


def test_widens_radius_twice_then_answers_with_alternatives():
    miss = {"match_found": False, "general_alternatives": [
        {"name": "City Clinic", "distance_km": 1.0}]}
    msgs = [user("stiff joints, maybe arthritis")]
    radii = []
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
    err = {"error": "Provider directory unreachable",
           "error_type": "upstream_unavailable"}
    msgs = [user("palpitations")]
    kind, out = decide(*msgs)
    msgs += step(*out[0], err, 1)
    kind, out = decide(*msgs)
    assert kind == "tools" and out[0][1]["radius_m"] == 8000   # same call
    msgs += step(*out[0], err, 2)
    kind, text = decide(*msgs)
    assert kind == "answer" and "unreachable" in text and "1." not in text


def test_pharmacy_not_clinic():
    kind, calls = decide(user("need to buy ORS"))
    assert calls[0][0] == "find_pharmacies"


def test_named_place_geocoded_then_saved_and_searched_together():
    kind, calls = decide(user("I'm staying in Hubballi and have a migraine",
                              where=UNKNOWN))
    assert calls == [("geocode_place", {"place": "Hubballi"})]
    geo = {"lat": 15.36, "lng": 75.12, "display_name": "Hubballi"}
    kind, calls = decide(user("I'm staying in Hubballi and have a migraine",
                              where=UNKNOWN),
                         *step("geocode_place", {"place": "Hubballi"}, geo, 1))
    assert [c[0] for c in calls] == ["update_patient_record", "find_providers"]
    assert calls[1][1]["patient_lat"] == 15.36
    assert calls[1][1]["specialty"] == "Neurology"


def test_geocode_failure_is_not_reported_as_unknown_place():
    err = {"error": "Geocoding failed: connection refused"}
    kind, text = decide(user("I'm in Mysuru with a fever", where=UNKNOWN),
                        *step("geocode_place", {"place": "Mysuru"}, err, 1))
    assert "can't look up" in text and "couldn't find" not in text


def test_no_location_asks_for_it():
    kind, text = decide(user("my knee hurts", where=UNKNOWN))
    assert kind == "answer" and "Where are you" in text


def test_child_maps_to_pediatrics():
    _, calls = decide(user("my 4 year old son has a rash"))
    assert calls[0][1]["specialty"] == "Pediatrics"


def test_only_tools_the_request_offers():
    """The mock must never call a tool the app did not bind: without
    geocode_place it asks for the location instead of geocoding."""
    kind, out = mock_llm.decide([user("I'm in Hubballi, knee pain", where=UNKNOWN)],
                                TOOLS - {"geocode_place"})
    assert kind == "answer" and "Where are you" in out


# --- HTTP contract ----------------------------------------------------------
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(mock_llm, "LAT_MIN", 0.0)
    monkeypatch.setattr(mock_llm, "LAT_MAX", 0.0)
    c = TestClient(mock_llm.app)
    c.post("/admin/reset")
    yield c
    c.post("/admin/reset")


def _body(*msgs):
    return {"model": "gpt-4o-mini", "messages": list(msgs),
            "tools": [{"type": "function", "function": {"name": n}} for n in TOOLS]}


def test_tool_call_response_is_valid_openai(client):
    r = client.post("/v1/chat/completions", json=_body(user("palpitations")))
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc["type"] == "function" and tc["id"].startswith("call_")
    assert json.loads(tc["function"]["arguments"])["specialty"] == "Cardiology"
    assert r.json()["usage"]["total_tokens"] > 0


def test_faults_switch_at_runtime(client):
    for mode, code in (("down", 503), ("ratelimit", 429), ("ok", 200)):
        client.post("/admin/fault", json={"mode": mode})
        assert client.post("/v1/chat/completions",
                           json=_body(user("x"))).status_code == code
    s = client.get("/admin/stats").json()
    assert s["faults_503"] == 1 and s["faults_429"] == 1 and s["requests"] == 3


def test_echo_mode_still_supported(client, monkeypatch):
    monkeypatch.setattr(mock_llm, "MODE", "echo")
    r = client.post("/v1/chat/completions", json=_body(user("a"), user("b")))
    assert r.json()["choices"][0]["message"]["content"].startswith(
        "turns_seen=2")


# --- end to end: real server.py + mock + PostGIS fixture ---------------------
DSN = os.getenv("GEO_TEST_DATABASE_URL", "").strip()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not DSN or not shutil.which("osm2pgsql"),
                    reason="needs GEO_TEST_DATABASE_URL and osm2pgsql")
def test_full_stack_turns_and_call_counts():
    import httpx
    subprocess.run(["bash", os.path.join(HERE, "geo", "import.sh")], check=True,
                   capture_output=True,
                   env={**os.environ, "GEO_DATABASE_URL": DSN,
                        "PBF": os.path.join(HERE, "geo", "fixtures", "koramangala_sample.osm"),
                        "STYLE": os.path.join(HERE, "geo", "healthcare.lua")})
    mport, aport = _free_port(), _free_port()
    env = {**os.environ,
           "OPENAI_API_KEY": "mock",
           "OPENAI_BASE_URL": f"http://127.0.0.1:{mport}/v1",
           "OPENAI_API_BASE": f"http://127.0.0.1:{mport}/v1",
           "MOCK_LLM_MODE": "agent",
           "MOCK_LLM_LATENCY_MIN_S": "0.01", "MOCK_LLM_LATENCY_MAX_S": "0.02",
           "USE_REAL_PROVIDERS": "osm", "OSM_SOURCE": "postgis",
           "GEO_DATABASE_URL": DSN,
           "OSRM_BASE_URL": "http://127.0.0.1:9",        # down: straight-line
           "NOMINATIM_URL": "http://127.0.0.1:9",        # down: geocode fails
           "REDIS_URL": "", "DATABASE_URL": "",
           "CAREROUTE_REQUIRE_SHARED_STATE": "0",
           "CAREROUTE_RATE_PER_MIN": "1000", "CAREROUTE_BURST": "1000"}
    procs = [subprocess.Popen([sys.executable, "-m", "uvicorn", app, "--port",
                               str(port), "--log-level", "warning"],
                              cwd=HERE, env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
             for app, port in (("mock_llm:app", mport), ("server:app", aport))]
    try:
        for _ in range(60):
            try:
                httpx.get(f"http://127.0.0.1:{aport}/version", timeout=1)
                httpx.get(f"http://127.0.0.1:{mport}/healthz", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.5)

        def turn(msg, **loc):
            before = httpx.get(f"http://127.0.0.1:{mport}/admin/stats").json()
            r = httpx.post(f"http://127.0.0.1:{aport}/chat", timeout=60,
                           json={"message": msg, **loc})
            after = httpx.get(f"http://127.0.0.1:{mport}/admin/stats").json()
            assert r.status_code == 200, r.text
            return r.json()["answer"], after["requests"] - before.get("requests", 0)

        here = {"lat": 12.9352, "lng": 77.6245}
        a, n = turn("I keep getting palpitations", **here)
        assert "1. Heart Care Clinic" in a and n == 2
        a, n = turn("stiff joints every morning, maybe arthritis", **here)
        assert "not specialists" in a and n == 4                # 8 -> 16 -> 30 km
        a, n = turn("sudden chest pain spreading to my left arm", **here)
        assert a.startswith(
            "This may be a medical emergency. Call 112") and n == 2
        assert "St. John's Medical College Hospital" in a
        a, n = turn("I need to buy paracetamol", **here)
        assert "1. Apollo Pharmacy" in a and "withheld" not in a  # guard fix
        a, n = turn("I'm in Mysuru and have a bad headache")
        assert "can't look up place names" in a and n == 2
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=10)
