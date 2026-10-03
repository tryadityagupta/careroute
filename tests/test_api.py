"""
tests/test_api.py — the HTTP layer, through FastAPI's TestClient.

The whole app runs for real (gate, services, agent, stores) with a scripted
model and a fake geocoder injected through the Container.
"""

import httpx
import openai
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from careroute.api.app import create_app
from tests.fakes import FailingModel, FakeGeocoder, ScriptedModel, call, make_container


def client_for(container):
    return TestClient(create_app(container))


def test_chat_location_from_gps_or_from_the_message():
    """No location box on the page: GPS on the first message, otherwise the
    user names the place in chat and the agent geocodes it."""
    c = make_container()
    c.geocoder = FakeGeocoder({"HSR Layout, Bengaluru": (12.91, 77.64, 20)})
    http = client_for(c)

    # 1) No GPS, no place: allowed; the model is told location=UNKNOWN.
    model = ScriptedModel([AIMessage(content="Where are you?", id="q1")])
    c.agent.model = model
    r = http.post("/chat", json={"message": "knee pain"})
    assert r.status_code == 200 and r.json()["location"] is None
    assert "location=UNKNOWN" in model.seen[0][-1].content
    sid = r.json()["session_id"]
    pid = c.sessions.get_session(sid)["patient_id"]

    # 2) The user names the place: the agent geocodes and saves it, and the
    #    page is told where the search now happens.
    c.agent.model = ScriptedModel([
        AIMessage(content="", id="g1", tool_calls=[call(
            "geocode_place", {"place": "HSR Layout, Bengaluru"}, 1)]),
        AIMessage(content="", id="g2", tool_calls=[call(
            "update_patient_record", {"patient_id": pid, "lat": 12.91, "lng": 77.64,
                                      "area": "HSR Layout, Bengaluru"}, 2)]),
        AIMessage(content="Searching near HSR Layout.", id="g3"),
    ])
    r2 = http.post("/chat", json={"message": "I'm in HSR Layout, Bengaluru",
                                  "session_id": sid}).json()
    assert c.patients.get(pid)["lat"] == 12.91
    assert r2["location"] == {"label": "HSR Layout, Bengaluru", "source": "chat"}

    # 3) GPS on turn 1; a follow-up without coordinates must not move the patient.
    c.agent.model = ScriptedModel([AIMessage(content="ok", id="p1"),
                                   AIMessage(content="ok", id="p2")])
    r3 = http.post("/chat", json={"message": "fever", "lat": 12.97, "lng": 77.64}).json()
    pid3 = c.sessions.get_session(r3["session_id"])["patient_id"]
    http.post("/chat", json={"message": "and a cough", "session_id": r3["session_id"]})
    assert c.patients.get(pid3)["lat"] == 12.97


def test_care_endpoint_cleans_up_its_record():
    c = make_container()
    c.agent.model = ScriptedModel([AIMessage(content="See a GP.", id="c1")])
    r = client_for(c).post("/care", json={"symptoms": "fever", "lat": 12.97, "lng": 77.64})
    assert r.status_code == 200 and r.json()["answer"] == "See a GP."
    assert r.json()["location"]["source"] == "gps"


def test_typed_place_that_cannot_be_found_is_a_friendly_422():
    c = make_container()
    r = client_for(c).post("/chat", json={"message": "fever", "location_text": "Atlantis"})
    assert r.status_code == 422 and "couldn't find" in r.json()["detail"]


def test_llm_outage_is_503_with_the_emergency_number():
    """Was an unhandled 500 with no emergency number."""
    c = make_container()
    req = httpx.Request("POST", "http://mock/v1/chat/completions")
    c.agent.model = FailingModel(openai.APIConnectionError(request=req))
    r = client_for(c).post("/chat", json={"message": "chest pain", "lat": 12.97, "lng": 77.64})
    assert r.status_code == 503
    assert "112" in r.json()["detail"]
    assert r.headers.get("retry-after") == "5"


def test_rate_limit_applies_to_chat():
    c = make_container(rate_per_min=1, burst=2)
    c.agent.model = ScriptedModel([])
    http = client_for(c)
    codes = [http.post("/chat", json={"message": " "}).status_code for _ in range(4)]
    assert codes == [400, 400, 429, 429]       # blank message: rejected AFTER the gate


def test_version_reports_backend():
    assert client_for(make_container()).get("/version").json()["backend"] == "dummy-json"
