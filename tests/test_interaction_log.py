"""tests/test_interaction_log.py — PII omitted and coordinates coarsened by default."""

import io
import json

from careroute.observability.interaction_log import InteractionLogger

EXAMPLE = dict(endpoint="/chat", user_text="chest pain and dizzy", answer="Call 112 now.",
               session_id="abc", turn=2, meds=["aspirin"], name="Aditya",
               lat=12.971900, lng=77.641200, backend="openstreetmap", latency_ms=812)


def _emit(**kw):
    buf = io.StringIO()
    InteractionLogger(stream=buf, **kw).log(**EXAMPLE)
    return json.loads(buf.getvalue().strip())


def test_pii_omitted_and_coords_coarsened_by_default():
    rec = _emit()
    assert rec["user_text"] == "chest pain and dizzy" and rec["answer"] == "Call 112 now."
    assert rec["loc"] == {"lat": 12.97, "lng": 77.64}
    assert "name" not in rec and "loc_exact" not in rec


def test_pii_only_when_asked():
    rec = _emit(log_pii=True)
    assert rec["name"] == "Aditya" and rec["loc_exact"]["lat"] == 12.9719
