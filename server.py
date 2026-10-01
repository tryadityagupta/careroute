"""
server.py — FastAPI backend that wraps the agent for the web demo.

You are the patient: describe symptoms, your browser shares your real location,
the agent finds real nearby providers. The only synthetic piece is "the patient"
(you), so the get_patient_record tool still demonstrates record retrieval.

Run it (PowerShell):
    $env:USE_REAL_PROVIDERS="osm"; uvicorn server:app --reload
Run it (bash/zsh):
    USE_REAL_PROVIDERS=osm uvicorn server:app --reload
Then open http://localhost:8000
"""

from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import tools
import os
import uuid
import time
import request_log
# Import ONLY the LangGraph engine. The previous version imported run_agent from
# BOTH agent and agent_langgraph; the second import silently shadowed the first,
# so agent.py was dead code that merely looked live. Now it's explicit.
# run_agent = single-shot (/care); continue_conversation = multi-turn (/chat).
from agent_langgraph import run_agent_traced, continue_conversation_traced
from security import require_api_key, rate_limit
import sessions
import patient_store
import shared_state
from dotenv import load_dotenv
load_dotenv()  # belt-and-braces; tools.py also loads .env before reading flags


app = FastAPI()

# Lock CORS to your real web origin(s) in production by setting
# CAREROUTE_ALLOWED_ORIGINS to a comma-separated list (e.g.
# "https://careroute.app"). Defaults to "*" so local dev is unchanged.
_origins = [o.strip() for o in os.getenv("CAREROUTE_ALLOWED_ORIGINS", "*").split(",")
            if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware, allow_origins=_origins, allow_methods=["*"], allow_headers=["*"]
)


class CareRequest(BaseModel):
    symptoms: str
    # Either the browser's coordinates OR a place the user typed.
    lat: float | None = None
    lng: float | None = None
    location_text: str | None = None
    name: str | None = None
    meds: str | None = None


class ChatRequest(BaseModel):
    # message = this turn's text (symptoms on turn 1, a follow-up afterwards).
    # session_id is None on the first turn; the response returns one to send back
    # on every follow-up. lat/lng are required only to START a conversation.
    message: str
    lat: float | None = None
    lng: float | None = None
    # A place the user typed: apartment, layout, landmark. Takes precedence
    # over lat/lng, because the user said it on purpose.
    location_text: str | None = None
    name: str | None = None
    meds: str | None = None
    session_id: str | None = None


def _record_summary(rec: dict) -> str:
    """The patient record, inline. SPEED FIX: the server already knows the
    location and meds, so hand them to the model directly instead of making it
    spend a full LLM round trip calling get_patient_record first."""
    meds = ", ".join(rec.get("current_medications") or []) or "none"
    if rec.get("lat") is None or rec.get("lng") is None:
        where = "location=UNKNOWN (browser location was not shared)"
    else:
        where = f"lat={rec.get('lat')}, lng={rec.get('lng')}, area={rec.get('area')}"
    return f"name={rec.get('name')}, {where}, current_medications={meds}"


def _moved_in_chat(rec: dict, before: tuple):
    """If the AGENT changed the location this turn (the user named a place in
    chat and it called update_patient_record), tell the page where it now
    searches, so the "Near: ..." status stays truthful."""
    after = (rec.get("lat"), rec.get("lng"))
    if after != before and after[0] is not None:
        return {"label": rec.get("area") or "the place you mentioned",
                "source": "chat"}
    return None


def _short_place(display_name: str) -> str:
    """'HSR Layout, Bengaluru South, Bengaluru Urban, Karnataka, 560102, India'
    -> 'HSR Layout, Bengaluru South, Bengaluru Urban'."""
    return ", ".join(p.strip() for p in display_name.split(",")[:3])


def _resolve_location(location_text, lat, lng, required: bool):
    """Turn what the user gave us into (lat, lng, area, info-for-the-UI).

    A typed place wins over GPS. Returns None when nothing was given and it
    is not required (a follow-up that doesn't change location). Raises a 422
    with a message the page shows as-is when a typed place can't be found.
    """
    text = (location_text or "").strip()
    if text:
        if len(text) > 200:
            raise HTTPException(422, "That location is too long. Try just your "
                                "area or a landmark, plus the city.")
        g = tools.geocode_place(text)
        if "lat" not in g:
            if g.get("error"):
                raise HTTPException(422, "We couldn't look that place up right "
                                    "now. Try again in a moment, or allow "
                                    "location access.")
            raise HTTPException(422, f"We couldn't find \u201c{text}\u201d. Try "
                                "your layout or area with the city, e.g. \u201cHSR "
                                "Layout Sector 2, Bengaluru\u201d, or a well-known "
                                "landmark near you.")
        label = _short_place(g["display_name"])
        return g["lat"], g["lng"], label, {
            "label": label, "source": "typed",
            "approximate": g.get("approximate", False),
            "broad": g.get("broad", False)}
    if lat is not None and lng is not None:
        return lat, lng, "Current location", {
            "label": "your current location", "source": "gps"}
    if required:
        raise HTTPException(400, "We need your location. Allow location access, "
                            "or type your area or a nearby landmark.")
    return None


def _safe_get(patient_id: str) -> dict:
    """Patient record for the ERROR log path: must never raise itself (the
    error being logged may well be that the store is down)."""
    try:
        return patient_store.get(patient_id) or {}
    except Exception:
        return {}


# A store outage is "try again shortly" (503), not a code bug (500). Redis
# backs sessions and patient records; Postgres backs the conversation history.
_STORE_DOWN = {"detail": "CareRoute is briefly unavailable. Please try again "
               "in a moment, and if this is an emergency call 112 now."}
if shared_state.USE_REDIS:
    import redis as _redis_mod

    @app.exception_handler(_redis_mod.exceptions.RedisError)
    def _redis_down(request, exc):
        print(f"[server] redis error: {exc}")
        return JSONResponse(status_code=503, content=_STORE_DOWN)
if shared_state.USE_POSTGRES:
    import psycopg as _psycopg_mod
    import psycopg_pool as _pool_mod

    @app.exception_handler(_psycopg_mod.OperationalError)
    @app.exception_handler(_pool_mod.PoolTimeout)
    def _postgres_down(request, exc):
        print(f"[server] postgres error: {exc}")
        return JSONResponse(status_code=503, content=_STORE_DOWN)


def _parse_meds(meds: str | None) -> list[str]:
    """Split a free-text meds field (commas or newlines) into a clean list."""
    return [m.strip() for m in (meds or "").replace("\n", ",").split(",") if m.strip()]


@app.get("/")
def home():
    return FileResponse("index.html")


# NOTE: this is a SYNC def, not async. run_agent makes blocking OpenAI calls,
# so FastAPI runs this in a threadpool — that keeps the blocking call off the
# event loop.
@app.post("/care")
def care(
    req: CareRequest,
    # Two gates run before the handler body. Order matters: rate_limit first so
    # even unauthenticated floods are throttled; then the (optional) API-key
    # check. Both are no-ops unless configured — see security.py.
    _rl: None = Depends(rate_limit),
    _auth: None = Depends(require_api_key),
):
    # Register the live user as a synthetic patient so get_patient_record works
    # unchanged. Each request now gets a UNIQUE id. The old hardcoded "LIVE" key
    # was a single shared slot in a module-level dict: two concurrent users
    # overwrote each other (last-writer-wins), so the agent could read the wrong
    # person's location. A per-request uuid isolates them; the finally-block
    # deletes it so the dict can't grow without bound. Name and medications are
    # OPTIONAL — empty fields fall back to the anonymous behaviour.
    lat, lng, area, loc_info = _resolve_location(
        req.location_text, req.lat, req.lng, required=True)
    patient_id = "LIVE-" + uuid.uuid4().hex[:12]
    meds_list = _parse_meds(req.meds)
    record = {
        "patient_id": patient_id,
        "name": (req.name or "").strip() or "Live user",
        "area": area,
        "lat": lat,
        "lng": lng,
        "history": [],
        "current_medications": meds_list,
    }
    # Stored (not just kept in a local) because the agent's tools read and
    # update it by id. Short TTL: a single-shot request never needs it after
    # it returns, and the TTL cleans up even if this replica dies mid-request.
    patient_store.put(patient_id, record, ttl=600)

    t0 = time.perf_counter()
    try:
        user_request = (
            f"Patient {patient_id} [record: {_record_summary(record)}] "
            f"reports these symptoms: {req.symptoms}. "
            f"Find the nearest appropriate specialists."
        )
        answer, trace = run_agent_traced(user_request)
        # may have been updated by the agent
        rec = patient_store.get(patient_id) or {}
        request_log.log_interaction(
            endpoint="/care", user_text=req.symptoms, answer=answer,
            meds=rec.get("current_medications", meds_list),
            name=rec.get("name", req.name),
            lat=rec.get("lat", req.lat), lng=rec.get("lng", req.lng),
            backend=tools._BACKEND,
            latency_ms=round((time.perf_counter() - t0) * 1000),
            trace=trace,
        )
        return {"answer": answer, "location": loc_info}
    except Exception as exc:
        rec = _safe_get(patient_id)
        request_log.log_interaction(
            endpoint="/care", user_text=req.symptoms, answer=None,
            meds=rec.get("current_medications", meds_list),
            name=rec.get("name", req.name),
            lat=rec.get("lat", req.lat), lng=rec.get("lng", req.lng),
            backend=tools._BACKEND,
            latency_ms=round((time.perf_counter() - t0) * 1000),
            status="error", error=exc,
        )
        raise
    finally:
        # Always clean up, even if run_agent raised — no leaked patient records.
        # (Best-effort: if the store is unreachable, the TTL cleans up.)
        try:
            patient_store.delete(patient_id)
        except Exception:
            pass


@app.post("/chat")
def chat(
    req: ChatRequest,
    _rl: None = Depends(rate_limit),
    _auth: None = Depends(require_api_key),
):
    """One turn of a conversation, run under that conversation's turn lock.

    Replicas are stateless, so two messages of ONE conversation (a double
    click, a client retry) could otherwise run at the same time on two
    replicas, both read the same checkpoint, and lose one turn. The lock makes
    turns of a conversation strictly sequential; different conversations still
    run fully in parallel. See sessions.py.
    """
    token = None
    if req.session_id:
        token = sessions.acquire_turn_lock(req.session_id)
        if token is None:
            raise HTTPException(
                status_code=409,
                detail="Still working on your previous message — please wait "
                       "for that answer before sending another.",
            )
    try:
        return _chat_turn(req)
    finally:
        if token:
            try:
                sessions.release_turn_lock(req.session_id, token)
            except Exception:
                pass  # the lock's own TTL frees it


def _chat_turn(req: ChatRequest):
    """Multi-turn sibling of /care.

    The FIRST call omits session_id and starts a conversation; the response
    returns a session_id the client sends back on every follow-up. The agent
    then remembers the thread (via the LangGraph checkpointer) and the SAME
    patient record persists, so medications/history added in later turns refine
    the routing instead of starting from scratch.

    Unlike /care, the patient record is NOT deleted after the request — it lives
    for the length of the session and is swept when the session expires.
    """
    # Opportunistic cleanup for the MEMORY backend. In Redis mode this is a
    # no-op: sessions and patient records expire by TTL on their own.
    for pid in sessions.sweep():
        patient_store.delete(pid)

    if not req.message.strip():
        raise HTTPException(
            status_code=400, detail="message must not be empty.")

    sess = sessions.get_session(req.session_id)

    if sess is None:
        # --- New conversation (turn 1) ---------------------------------------
        # Resolve BEFORE creating a session, so a place we can't find doesn't
        # leave an orphan session behind. No location at all is allowed: the
        # agent then takes the place from the message, or asks for it.
        resolved = _resolve_location(
            req.location_text, req.lat, req.lng, required=False)
        lat, lng, area, loc_info = resolved or (None, None, None, None)
        patient_id = "LIVE-" + uuid.uuid4().hex[:12]
        rec = {
            "patient_id": patient_id,
            "name": (req.name or "").strip() or "Live user",
            "area": area,
            "lat": lat,
            "lng": lng,
            "history": [],
            "current_medications": _parse_meds(req.meds),
        }
        # Record first, session second: a session must never point at a
        # record that doesn't exist yet.
        patient_store.put(patient_id, rec)
        session_id = sessions.create_session(patient_id)
        turn = 1
        user_message = (
            f"Patient {patient_id} [record: {_record_summary(rec)}] "
            f"reports these symptoms: {req.message}. "
            f"Find the nearest appropriate specialists."
        )
    else:
        # --- Follow-up turn ---------------------------------------------------
        session_id = req.session_id
        patient_id = sess["patient_id"]
        turn = sessions.next_turn(session_id)
        rec = patient_store.get(patient_id)
        if rec is None:
            # The record was swept while the client still held the id.
            raise HTTPException(
                status_code=409,
                detail="Session expired. Please start a new conversation.",
            )
        # Merge any newly supplied details so the agent can refine its routing.
        for med in _parse_meds(req.meds):
            if med not in rec["current_medications"]:
                rec["current_medications"].append(med)
        if (req.name or "").strip():
            rec["name"] = req.name.strip()
        # The page sends location on a follow-up only when the user CHANGED
        # it. (It used to resend GPS every turn, silently undoing a location
        # the agent had set from the conversation, e.g. "she is in Guwahati".)
        loc_info = None
        moved = _resolve_location(req.location_text, req.lat, req.lng,
                                  required=False)
        if moved:
            rec["lat"], rec["lng"], rec["area"], loc_info = moved
        # Write back (rec is a copy) — this also refreshes the record's TTL,
        # so it lives exactly as long as an active conversation.
        patient_store.put(patient_id, rec)
        user_message = (
            f"Patient {patient_id} (same conversation) [current record: "
            f"{_record_summary(rec)}] now says: {req.message}. "
            f"Treat this on its own merits — it may add detail to the earlier "
            f"complaint or raise a NEW need (e.g. wanting painkillers, which "
            f"means a pharmacy). Run the search that fits THIS message."
        )

    before = (rec.get("lat"), rec.get("lng"))
    t0 = time.perf_counter()
    try:
        answer, trace = continue_conversation_traced(
            user_message, thread_id=session_id)
        rec = patient_store.get(patient_id) or {}  # reflects agent updates
        loc_info = _moved_in_chat(rec, before) or loc_info
        request_log.log_interaction(
            endpoint="/chat", user_text=req.message, answer=answer,
            session_id=session_id, turn=turn,
            meds=rec.get("current_medications", _parse_meds(req.meds)),
            name=rec.get("name", req.name),
            lat=rec.get("lat", req.lat), lng=rec.get("lng", req.lng),
            backend=tools._BACKEND,
            latency_ms=round((time.perf_counter() - t0) * 1000),
            trace=trace,
        )
        return {"session_id": session_id, "answer": answer, "location": loc_info}
    except Exception as exc:
        rec = _safe_get(patient_id)
        request_log.log_interaction(
            endpoint="/chat", user_text=req.message, answer=None,
            session_id=session_id, turn=turn,
            meds=rec.get("current_medications", _parse_meds(req.meds)),
            name=rec.get("name", req.name),
            lat=rec.get("lat", req.lat), lng=rec.get("lng", req.lng),
            backend=tools._BACKEND,
            latency_ms=round((time.perf_counter() - t0) * 1000),
            status="error", error=exc,
        )
        raise


CODE_VERSION = "2026-10-stateless-replicas"
GIT_SHA = os.environ.get("GIT_SHA", "local-dev")


@app.get("/healthz")
def healthz():
    """Readiness probe: can THIS replica reach the shared stores?

    Point the Container Apps readiness probe here. A replica that can't reach
    Redis or Postgres is taken out of rotation instead of failing users'
    requests; liveness should stay on a cheap endpoint like /version, so a
    store outage doesn't make the platform restart every replica at once.
    """
    checks = {}
    ok = True
    if shared_state.USE_REDIS:
        try:
            shared_state.redis_client().ping()
            checks["redis"] = "ok"
        except Exception as e:
            checks["redis"], ok = f"error: {type(e).__name__}", False
    if shared_state.USE_POSTGRES:
        try:
            from agent_langgraph import _get_conversation_app
            pool = _get_conversation_app().checkpointer.conn
            with pool.connection(timeout=2) as conn:
                conn.execute("SELECT 1")
            checks["postgres"] = "ok"
        except Exception as e:
            checks["postgres"], ok = f"error: {type(e).__name__}", False
    body = {"status": "ok" if ok else "degraded",
            "mode": "shared" if shared_state.USE_REDIS else "memory",
            "checks": checks}
    return JSONResponse(status_code=200 if ok else 503, content=body)


@app.get("/version")
def version():
    return {"version": CODE_VERSION, "commit": GIT_SHA, "backend": tools._BACKEND}
