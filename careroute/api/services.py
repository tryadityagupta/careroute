"""
api/services.py — what one /care request or one /chat turn does, end to end.

Kept out of the route functions so the HTTP layer stays thin and these can be
unit-tested without a server. Order of operations matters and is commented.
"""

from __future__ import annotations

import time

from fastapi import HTTPException

from careroute.agent.graph import CareRouteAgent
from careroute.api.location import LocationResolver
from careroute.api.schemas import CareRequest, ChatRequest
from careroute.domain.patients import PatientRepository
from careroute.observability.interaction_log import InteractionLogger
from careroute.storage.sessions import SessionStore


class TurnPromptBuilder:
    """The user-message wrapper the model sees.

    SPEED FIX: the server already knows the location and meds, so they go
    inline as [record: ...] instead of costing a get_patient_record round trip.
    The mock LLM parses this exact wording — change both together.
    """

    @staticmethod
    def record_summary(rec: dict) -> str:
        meds = ", ".join(rec.get("current_medications") or []) or "none"
        if rec.get("lat") is None or rec.get("lng") is None:
            where = "location=UNKNOWN (browser location was not shared)"
        else:
            where = f"lat={rec.get('lat')}, lng={rec.get('lng')}, area={rec.get('area')}"
        return f"name={rec.get('name')}, {where}, current_medications={meds}"

    def first_turn(self, rec: dict, text: str) -> str:
        return (f"Patient {rec['patient_id']} [record: {self.record_summary(rec)}] "
                f"reports these symptoms: {text}. "
                f"Find the nearest appropriate specialists.")

    def follow_up(self, rec: dict, text: str) -> str:
        return (f"Patient {rec['patient_id']} (same conversation) [current record: "
                f"{self.record_summary(rec)}] now says: {text}. "
                f"Treat this on its own merits — it may add detail to the earlier "
                f"complaint or raise a NEW need (e.g. wanting painkillers, which "
                f"means a pharmacy). Run the search that fits THIS message.")


class _BaseService:
    def __init__(self, agent: CareRouteAgent, patients: PatientRepository,
                 locations: LocationResolver, log: InteractionLogger, backend_name: str):
        self.agent = agent
        self.patients = patients
        self.locations = locations
        self.log = log
        self.backend_name = backend_name
        self.prompts = TurnPromptBuilder()

    def _log(self, endpoint, req_text, req, rec, t0, **extra):
        self.log.log(endpoint=endpoint, user_text=req_text,
                     meds=rec.get("current_medications", self.patients.parse_meds(req.meds)),
                     name=rec.get("name", req.name),
                     lat=rec.get("lat", req.lat), lng=rec.get("lng", req.lng),
                     backend=self.backend_name,
                     latency_ms=round((time.perf_counter() - t0) * 1000), **extra)


class CareService(_BaseService):
    """/care: one self-contained request."""

    RECORD_TTL_S = 600     # a single-shot record is never needed after it returns

    def handle(self, req: CareRequest) -> dict:
        loc = self.locations.resolve(req.location_text, req.lat, req.lng, required=True)
        # A UNIQUE id per request: the old shared "LIVE" slot let two concurrent
        # users overwrite each other's location.
        record = self.patients.new_live_record(name=req.name, lat=loc.lat, lng=loc.lng,
                                               area=loc.area, meds=req.meds)
        pid = record["patient_id"]
        # Stored (not just a local) because the agent's tools read and update it.
        self.patients.save(record, ttl=self.RECORD_TTL_S)
        t0 = time.perf_counter()
        try:
            answer, trace = self.agent.run_single(self.prompts.first_turn(record, req.symptoms),
                                                  patient_id=pid)
            rec = self.patients.get(pid) or {}            # may have been updated
            self._log("/care", req.symptoms, req, rec, t0, answer=answer, trace=trace)
            return {"answer": answer, "location": loc.info}
        except Exception as exc:
            self._log("/care", req.symptoms, req, self.patients.safe_get(pid), t0,
                      answer=None, status="error", error=exc)
            raise
        finally:
            try:
                self.patients.delete(pid)    # best-effort; the TTL cleans up too
            except Exception:
                pass


class ChatService(_BaseService):
    """/chat: one turn of a conversation, under that conversation's turn lock."""

    def __init__(self, *args, sessions: SessionStore, **kwargs):
        super().__init__(*args, **kwargs)
        self.sessions = sessions

    def handle(self, req: ChatRequest) -> dict:
        # Two turns of ONE conversation (double click, retry) must not run at
        # once on two replicas: both would read the same checkpoint and lose a
        # turn. Different conversations stay fully parallel.
        token = None
        if req.session_id:
            token = self.sessions.acquire_turn_lock(req.session_id)
            if token is None:
                raise HTTPException(409, "Still working on your previous message — "
                                         "please wait for that answer before sending another.")
        try:
            return self._turn(req)
        finally:
            if token:
                try:
                    self.sessions.release_turn_lock(req.session_id, token)
                except Exception:
                    pass                     # the lock's own TTL frees it

    @staticmethod
    def _moved_in_chat(rec: dict, before: tuple):
        """The AGENT moved the patient this turn (the user named a place):
        tell the page, so its "Near: ..." status stays truthful."""
        after = (rec.get("lat"), rec.get("lng"))
        if after != before and after[0] is not None:
            return {"label": rec.get("area") or "the place you mentioned", "source": "chat"}
        return None

    def _turn(self, req: ChatRequest) -> dict:
        # Memory backend housekeeping; a no-op with Redis (TTLs do it).
        for pid in self.sessions.sweep():
            self.patients.delete(pid)
        if not req.message.strip():
            raise HTTPException(400, "message must not be empty.")

        sess = self.sessions.get_session(req.session_id)
        if sess is None:
            # --- turn 1 ------------------------------------------------------
            # Resolve BEFORE creating anything, so an unknown place leaves no
            # orphan session. No location at all is allowed: the agent takes it
            # from the message, or asks.
            loc = self.locations.resolve(req.location_text, req.lat, req.lng, required=False)
            rec = self.patients.new_live_record(
                name=req.name, lat=loc.lat if loc else None, lng=loc.lng if loc else None,
                area=loc.area if loc else None, meds=req.meds)
            loc_info = loc.info if loc else None
            # Record first, session second: a session must never point at a
            # record that doesn't exist yet.
            self.patients.save(rec)
            session_id = self.sessions.create_session(rec["patient_id"])
            turn, new_conversation = 1, True
            user_message = self.prompts.first_turn(rec, req.message)
        else:
            # --- follow-up -----------------------------------------------------
            session_id = req.session_id
            turn, new_conversation = self.sessions.next_turn(session_id), False
            rec = self.patients.get(sess["patient_id"])
            if rec is None:
                raise HTTPException(409, "Session expired. Please start a new conversation.")
            for med in self.patients.parse_meds(req.meds):
                if med not in rec["current_medications"]:
                    rec["current_medications"].append(med)
            if (req.name or "").strip():
                rec["name"] = req.name.strip()
            # The page sends a location on a follow-up only when the user
            # CHANGED it (resending GPS every turn undid locations the agent set).
            loc_info = None
            moved = self.locations.resolve(req.location_text, req.lat, req.lng, required=False)
            if moved:
                rec["lat"], rec["lng"], rec["area"] = moved.lat, moved.lng, moved.area
                loc_info = moved.info
            self.patients.save(rec)          # write back; also refreshes the TTL
            user_message = self.prompts.follow_up(rec, req.message)

        pid = rec["patient_id"]
        before = (rec.get("lat"), rec.get("lng"))
        t0 = time.perf_counter()
        try:
            answer, trace = self.agent.run_turn(user_message, thread_id=session_id,
                                                patient_id=pid,
                                                new_conversation=new_conversation)
            rec = self.patients.get(pid) or {}             # reflects agent updates
            loc_info = self._moved_in_chat(rec, before) or loc_info
            self._log("/chat", req.message, req, rec, t0, answer=answer,
                      session_id=session_id, turn=turn, trace=trace)
            return {"session_id": session_id, "answer": answer, "location": loc_info}
        except Exception as exc:
            self._log("/chat", req.message, req, self.patients.safe_get(pid), t0,
                      answer=None, session_id=session_id, turn=turn,
                      status="error", error=exc)
            raise
