"""
domain/patients.py — one place to read and edit a patient record.

Reads check the LIVE store first (live users, and any edits the agent made),
then the STATIC demo data. Writes always go to the live store: the static
reference data is never mutated, so replicas can't drift apart.
"""

from __future__ import annotations

import copy
import uuid

from careroute.storage.patient_store import PatientStore


class PatientRepository:
    def __init__(self, static: dict, live: PatientStore):
        self._static = static          # read-only, identical on every replica
        self.live = live

    @staticmethod
    def new_live_id() -> str:
        return "LIVE-" + uuid.uuid4().hex[:12]

    @staticmethod
    def parse_meds(meds) -> list[str]:
        """'a, b\\nc' -> ['a', 'b', 'c']."""
        return [m.strip() for m in (meds or "").replace("\n", ",").split(",") if m.strip()]

    def new_live_record(self, *, name: str | None, lat, lng, area, meds) -> dict:
        return {"patient_id": self.new_live_id(),
                "name": (name or "").strip() or "Live user",
                "area": area, "lat": lat, "lng": lng, "history": [],
                "current_medications": self.parse_meds(meds)}

    async def get(self, patient_id: str) -> dict | None:
        rec = await self.live.get(patient_id)
        if rec is None:
            rec = self._static.get(patient_id)
        return rec

    async def safe_get(self, patient_id: str) -> dict:
        """For ERROR paths: never raises (the error may be that the store is down)."""
        try:
            return await self.get(patient_id) or {}
        except Exception:
            return {}

    async def save(self, record: dict, ttl: int | None = None) -> None:
        await self.live.put(record["patient_id"], record, ttl=ttl)

    async def delete(self, patient_id: str) -> None:
        await self.live.delete(patient_id)

    async def update(self, patient_id: str, *, name=None, medications=None,
               lat=None, lng=None, area=None) -> dict | None:
        """Copy-on-write edit; only the fields given change. None if unknown."""
        rec = await self.get(patient_id)
        if rec is None:
            return None
        rec = copy.deepcopy(rec)
        if name:
            rec["name"] = name.strip()
        if medications is not None:
            meds = ([m.strip() for m in medications.split(",")]
                    if isinstance(medications, str) else list(medications))
            existing = rec.setdefault("current_medications", [])
            for m in meds:
                if m and m not in existing:
                    existing.append(m)
        if lat is not None and lng is not None:
            rec["lat"], rec["lng"] = float(lat), float(lng)
        if area:
            rec["area"] = area.strip()
        await self.live.put(patient_id, rec)
        return rec
