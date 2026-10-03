"""
storage/patient_store.py — LIVE patient records that must survive across
requests and replicas.

Two kinds of patient record are kept apart on purpose:
  * STATIC demo patients (P001..., Synthea) — read-only reference data,
    loaded once per replica by seed_data.py. Never written.
  * LIVE records (LIVE-xxxx, plus edits the agent makes to a demo patient) —
    stored HERE, in Redis with a TTL, so every replica sees them.

Copy-on-read / write-back in BOTH backends: get() returns a copy and nothing
changes until put(). Redis behaves that way anyway; making memory behave the
same means "mutated it but forgot to save" bugs fail offline, not only in prod.
"""

from __future__ import annotations

import copy
import json
import time
from abc import ABC, abstractmethod

from careroute.storage.redis_client import RedisConnection


class PatientStore(ABC):
    def __init__(self, ttl_s: int):
        self.ttl_s = ttl_s            # records expire with their conversation

    @abstractmethod
    async def get(self, patient_id: str) -> dict | None: ...

    @abstractmethod
    async def put(self, patient_id: str, record: dict, ttl: int | None = None) -> None: ...

    @abstractmethod
    async def delete(self, patient_id: str) -> None: ...


class MemoryPatientStore(PatientStore):
    def __init__(self, ttl_s: int = 1800):
        super().__init__(ttl_s)
        self._recs: dict[str, tuple[dict, float]] = {}

    async def get(self, patient_id):
        item = self._recs.get(patient_id)
        if item is None:
            return None
        rec, expires = item
        if expires < time.monotonic():
            self._recs.pop(patient_id, None)
            return None
        return copy.deepcopy(rec)

    async def put(self, patient_id, record, ttl=None):
        self._recs[patient_id] = (copy.deepcopy(record),
                                  time.monotonic() + (ttl or self.ttl_s))

    async def delete(self, patient_id):
        self._recs.pop(patient_id, None)


class RedisPatientStore(PatientStore):
    def __init__(self, redis: RedisConnection, ttl_s: int = 1800):
        super().__init__(ttl_s)
        self.redis = redis

    def _k(self, pid: str) -> str:
        return self.redis.key("patient", pid)

    async def get(self, patient_id):
        raw = await self.redis.client.get(self._k(patient_id))
        return json.loads(raw) if raw else None

    async def put(self, patient_id, record, ttl=None):
        await self.redis.client.set(self._k(patient_id), json.dumps(record), ex=ttl or self.ttl_s)

    async def delete(self, patient_id):
        await self.redis.client.delete(self._k(patient_id))
