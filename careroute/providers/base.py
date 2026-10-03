"""
providers/base.py — the contract every provider directory implements.

The agent never knows which backend is live: tools call these three methods
and get the same result SHAPES back from every implementation:

  find_providers           list[dict] of confirmed specialty matches, or a
                           dict with match_found=False + a recovery hint
  find_general_facilities  {"disclaimer", "facilities": [...]} — explicitly
                           NOT specialists — or match_found=False
  find_pharmacies          {"disclaimer", "pharmacies": [...]} or
                           match_found=False (never a hospital in disguise)

Any of them may instead return {"error": ..., "error_type": ...} when the
directory itself is unreachable — a failed lookup is not an empty lookup.

All three are coroutines: a backend may wait on a database or an HTTP API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProviderDirectory(ABC):
    #: Shown at /version and in logs, e.g. "openstreetmap".
    backend_name: str = "abstract"

    @abstractmethod
    async def find_providers(self, specialty: str, patient_lat: float, patient_lng: float,
                       k: int = 3, radius_m: int = 8000) -> list | dict: ...

    @abstractmethod
    async def find_general_facilities(self, patient_lat: float, patient_lng: float,
                                k: int = 3, radius_m: int = 8000) -> dict: ...

    @abstractmethod
    async def find_pharmacies(self, patient_lat: float, patient_lng: float,
                        k: int = 3, radius_m: int = 8000) -> dict: ...
