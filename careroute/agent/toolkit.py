"""
agent/toolkit.py — the tools exactly as the MODEL sees them.

With @tool, the DOCSTRING is the description the model reads and each
Annotated[...] becomes a parameter description. These texts are behaviour:
drop "Call this FIRST" or "double radius_m on retry, up to 30000" and the
model gets measurably worse, because it acts on what it is told. A one-line
docstring is a silent regression — tests/test_agent_graph.py pins them.

The business logic is in domain/tools.py (CareTools); these are thin adapters.

PATIENT BINDING (security): patient_id arrives from the MODEL, and the model
follows instructions in user text — "show me patient P002's record" used to
work. A turn now carries its patient in config["configurable"]["patient_id"],
and record tools refuse any other id. LangChain injects `config` and keeps it
out of the schema, so the model never sees or controls it. Callers that bind
nothing (the CLI demo) keep the old, unrestricted behaviour.
"""

from __future__ import annotations

import inspect
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from careroute.domain.tools import CareTools


def _bound_patient(config: RunnableConfig | None) -> str | None:
    return ((config or {}).get("configurable") or {}).get("patient_id")


def _refuse(patient_id: str) -> dict:
    return {"error": (f"Access to patient {patient_id} is not allowed in this "
                      "conversation. Only the current patient's record (the id "
                      "given in the message) can be read or updated."),
            "error_type": "forbidden"}


class LangChainToolkit:
    def __init__(self, care: CareTools):
        self.care = care
        self.tools: list[BaseTool] = self._build()

    def by_name(self, name: str) -> BaseTool:
        return next(t for t in self.tools if t.name == name)

    def _build(self) -> list[BaseTool]:
        care = self.care

        @tool
        async def get_patient_record(
            patient_id: Annotated[str, "The patient's ID, e.g. 'P001'"],
            config: RunnableConfig,
        ) -> dict:
            """Retrieve a patient's clinical record (location + history) by patient
            ID. Call this FIRST to get the patient's coordinates before searching for
            providers."""
            bound = _bound_patient(config)
            if bound and patient_id != bound:
                return _refuse(patient_id)
            return await care.get_patient_record(patient_id)

        @tool
        async def geocode_place(
            place: Annotated[str, "A place name to resolve, e.g. 'Guwahati'"],
        ) -> dict:
            """Resolve a place NAME to coordinates. Call this whenever the user gives a
            location by name (e.g. "she is in Guwahati") rather than trusting the
            patient's stored coordinates. Feed the returned lat/lng into the search
            tools so the search happens THERE. Never guess coordinates, and never claim
            a result is in a city you did not resolve with this tool."""
            return await care.geocode_place(place)

        @tool
        async def update_patient_record(
            patient_id: Annotated[str, "The patient's ID"],
            config: RunnableConfig,
            name: Annotated[str | None, "Patient name, if the user stated it"] = None,
            medications: Annotated[str | None,
                                   "Comma-separated meds the user mentioned"] = None,
            lat: Annotated[float | None, "New latitude, from geocode_place"] = None,
            lng: Annotated[float | None, "New longitude, from geocode_place"] = None,
            area: Annotated[str | None,
                            "Human name of the location, e.g. 'Guwahati'"] = None,
        ) -> dict:
            """Persist details the user states onto the patient record: their NAME,
            MEDICATIONS, or a corrected LOCATION. For a named location, call
            geocode_place first, then pass its lat/lng here plus area=<place>. Only the
            fields you pass change."""
            bound = _bound_patient(config)
            if bound and patient_id != bound:
                return _refuse(patient_id)
            return await care.update_patient_record(patient_id=patient_id, name=name,
                                              medications=medications, lat=lat,
                                              lng=lng, area=area)

        @tool
        async def find_providers(
            specialty: Annotated[str, "Medical specialty, e.g. 'Cardiology', 'Orthopedics'"],
            patient_lat: Annotated[float, "Patient latitude"],
            patient_lng: Annotated[float, "Patient longitude"],
            k: Annotated[int, "How many providers to return (default 3)"] = 3,
            radius_m: Annotated[int, (
                "Search radius in metres (default 8000). If no match was found, "
                "double it on retry, up to a maximum of 30000."
            )] = 8000,
        ) -> list | dict:
            """Find the k nearest providers of a given medical specialty to the
            patient's location. Call this AFTER you know the patient's coordinates
            and have decided the specialty the condition requires. If it returns
            match_found=false, follow the hint in the result: retry with a larger
            radius_m, or switch to one of the available_specialties."""
            return await care.find_providers(specialty=specialty, patient_lat=patient_lat,
                                       patient_lng=patient_lng, k=k, radius_m=radius_m)

        @tool
        async def find_general_facilities(
            patient_lat: Annotated[float, "Patient latitude"],
            patient_lng: Annotated[float, "Patient longitude"],
            k: Annotated[int, "How many facilities to return (default 3)"] = 3,
            radius_m: Annotated[int, "Search radius in metres (default 8000)"] = 8000,
        ) -> dict:
            """Nearest healthcare facilities of ANY type — these are NOT specialists.
            Only call this AFTER find_providers has returned match_found=false and
            you have exhausted your radius retries. Everything it returns must be
            presented as a general option, never as a specialist match."""
            return await care.find_general_facilities(patient_lat=patient_lat,
                                                patient_lng=patient_lng, k=k,
                                                radius_m=radius_m)

        @tool
        async def find_pharmacies(
            patient_lat: Annotated[float, "Patient latitude"],
            patient_lng: Annotated[float, "Patient longitude"],
            k: Annotated[int, "How many pharmacies to return (default 3)"] = 3,
            radius_m: Annotated[int, "Search radius in metres (default 8000)"] = 8000,
        ) -> dict:
            """Nearest PHARMACIES / chemists — where a user goes to OBTAIN medicines.
            Call this when the user wants to buy or pick up a medicine or OTC drug
            (painkillers, antacids, ORS, cold medicine, etc.). You are routing them to a
            provider, NOT prescribing: do not recommend a specific medicine, dose, or
            brand. Returns match_found=false when no pharmacy is mapped nearby — if so,
            say that plainly and never substitute a hospital or clinic for a pharmacy."""
            return await care.find_pharmacies(patient_lat=patient_lat, patient_lng=patient_lng,
                                        k=k, radius_m=radius_m)

        @tool
        async def get_emergency_help(
            patient_lat: Annotated[float, "Patient latitude"],
            patient_lng: Annotated[float, "Patient longitude"],
        ) -> dict:
            """Return the LOCAL emergency number to call NOW plus the nearest hospitals
            (which have emergency departments). Call this FIRST for any medical
            emergency — seizure, stroke signs, major trauma or a serious accident, heavy
            bleeding, chest pain with cardiac features, fainting, or trouble breathing —
            before any specialty search, and lead the answer with the number."""
            return await care.get_emergency_help(patient_lat=patient_lat, patient_lng=patient_lng)

        tools = [get_patient_record, update_patient_record, geocode_place,
                 find_providers, find_general_facilities, find_pharmacies,
                 get_emergency_help]
        # @tool keeps the docstring's source indentation; strip it so the
        # model isn't sent (and billed for) leading whitespace on every line.
        for t in tools:
            t.description = inspect.cleandoc(t.description)
        return tools
