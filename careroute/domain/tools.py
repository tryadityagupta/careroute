"""
domain/tools.py — the operations the agent is allowed to call.

KEY IDEA: the LLM never touches data directly. It DECIDES which of these to
call and with what arguments; our code runs it and hands the result back.
That hand-off is the whole mechanic of tool calling.

This class is framework-free: plain Python in, plain dicts/lists out. The
LangChain adapters that the model actually sees (names, docstrings, argument
descriptions) live in agent/toolkit.py. The provider backend is injected, so
the agent can't tell whether the data came from JSON, OpenStreetMap or Google.
"""

from __future__ import annotations

from careroute.domain.emergency import EmergencyService
from careroute.domain.patients import PatientRepository
from careroute.maps.geocoding import NominatimGeocoder
from careroute.providers.base import ProviderDirectory


class CareTools:
    def __init__(self, patients: PatientRepository, directory: ProviderDirectory,
                 geocoder: NominatimGeocoder, emergency: EmergencyService):
        self.patients = patients
        self.directory = directory
        self.geocoder = geocoder
        self.emergency = emergency

    @property
    def backend_name(self) -> str:
        return self.directory.backend_name

    def get_patient_record(self, patient_id: str) -> dict:
        record = self.patients.get(patient_id)
        if record is None:
            return {"error": f"No patient found with id {patient_id}"}
        return record

    def update_patient_record(self, patient_id: str, name: str | None = None,
                              medications=None, lat: float | None = None,
                              lng: float | None = None, area: str | None = None) -> dict:
        rec = self.patients.update(patient_id, name=name, medications=medications,
                                   lat=lat, lng=lng, area=area)
        if rec is None:
            return {"error": f"No patient found with id {patient_id}"}
        return rec

    def geocode_place(self, place: str) -> dict:
        return self.geocoder.geocode(place)

    def find_providers(self, specialty: str, patient_lat: float, patient_lng: float,
                       k: int = 3, radius_m: int = 8000):
        return self.directory.find_providers(specialty, patient_lat, patient_lng, k, radius_m)

    def find_general_facilities(self, patient_lat: float, patient_lng: float,
                                k: int = 3, radius_m: int = 8000):
        return self.directory.find_general_facilities(patient_lat, patient_lng, k, radius_m)

    def find_pharmacies(self, patient_lat: float, patient_lng: float,
                        k: int = 3, radius_m: int = 8000):
        return self.directory.find_pharmacies(patient_lat, patient_lng, k, radius_m)

    def get_emergency_help(self, patient_lat: float, patient_lng: float) -> dict:
        return self.emergency.get_help(patient_lat, patient_lng)
