"""
mock_llm/policy.py — what the fake model DECIDES. Pure: transcript in, next
step out, no state, so any number of mock replicas answer identically.

Two policies (MOCK_LLM_MODE):
  AgentPolicy  behaves like a careful tool-calling model: emergency first,
               pharmacy for "buy medicine", geocode a named place, specialty
               search widening 8 -> 16 -> 30 km, one retry on a directory
               error, and a final answer written ONLY from tool results.
               A turn costs 2-4 model calls, like the real thing.
  EchoPolicy   the statelessness probe: "turns_seen=N | meds=..." so
               tests/test_stateless.py can see whether history survived.

The server's message wrapper (careroute/api/services.py TurnPromptBuilder)
is parsed here — change both together.
"""

from __future__ import annotations

import json
import re

Step = tuple[str, object]       # ("tools", [(name, args), ...]) | ("answer", text)


def content_of(m: dict) -> str:
    c = m.get("content") or ""
    if isinstance(c, list):                               # content-parts format
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c


def is_error(result) -> bool:
    return (isinstance(result, dict) and "error" in result) or (
        isinstance(result, str) and result.startswith("Error"))


class EchoPolicy:
    _MEDS = re.compile(r"current_medications=([^\]]*)\]")

    def decide(self, messages: list[dict], tool_names: set[str]) -> Step:
        users = [m for m in messages if m.get("role") == "user"]
        meds = self._MEDS.search(content_of(users[-1]) if users else "")
        return "answer", (f"turns_seen={len(users)} | "
                          f"meds={meds.group(1).strip() if meds else '?'}")


class AgentPolicy:
    # --- reading the server's wrapper -------------------------------------------
    _PATIENT = re.compile(r"Patient (\S+)")
    _LATLNG = re.compile(r"lat=(-?\d+(?:\.\d+)?), lng=(-?\d+(?:\.\d+)?)")
    _COMPLAINT = re.compile(r"(?:reports these symptoms:|now says:)\s*(.*?)\.\s*"
                            r"(?:Find the nearest|Treat this on its own)", re.S)

    # --- "the model's judgement": keyword triage. Deliberately BROADER than any
    # deterministic pre-pass step 5 builds, so the mock keeps exercising the
    # LLM path for complaints a rule table would not cover.
    EMERGENCY = re.compile(
        r"chest pain|seizure|\bfits?\b|unconscious|fainted|passed out|"
        r"not breathing|can'?t breathe|trouble breathing|stroke|face droop|"
        r"slurred|heavy bleeding|severe bleeding|accident|overdose|suicid", re.I)
    PHARMACY = re.compile(
        r"pharmac|chemist|medicine|tablets?|painkiller|paracetamol|antacid|"
        r"\bors\b|cough syrup|\bbuy\b", re.I)
    SPECIALTIES = [
        (r"\b(child|kid|baby|infant|toddler|son|daughter)\b", "Pediatrics"),
        (r"pregnan|period|menstrua|pcos|gyn", "Gynecology"),
        (r"palpitation|heart|blood pressure|\bbp\b|cardi", "Cardiology"),
        (r"knee|back pain|joint pain|fracture|sprain|bone|ortho", "Orthopedics"),
        (r"skin|rash|acne|itch|eczema", "Dermatology"),
        (r"anxi|depress|panic|insomnia|can'?t sleep|stress", "Psychiatry"),
        (r"\bear\b|throat|sinus|tonsil|nose bleed|blocked nose", "ENT"),
        (r"\beye|vision|blurr", "Ophthalmology"),
        (r"tooth|teeth|gum|dental", "Dentistry"),
        (r"stomach|acidity|diarrh|vomit|constipat|abdomen|abdominal", "Gastroenterology"),
        (r"urin|kidney stone", "Urology"),
        (r"headache|migraine|numb|dizz|tingling", "Neurology"),
        (r"cough|asthma|wheez|breathless", "Pulmonology"),
        (r"diabet|sugar|thyroid", "Endocrinology"),
        (r"arthritis|stiff joints|lupus|autoimmune", "Rheumatology"),
    ]
    _PLACE_EXPLICIT = re.compile(
        r"\b(?:i am|i'm|she is|he is|she's|he's|we are|we're|staying|currently)\s+"
        r"(?:in|at|near)\s+([A-Za-z][\w .'-]{2,40}?)(?=[,.!?]|\s+and\b|$)", re.I)
    _PLACE_ANY = re.compile(r"\b(?:in|at|near)\s+([A-Z][\w .'-]{2,40}?)(?=[,.!?]|\s+and\b|$)")

    # --- helpers -----------------------------------------------------------------
    @classmethod
    def specialty_for(cls, text: str) -> str:
        for pat, spec in cls.SPECIALTIES:
            if re.search(pat, text, re.I):
                return spec
        return "General Medicine"

    @staticmethod
    def read_turn(messages: list[dict]):
        """(last user text, [(tool_name, args, result), ...] since it)."""
        idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
        user = content_of(messages[idx]) if idx >= 0 else ""
        names, calls = {}, []
        for m in messages[idx + 1:]:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                names[tc.get("id")] = (fn.get("name"), args)
            if m.get("role") == "tool":
                name, args = names.get(m.get("tool_call_id"), (None, {}))
                try:
                    result = json.loads(content_of(m))
                except ValueError:
                    result = content_of(m)          # e.g. ToolNode "Error: ..."
                calls.append((name, args, result))
        return user, calls

    @staticmethod
    def _dist(f: dict) -> str:
        d = f.get("distance_km")
        s = f"{d} km" if d is not None else "distance unknown"
        if f.get("distance_type") == "road":
            mins = f.get("drive_min_no_traffic")
            s += " by road" + (f", ~{mins} min without traffic" if mins is not None else "")
        elif f.get("distance_type") == "straight_line":
            s += " (straight-line)"
        return s

    @classmethod
    def numbered(cls, items) -> str:
        return "\n".join(f"{i}. {f['name']} — {cls._dist(f)}"
                         for i, f in enumerate(items, 1) if isinstance(f, dict))

    # --- the decision --------------------------------------------------------------
    def decide(self, messages: list[dict], tool_names: set[str]) -> Step:
        user, calls = self.read_turn(messages)
        m = self._COMPLAINT.search(user)
        complaint = (m.group(1) if m else user).strip()
        pid = (self._PATIENT.search(user) or [None, None])[1]
        ll = self._LATLNG.search(user)
        lat, lng = (float(ll.group(1)), float(ll.group(2))) if ll else (None, None)
        done = [c[0] for c in calls]

        # 1. A place named in the message: geocode it, then search THERE.
        place_m = self._PLACE_EXPLICIT.search(complaint) or (
            self._PLACE_ANY.search(complaint) if lat is None else None)
        place = place_m.group(1).strip() if place_m else None
        pending_update = None
        if place and "geocode_place" in tool_names:
            if "geocode_place" not in done:
                return "tools", [("geocode_place", {"place": place})]
            geo = next(r for n, _, r in calls if n == "geocode_place")
            if isinstance(geo, dict) and "lat" in geo:
                lat, lng = geo["lat"], geo["lng"]
                if pid and "update_patient_record" not in done:
                    pending_update = ("update_patient_record",
                                      {"patient_id": pid, "lat": lat, "lng": lng, "area": place})
            elif lat is None and is_error(geo):
                # Lookup FAILED: says nothing about whether the place exists.
                return "answer", ("I can't look up place names right now. Please "
                                  "share your location, or try again shortly. If "
                                  "this is urgent, call 112.")
            elif lat is None:
                return "answer", (f"I couldn't find \"{place}\" on the map. Could you "
                                  "give a nearby area, landmark or city?")

        def search(name, args):
            """Batch the record update with the search, as models do."""
            return "tools", ([pending_update] if pending_update else []) + [(name, args)]

        # 2. No location: emergencies still get the number, else ask.
        if lat is None:
            if self.EMERGENCY.search(complaint):
                return "answer", ("This may be a medical emergency. Call 112 now "
                                  "(ambulance: 108). Then tell me where you are so I "
                                  "can find the nearest hospital.")
            return "answer", ("Where are you right now? Share your location or "
                              "tell me the area, and I'll find the nearest options.")
        here = {"patient_lat": lat, "patient_lng": lng}

        # 3. Emergency: number first, hospitals second.
        if self.EMERGENCY.search(complaint):
            res = [r for n, _, r in calls if n == "get_emergency_help"]
            if not res:
                return search("get_emergency_help", here)
            r = res[-1] if isinstance(res[-1], dict) else {}
            number = r.get("emergency_number", "112")
            text = (f"This may be a medical emergency. Call {number} now — an "
                    "ambulance can start care on the way.")
            if r.get("nearest_hospitals"):
                text += ("\n\nNearest hospitals with emergency care:\n"
                         + self.numbered(r["nearest_hospitals"]))
            return "answer", text

        # 4. Obtaining medicine: a pharmacy, never a clinic.
        if self.PHARMACY.search(complaint):
            res = [r for n, _, r in calls if n == "find_pharmacies"]
            if not res:
                return search("find_pharmacies", here)
            r = res[-1]
            if is_error(r):
                return "answer", ("The pharmacy directory is temporarily unreachable. "
                                  "Please try again shortly.")
            if isinstance(r, dict) and r.get("pharmacies"):
                return "answer", ("Nearest pharmacies:\n" + self.numbered(r["pharmacies"])
                                  + "\n\nA pharmacist can advise on over-the-counter "
                                  "options; I can't recommend a specific medicine.")
            return "answer", ("I couldn't find a pharmacy mapped near you. A wider "
                              "search or a local check may help.")

        # 5. Specialty search, widening on a miss, one retry on a directory error.
        spec = self.specialty_for(complaint)
        res = [(a, r) for n, a, r in calls if n == "find_providers"]
        if not res:
            return search("find_providers", {"specialty": spec, **here,
                                             "k": 3, "radius_m": 8000})
        args, r = res[-1]
        if is_error(r):
            if sum(is_error(x) for _, x in res) < 2:
                return search("find_providers", dict(args))
            return "answer", ("The provider directory is temporarily unreachable, so "
                              "I can't confirm who is nearby right now. Please try "
                              "again shortly. If symptoms are severe, call 112.")
        if isinstance(r, list) and r:
            return "answer", (f"Nearest {spec} options:\n" + self.numbered(r)
                              + "\n\nDistances are without traffic.")
        radius = int(args.get("radius_m", 8000))
        if radius < 30000:
            return search("find_providers", {**args, "radius_m": min(radius * 2, 30000)})
        alts = r.get("general_alternatives") if isinstance(r, dict) else None
        if alts:
            return "answer", (f"I couldn't find a confirmed {spec} specialist within "
                              "30 km. Nearby general options (not specialists):\n"
                              + self.numbered(alts))
        return "answer", (f"I couldn't find a {spec} specialist nearby. A general "
                          "physician can assess you and refer you.")


def policy_for(mode: str):
    return EchoPolicy() if mode == "echo" else AgentPolicy()
