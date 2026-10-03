"""
agent/guard.py — the deterministic last line of defence on the final answer.

The model has presented facilities as specialists that no tool confirmed, and
has forgotten the emergency number on a follow-up. Prompt rules did not hold,
so these checks run in CODE. Code beats prompt.

Two checks:
  check_recommendations  a numbered provider list must be backed by a tool
  remind_emergency       once a conversation had an emergency, every later
                         answer carries the number
"""

from __future__ import annotations

import re

_LIST_ITEM = re.compile(r"^\s*\d+[.)]\s+\S", re.M)
_LIST_LINE = re.compile(r"^\s*\d+[.)]\s+(.+)$", re.M)
_DIGITS = re.compile(r"\d{2,}")

DIRECTORY_DOWN_PREFIX = (
    "The provider directory could not be reached, so this is NOT a "
    "confirmed result \u2014 matching specialists may well exist nearby. "
    "Please try again shortly, and seek emergency care now if your "
    "symptoms are severe.\n\n")
UNVERIFIED_PREFIX = "No verified specialist match was found nearby for this condition.\n\n"
WITHHELD = (
    "No verified specialist match was found nearby for this condition, "
    "so I can't recommend specific providers. Consider widening the "
    "search area, or seeing a general physician who can refer you.\n\n"
    "(The model attempted to list providers that no tool confirmed; "
    "that response was withheld.)")


class AnswerGuard:
    @staticmethod
    def _items_all_known(answer: str, known: set[str]) -> bool:
        """Every numbered line names a provider some tool confirmed."""
        lowered = [n.lower() for n in known]
        lines = _LIST_LINE.findall(answer)
        return bool(lines) and all(any(n in line.lower() for n in lowered) for line in lines)

    def check_recommendations(self, answer: str, *, confirmed_this_turn: set[str],
                              confirmed_ever: set[str], offered: set[str],
                              failed: list | tuple = ()) -> str:
        if not answer or not _LIST_ITEM.search(answer):
            return answer                       # not a recommendation list
        if confirmed_this_turn:
            return answer                       # a tool confirmed matches just now
        if confirmed_ever and self._items_all_known(answer, confirmed_ever):
            return answer                       # re-listing earlier confirmed names
        if failed:
            # Empty confirmations mean "nothing matched" OR "the lookup never
            # ran". Only the first justifies saying no specialist is nearby.
            return DIRECTORY_DOWN_PREFIX + answer
        if offered:
            return UNVERIFIED_PREFIX + answer   # only labelled fallbacks exist
        return WITHHELD

    @staticmethod
    def remind_emergency(answer: str, number: str) -> str:
        if not number:
            return answer
        if any(d in answer for d in _DIGITS.findall(number)):
            return answer                       # the model already stated it
        return (f"If the symptoms you described earlier are still going on, call "
                f"{number} now rather than travelling yourself: an ambulance comes to "
                f"you and the crew can start care on the way.\n\n" + answer)

    def review(self, answer: str, state: dict) -> str:
        """The guarded answer for this state (== answer when nothing changes)."""
        number = state.get("emergency_number") or ""
        if state.get("is_emergency"):
            # This turn's answer IS the emergency answer (call-for-help +
            # hospitals), so the specialist check doesn't apply — but the
            # number must be in it.
            return self.remind_emergency(answer, number)
        guarded = self.check_recommendations(
            answer,
            confirmed_this_turn=state.get("turn_confirmed") or set(),
            confirmed_ever=state.get("confirmed_providers") or set(),
            offered=state.get("offered_facilities") or set(),
            failed=state.get("tool_errors") or [])
        return self.remind_emergency(guarded, number)
