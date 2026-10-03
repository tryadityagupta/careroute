"""
agent/state.py — the one object every graph node shares.

`Annotated[list, add_messages]` attaches a REDUCER to messages: a node's
return is appended (or replaces by id, or is removed via RemoveMessage). The
other keys have no reducer, so a node's return REPLACES the old value — which
is why the tools node merges the sets itself before returning them.

PER-TURN vs PER-CONVERSATION (reset by CareRouteAgent at the start of a turn):
    per turn          llm_calls, tool_errors, is_emergency, turn_confirmed
    per conversation  confirmed_providers, offered_facilities, emergency_number
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class CareRouteState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    # Names a SUCCESSFUL find_providers / find_pharmacies returned, at any
    # point in the conversation: re-mentioning them later is not a hallucination.
    confirmed_providers: set[str]
    # The subset confirmed in THIS turn. The guard needs both: a list in turn 3
    # is only trusted if this turn confirmed something, or if every item names
    # a provider confirmed earlier. (With only the conversation-wide set, one
    # confirmation in turn 1 switched the guard off for the rest of the chat.)
    turn_confirmed: set[str]
    offered_facilities: set[str]   # names from fallbacks / unverified results
    tool_errors: list[str]         # tools that FAILED (not: found nothing)
    llm_calls: int                 # model calls this turn (the step budget)
    is_emergency: bool             # get_emergency_help fired THIS turn
    emergency_number: str          # STICKY once set: never reset per turn


def fresh_turn() -> dict:
    """Keys reset at the start of every turn."""
    return {"llm_calls": 0, "tool_errors": [], "is_emergency": False,
            "turn_confirmed": set()}


def fresh_conversation() -> dict:
    """Keys initialised once, when a conversation starts."""
    return {**fresh_turn(), "confirmed_providers": set(),
            "offered_facilities": set(), "emergency_number": ""}
