"""
agent/messages.py — small, pure helpers for reading messages and tool results.
"""

from __future__ import annotations

import json

from langchain_core.messages import ToolMessage

# Keys of a tool payload that hold facility lists. 'facilities' and
# 'general_alternatives' are unverified names an answer may mention, so the
# guard must know about them too.
FACILITY_LIST_KEYS = ("facilities", "general_alternatives", "nearest_hospitals", "pharmacies")


def payload_of(msg: ToolMessage):
    """ToolNode JSON-encodes dict/list results; decode them back. A tool that
    raised becomes ToolNode's 'Error: ...' string -> {'error': ...}."""
    if not isinstance(msg.content, str):
        return msg.content
    try:
        return json.loads(msg.content)
    except (ValueError, TypeError):
        return {"error": msg.content}


def text_of(message) -> str:
    """Message content as plain text (newer models may return content blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def names_in(result) -> set[str]:
    """Facility names present in a tool result, whatever its shape."""
    items = []
    if isinstance(result, list):
        items = result
    elif isinstance(result, dict):
        for key in FACILITY_LIST_KEYS:
            if isinstance(result.get(key), list):
                items += result[key]
    return {i["name"] for i in items if isinstance(i, dict) and i.get("name")}


def summarize(result) -> str:
    """One-line summary of a tool result, for step logs and traces."""
    if isinstance(result, list):
        def one(r):
            n = str(r.get("name", "?"))[:40]
            via = r.get("matched_via")
            return f"{n} [{via}]" if via else n
        names = ", ".join(one(r) for r in result[:5] if isinstance(r, dict))
        return f"OK list[{len(result)}]: {names}"
    if isinstance(result, dict):
        if "error" in result:
            return f"ERROR: {result['error']}"
        if result.get("match_found") is False:
            return f"MISS: {result.get('reason')}"
        if "facilities" in result:
            names = ", ".join(str(f.get("name", "?"))[:40] for f in result["facilities"][:5])
            return f"FALLBACK[{len(result['facilities'])}]: {names}"
    return f"OK: {str(result)[:120]}"
