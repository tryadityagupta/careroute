"""
agent/tracing.py — what the agent DID on the latest turn, safe to log.

Every tool it called, with privacy-trimmed arguments and a one-line result.
Only messages after the last HumanMessage count, so in a conversation this
describes THIS turn. tools == [] means the model answered from memory.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from careroute.agent.messages import payload_of, summarize


class TurnTracer:
    DROP_KEYS = frozenset({"name"})          # personal identifiers never logged
    COARSE_SUFFIXES = ("lat", "lng")         # rounded like the interaction log

    def __init__(self, coord_dp: int = 2):
        self.coord_dp = coord_dp

    def trim_args(self, args: dict | None) -> dict:
        out = {}
        for k, v in (args or {}).items():
            if k in self.DROP_KEYS:
                continue
            if k.endswith(self.COARSE_SUFFIXES) and isinstance(v, (int, float)):
                v = round(v, self.coord_dp)
            out[k] = v
        return out

    def trace(self, messages: list) -> dict:
        humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        turn = messages[(humans[-1] + 1) if humans else 0:]
        results = {m.tool_call_id: summarize(payload_of(m))
                   for m in turn if isinstance(m, ToolMessage)}
        calls = [{"tool": c["name"], "args": self.trim_args(c.get("args")),
                  "result": results.get(c.get("id"), "(no result)")[:300]}
                 for m in turn for c in (getattr(m, "tool_calls", None) or [])]
        return {"tools": calls, "llm_calls": sum(isinstance(m, AIMessage) for m in turn)}
