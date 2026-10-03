"""
mock_llm/app.py — a stand-in for OpenAI's /v1/chat/completions. Free, offline.

Point the app at it and the WHOLE stack runs for real — LangGraph loop, tool
execution, PostGIS, OSRM, Nominatim, Redis, Postgres, answer guard — with only
the model faked:

    OPENAI_API_KEY=mock  OPENAI_BASE_URL=http://mock-llm:9100/v1
    uvicorn mock_llm.app:app --port 9100

Latency: uniform in [MOCK_LLM_LATENCY_MIN_S, MOCK_LLM_LATENCY_MAX_S] (default
0.8-2.0 s, roughly gpt-4o-mini with tools); MOCK_LLM_LATENCY_S pins one value.

Faults, switchable at RUNTIME so a load test can break the model mid-run:
    curl -XPOST localhost:9100/admin/fault -d '{"mode":"down"}'       # 503s
    curl -XPOST localhost:9100/admin/fault -d '{"mode":"ratelimit"}'  # 429s
    curl -XPOST localhost:9100/admin/fault -d '{"mode":"slow","slow_s":30}'
    curl -XPOST localhost:9100/admin/fault -d '{"mode":"flaky","error_rate":0.2}'
    curl -XPOST localhost:9100/admin/fault -d '{"mode":"ok"}'
Stats for the cost model:  GET /admin/stats   POST /admin/reset
Token counts are estimates (~4 chars/token): fine for a cost model, not billing.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from collections import Counter
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mock_llm.policy import policy_for


@dataclass
class MockConfig:
    mode: str = "agent"
    latency_min_s: float = 0.8
    latency_max_s: float = 2.0

    @classmethod
    def from_env(cls) -> "MockConfig":
        fixed = os.getenv("MOCK_LLM_LATENCY_S")
        return cls(mode=os.getenv("MOCK_LLM_MODE", "agent").strip().lower(),
                   latency_min_s=float(fixed or os.getenv("MOCK_LLM_LATENCY_MIN_S", "0.8")),
                   latency_max_s=float(fixed or os.getenv("MOCK_LLM_LATENCY_MAX_S", "2.0")))


class FaultInjector:
    MODES = ("ok", "down", "ratelimit", "slow", "flaky")

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.mode, self.error_rate, self.slow_s = "ok", 0.0, 30.0

    def update(self, body: dict) -> dict:
        self.mode = body.get("mode", self.mode)
        self.error_rate = float(body.get("error_rate", self.error_rate))
        self.slow_s = float(body.get("slow_s", self.slow_s))
        return self.as_dict()

    def as_dict(self) -> dict:
        return {"mode": self.mode, "error_rate": self.error_rate, "slow_s": self.slow_s}

    def error_response(self) -> JSONResponse | None:
        if self.mode == "down" or (self.mode == "flaky" and random.random() < self.error_rate):
            return JSONResponse(status_code=503, content={"error": {
                "message": "mock: model unavailable", "type": "server_error"}})
        if self.mode == "ratelimit":
            return JSONResponse(status_code=429, headers={"retry-after": "1"},
                                content={"error": {"message": "mock: rate limited",
                                                   "type": "rate_limit_error"}})
        return None


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def create_app(config: MockConfig | None = None) -> FastAPI:
    cfg = config or MockConfig.from_env()
    policy = policy_for(cfg.mode)
    faults = FaultInjector()
    stats: Counter = Counter()
    app = FastAPI(title="CareRoute mock LLM")
    app.state.config, app.state.faults, app.state.stats = cfg, faults, stats

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        msgs = body.get("messages", [])
        tool_names = {t.get("function", {}).get("name") for t in body.get("tools") or []}
        stats["requests"] += 1

        err = faults.error_response()
        if err is not None:
            stats["faults_503" if err.status_code == 503 else "faults_429"] += 1
            return err
        await asyncio.sleep(faults.slow_s if faults.mode == "slow"
                            else random.uniform(cfg.latency_min_s, cfg.latency_max_s))

        kind, out = policy.decide(msgs, tool_names)
        if kind == "tools":
            tool_calls = [{"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                           "function": {"name": n, "arguments": json.dumps(a)}}
                          for n, a in out]
            message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            finish, completion_text = "tool_calls", json.dumps(tool_calls)
            for n, _ in out:
                stats["tool:" + n] += 1
        else:
            message = {"role": "assistant", "content": out}
            finish, completion_text = "stop", out
            stats["answers"] += 1

        usage = {"prompt_tokens": _tokens(json.dumps(msgs) + json.dumps(body.get("tools") or [])),
                 "completion_tokens": _tokens(completion_text)}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        stats["prompt_tokens"] += usage["prompt_tokens"]
        stats["completion_tokens"] += usage["completion_tokens"]
        return {"id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion",
                "created": int(time.time()), "model": body.get("model", "mock"),
                "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                "usage": usage}

    @app.post("/admin/fault")
    async def set_fault(request: Request):
        body = await request.json()
        if body.get("mode") not in FaultInjector.MODES:
            return JSONResponse(status_code=400, content={"error": "bad mode"})
        return faults.update(body)

    @app.get("/admin/stats")
    async def get_stats():
        return {"mode": cfg.mode, "fault": faults.as_dict(),
                "latency_s": [cfg.latency_min_s, cfg.latency_max_s], **stats}

    @app.post("/admin/reset")
    async def reset():
        stats.clear()
        faults.reset()
        return {"ok": True}

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    return app


app = create_app()          # `uvicorn mock_llm.app:app` (reads MOCK_LLM_* env)
