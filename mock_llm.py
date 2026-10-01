"""
mock_llm.py — a stand-in for OpenAI's /v1/chat/completions. Free, offline.

Used by test_stateless.py (and later the load tests): point the app at it with
    OPENAI_BASE_URL=http://127.0.0.1:9100/v1

Its reply reports what the model was SHOWN, which is exactly what a
statelessness test needs to check:
    "turns_seen=3 | meds=ibuprofen, cetirizine"
turns_seen counts the user messages in the transcript it received — if a turn
lands on a different replica and the conversation history did not travel with
it, this number comes back as 1 instead of 3.

MOCK_LLM_LATENCY_S simulates a real model's latency (default 0.3 s), so
concurrency behaviour (turn locks, threadpool limits) is realistic.

Run standalone:  uvicorn mock_llm:app --port 9100
"""

import asyncio
import os
import re
import time
import uuid

from fastapi import FastAPI, Request

app = FastAPI()
LATENCY_S = float(os.getenv("MOCK_LLM_LATENCY_S", "0.3"))
_MEDS = re.compile(r"current_medications=([^\]]*)\]")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    msgs = body.get("messages", [])
    users = [m for m in msgs if m.get("role") == "user"]
    last = users[-1]["content"] if users else ""
    if isinstance(last, list):  # content-parts format
        last = " ".join(p.get("text", "") for p in last if isinstance(p, dict))
    meds = _MEDS.search(last)
    reply = (f"turns_seen={len(users)} | "
             f"meds={meds.group(1).strip() if meds else '?'}")
    await asyncio.sleep(LATENCY_S)   # async: the mock itself never bottlenecks
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock"),
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": reply}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                  "total_tokens": 0},
    }
