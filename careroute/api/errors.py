"""
api/errors.py — dependency outages become "try again shortly" (503), not 500.

A 500 says "our code is broken"; these are "a dependency is briefly down", and
every one of them tells the user to call 112 if it's an emergency, because the
person reading the error may be in one.

  Redis down          sessions / patient records unavailable
  Postgres down       conversation history unavailable
  LLM down / slow     the model timed out, refused, or rate-limited us

The LLM handler is new: before it, a model outage was an unhandled 500 with
no emergency number. (Step 5's deterministic pre-pass will go further and
answer emergencies without the model at all.)
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

STORE_DOWN = {"detail": "CareRoute is briefly unavailable. Please try again "
                        "in a moment, and if this is an emergency call 112 now."}
LLM_DOWN = {"detail": "CareRoute's assistant is not responding right now. Please "
                      "try again in a moment, and if this is an emergency call "
                      "112 (ambulance: 108) now."}


def register_error_handlers(app: FastAPI, *, use_redis: bool, use_postgres: bool) -> None:
    if use_redis:
        import redis

        @app.exception_handler(redis.exceptions.RedisError)
        def _redis_down(request, exc):
            print(f"[api] redis error: {exc}")
            return JSONResponse(status_code=503, content=STORE_DOWN)

    if use_postgres:
        import psycopg
        import psycopg_pool

        @app.exception_handler(psycopg.OperationalError)
        @app.exception_handler(psycopg_pool.PoolTimeout)
        def _postgres_down(request, exc):
            print(f"[api] postgres error: {exc}")
            return JSONResponse(status_code=503, content=STORE_DOWN)

    import openai

    # APIError covers connection errors, timeouts, 429s and 5xx from the model.
    @app.exception_handler(openai.APIError)
    def _llm_down(request, exc):
        print(f"[api] LLM error: {type(exc).__name__}: {str(exc)[:200]}")
        return JSONResponse(status_code=503, content=LLM_DOWN, headers={"Retry-After": "5"})
