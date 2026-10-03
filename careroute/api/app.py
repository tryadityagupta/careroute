"""
api/app.py — the FastAPI application.

Run it:
    uvicorn --factory careroute.api.app:create_app --reload
Then open http://localhost:8000

create_app() is a FACTORY: nothing is built or connected at import time.
Tests call create_app(container) with a Container whose parts they replaced.

Handlers are still sync `def` (FastAPI runs them in its threadpool, ~40
concurrent requests per process). Converting them to async is the next
change; with I/O behind the classes in maps/, storage/ and agent/, that is a
change to those classes, not to this file's structure.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from careroute.api.errors import register_error_handlers
from careroute.api.schemas import CareRequest, ChatRequest
from careroute.container import Container
from careroute.maps.postgis import GeoUnavailable

CODE_VERSION = "2026-10-oop-package"


def create_app(container: Container | None = None) -> FastAPI:
    c = container or Container()
    s = c.settings
    print(c.describe())

    app = FastAPI(title="CareRoute")
    app.state.container = c
    # Lock to your real origin(s) in production via CAREROUTE_ALLOWED_ORIGINS.
    app.add_middleware(CORSMiddleware, allow_origins=list(s.allowed_origins),
                       allow_methods=["*"], allow_headers=["*"])
    register_error_handlers(app, use_redis=s.use_redis, use_postgres=s.use_postgres)

    # Rate limit FIRST (throttles even unauthenticated floods), then API key.
    guards = [Depends(c.gate.rate_limit), Depends(c.gate.require_api_key)]

    @app.get("/")
    def home():
        return FileResponse(s.web_dir / "index.html")

    @app.post("/care", dependencies=guards)
    def care(req: CareRequest):
        return c.care_service.handle(req)

    @app.post("/chat", dependencies=guards)
    def chat(req: ChatRequest):
        return c.chat_service.handle(req)

    @app.get("/healthz")
    def healthz():
        """Readiness: can THIS replica reach its shared stores?

        Point the readiness probe here and keep liveness on /version, so a
        store outage takes replicas out of rotation instead of restarting them
        all at once. The provider directory is REPORTED but doesn't fail
        readiness: it's shared by every replica, so failing on it would pull
        them all — including the emergency path, which works without it.
        """
        checks, ok = {}, True
        if c.redis:
            try:
                c.redis.ping()
                checks["redis"] = "ok"
            except Exception as e:
                checks["redis"], ok = f"error: {type(e).__name__}", False
        if s.use_postgres:
            try:
                c.checkpointers.ping()
                checks["postgres"] = "ok"
            except Exception as e:
                checks["postgres"], ok = f"error: {type(e).__name__}", False
        info = {}
        if s.osm_source == "postgis":
            try:
                info["provider_directory"] = c.geo_db.dataset_info()
            except GeoUnavailable as e:
                info["provider_directory"] = f"unavailable: {e}"
        body = {"status": "ok" if ok else "degraded",
                "mode": "shared" if s.use_redis else "memory",
                "checks": checks, "info": info}
        return JSONResponse(status_code=200 if ok else 503, content=body)

    @app.get("/version")
    def version():
        return {"version": CODE_VERSION, "commit": s.git_sha,
                "backend": c.directory.backend_name}

    return app
