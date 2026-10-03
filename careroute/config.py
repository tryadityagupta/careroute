"""
careroute/config.py — every setting CareRoute reads, in ONE place.

Before this file, a dozen modules each called load_dotenv() and read their own
env vars at import time. That made import order matter (USE_REAL_PROVIDERS
once silently didn't work because it was read before .env was loaded) and
forced tests to set env vars BEFORE importing anything.

Now the environment is read exactly once, into an immutable Settings object,
and that object is handed to whoever needs it (see container.py). Tests build
their own Settings with dataclasses.replace() — no env juggling, no reloads.

If you add a setting: add the field here, read it in from_env(), document it
in .env.example. Nothing else should call os.getenv().
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PUBLIC_NOMINATIM = "https://nominatim.openstreetmap.org"
PUBLIC_OSRM = "https://router.project-osrm.org"


def _flag(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(v.strip() for v in (value or "").split(",") if v.strip())


@dataclass(frozen=True)
class Settings:
    # --- LLM -----------------------------------------------------------------
    openai_model: str = "gpt-4o-mini"
    # Seconds before ONE model call is abandoned. The OpenAI SDK default is
    # 600 s, which let a hung model hold a request slot for ten minutes.
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    max_llm_calls: int = 8                 # per turn (the old max_steps)

    # --- Provider directory --------------------------------------------------
    provider_backend: str = ""             # "" (dummy JSON) | "osm" | "google"
    osm_source: str = "overpass"           # "overpass" | "postgis"
    google_maps_api_key: str = ""

    # --- Self-hostable map services -----------------------------------------
    osrm_base_url: str = PUBLIC_OSRM
    osrm_timeout_s: float = 3.0
    nominatim_url: str = PUBLIC_NOMINATIM
    geocode_countries: str = "in"

    # --- Shared state (stateless replicas) ----------------------------------
    redis_url: str = ""
    database_url: str = ""
    geo_database_url: str = ""             # defaults to database_url
    require_shared_state: bool = False
    key_prefix: str = "careroute"
    redis_timeout_s: float = 1.0
    pg_pool_max: int = 10
    pg_auto_migrate: bool = True
    geo_pool_max: int = 10
    geo_statement_timeout_ms: int = 2000
    session_ttl_s: int = 1800
    max_sessions: int = 5000               # memory backend only
    turn_lock_s: int = 180

    # --- Abuse protection ----------------------------------------------------
    rate_per_min: int = 20
    burst: int = 5
    daily_call_cap: int = 0
    api_keys: tuple[str, ...] = ()
    trust_xff: bool = True
    allowed_origins: tuple[str, ...] = ("*",)

    # --- Logging -------------------------------------------------------------
    log_pii: bool = False
    log_coord_dp: int = 2
    log_file: str = ""

    # --- Seed data -----------------------------------------------------------
    blob_account_url: str = ""
    blob_container: str = "patient-data"
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    web_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "web")

    git_sha: str = "local-dev"

    # --- Derived -------------------------------------------------------------
    @property
    def use_redis(self) -> bool:
        return bool(self.redis_url)

    @property
    def use_postgres(self) -> bool:
        return bool(self.database_url)

    @property
    def geo_dsn(self) -> str:
        return self.geo_database_url or self.database_url

    @property
    def nominatim_is_public(self) -> bool:
        return self.nominatim_url.rstrip("/") == PUBLIC_NOMINATIM

    def replace(self, **changes) -> "Settings":
        """Copy with some fields changed (tests use this)."""
        return replace(self, **changes)

    def validate(self) -> None:
        """Refuse to start in a configuration that would misbehave silently."""
        if self.require_shared_state and not (self.use_redis and self.use_postgres):
            missing = [n for n, ok in (("REDIS_URL", self.use_redis),
                                       ("DATABASE_URL", self.use_postgres)) if not ok]
            raise RuntimeError(
                "CAREROUTE_REQUIRE_SHARED_STATE=1 but " + " and ".join(missing)
                + " not set. Refusing to start in single-replica memory mode.")
        if self.osm_source not in ("overpass", "postgis"):
            raise ValueError(f"OSM_SOURCE must be 'overpass' or 'postgis', "
                             f"got {self.osm_source!r}")

    # --- Construction --------------------------------------------------------
    @classmethod
    def from_env(cls, *, load_env_file: bool = True) -> "Settings":
        """Read the process environment (and .env, unless told not to).

        Keep .env comments on their own lines: python-dotenv reads
        `KEY=   # note` as the literal value '# note'.
        """
        if load_env_file:
            load_dotenv()
        env = os.environ.get
        backend = env("USE_REAL_PROVIDERS", "").strip().lower()
        if backend == "1":                 # historical alias for google
            backend = "google"
        return cls(
            openai_model=env("CAREROUTE_OPENAI_MODEL", "gpt-4o-mini"),
            llm_timeout_s=float(env("CAREROUTE_LLM_TIMEOUT_S", "30")),
            llm_max_retries=int(env("CAREROUTE_LLM_MAX_RETRIES", "2")),
            max_llm_calls=int(env("CAREROUTE_MAX_LLM_CALLS", "8")),
            provider_backend=backend,
            osm_source=env("OSM_SOURCE", "overpass").strip().lower(),
            google_maps_api_key=env("GOOGLE_MAPS_API_KEY", "") or "",
            osrm_base_url=(env("OSRM_BASE_URL") or PUBLIC_OSRM).rstrip("/"),
            osrm_timeout_s=float(env("OSRM_TIMEOUT_S", "3")),
            nominatim_url=(env("NOMINATIM_URL") or PUBLIC_NOMINATIM).rstrip("/"),
            geocode_countries=env("CAREROUTE_GEOCODE_COUNTRIES", "in").strip(),
            redis_url=env("REDIS_URL", "").strip(),
            database_url=env("DATABASE_URL", "").strip(),
            geo_database_url=env("GEO_DATABASE_URL", "").strip(),
            require_shared_state=_flag(env("CAREROUTE_REQUIRE_SHARED_STATE"), False),
            key_prefix=env("CAREROUTE_KEY_PREFIX", "careroute"),
            redis_timeout_s=float(env("CAREROUTE_REDIS_TIMEOUT_S", "1.0")),
            pg_pool_max=int(env("CAREROUTE_PG_POOL_MAX", "10")),
            pg_auto_migrate=env("CAREROUTE_PG_AUTO_MIGRATE", "1") != "0",
            geo_pool_max=int(env("GEO_POOL_MAX", "10")),
            geo_statement_timeout_ms=int(env("GEO_STATEMENT_TIMEOUT_MS", "2000")),
            session_ttl_s=int(env("CAREROUTE_SESSION_TTL", "1800")),
            max_sessions=int(env("CAREROUTE_MAX_SESSIONS", "5000")),
            turn_lock_s=int(env("CAREROUTE_TURN_LOCK_S", "180")),
            rate_per_min=int(env("CAREROUTE_RATE_PER_MIN", "20")),
            burst=int(env("CAREROUTE_BURST", "5")),
            daily_call_cap=int(env("CAREROUTE_DAILY_CALL_CAP", "0")),
            api_keys=_csv(env("CAREROUTE_API_KEYS")),
            trust_xff=_flag(env("CAREROUTE_TRUST_XFF"), True),
            allowed_origins=_csv(env("CAREROUTE_ALLOWED_ORIGINS", "*")) or ("*",),
            log_pii=_flag(env("CAREROUTE_LOG_PII"), False),
            log_coord_dp=int(env("CAREROUTE_LOG_COORD_DP", "2")),
            log_file=env("CAREROUTE_LOG_FILE", "").strip(),
            blob_account_url=env("BLOB_ACCOUNT_URL", "") or "",
            blob_container=env("BLOB_CONTAINER", "patient-data"),
            data_dir=Path(env("DATA_DIR") or PROJECT_ROOT / "data"),
            git_sha=env("GIT_SHA", "local-dev"),
        )
