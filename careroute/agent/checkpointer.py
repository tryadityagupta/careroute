"""
agent/checkpointer.py — where conversation state lives between turns.

    DATABASE_URL set  -> AsyncPostgresSaver: any replica can resume any thread
    not set           -> MemorySaver:        one process only (dev, tests)

Same graph either way; only persistence changes. Built lazily on first use so
importing the agent (as the offline tests do) never needs a database.

ASYNC: the saver sits on psycopg's AsyncConnectionPool. An async pool must be
opened ON the running event loop, so it is opened on the first turn (under an
asyncio.Lock, so concurrent first requests build one pool, not several) and
closed by Container.aclose() when the app shuts down. A turn holds a pooled
connection only while it loads or saves its checkpoint — never while it waits
on the LLM — so a pool of 10 serves hundreds of concurrent conversations.
"""

from __future__ import annotations

import asyncio

from careroute.runtime import ensure_psycopg_compatible_loop


class CheckpointerFactory:
    def __init__(self, database_url: str = "", *, pool_max: int = 10,
                 auto_migrate: bool = True):
        self.database_url = database_url
        self.pool_max = pool_max
        # Production: run `python -m careroute.storage.checkpoints migrate`
        # once per deploy and set CAREROUTE_PG_AUTO_MIGRATE=0, so N replicas
        # booting together don't race to migrate.
        self.auto_migrate = auto_migrate
        self._saver = None
        self._pool = None
        self._lock: asyncio.Lock | None = None

    @property
    def is_postgres(self) -> bool:
        return bool(self.database_url)

    async def get(self):
        if self._saver is None:
            if self._lock is None:
                self._lock = asyncio.Lock()
            async with self._lock:
                if self._saver is None:
                    self._saver = await self._build()
        return self._saver

    async def _build(self):
        if not self.is_postgres:
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()
        ensure_psycopg_compatible_loop()     # clear error on Windows
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
        pool = AsyncConnectionPool(
            conninfo=self.database_url, min_size=1, max_size=self.pool_max,
            # Required by the saver; prepare_threshold=0 also keeps it working
            # behind PgBouncer in transaction mode.
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=False)
        try:
            await pool.open(wait=True, timeout=5)
            saver = AsyncPostgresSaver(pool)
            if self.auto_migrate:
                await saver.setup()
        except Exception:
            await pool.close()
            raise
        self._pool = pool
        return saver

    async def ping(self) -> None:
        """Raise unless Postgres answers (for /healthz)."""
        await self.get()
        if self._pool is not None:
            async with self._pool.connection(timeout=2) as conn:
                await conn.execute("SELECT 1")

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._saver = None
