"""
agent/checkpointer.py — where conversation state lives between turns.

    DATABASE_URL set  -> PostgresSaver: any replica can resume any thread
    not set           -> MemorySaver:   one process only (dev, tests)

Same graph either way; only persistence changes. Built lazily on first use so
importing the agent (as the offline tests do) never needs a database.
"""

from __future__ import annotations

import atexit
import threading


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
        self._lock = threading.Lock()

    @property
    def is_postgres(self) -> bool:
        return bool(self.database_url)

    def get(self):
        if self._saver is None:
            with self._lock:
                if self._saver is None:
                    self._saver = self._build()
        return self._saver

    def _build(self):
        if not self.is_postgres:
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
        # One pool per replica. A turn holds a connection only while it loads
        # or saves a checkpoint, never while it waits on the LLM.
        self._pool = ConnectionPool(
            conninfo=self.database_url, min_size=1, max_size=self.pool_max,
            # Required by PostgresSaver; prepare_threshold=0 also keeps it
            # working behind PgBouncer in transaction mode.
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=True)
        # Close explicitly at exit, or interpreter shutdown waits ~5 s per
        # pool thread and Ctrl+C hangs.
        atexit.register(self._pool.close)
        saver = PostgresSaver(self._pool)
        if self.auto_migrate:
            saver.setup()
        return saver

    def ping(self) -> None:
        """Raise unless Postgres answers (for /healthz)."""
        self.get()
        if self._pool is not None:
            with self._pool.connection(timeout=2) as conn:
                conn.execute("SELECT 1")

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
