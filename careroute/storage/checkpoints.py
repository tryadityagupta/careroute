"""
storage/checkpoints.py — operate the Postgres conversation store.

    python -m careroute.storage.checkpoints migrate                 # once per deploy
    python -m careroute.storage.checkpoints prune [--idle-hours 2]  # on a schedule
    python -m careroute.storage.checkpoints stats

WHY PRUNE: Redis forgets an idle session by TTL, but Postgres keeps every
checkpoint forever. Once a session has expired nobody can resume its thread,
so its checkpoints are dead weight — and they hold symptoms and medications,
so keeping them is a privacy liability (data minimisation). Run prune every
15 minutes with --idle-hours comfortably above CAREROUTE_SESSION_TTL.

Deletes go in batches so a backlog never becomes one giant transaction that
locks the tables live traffic writes to.
"""

from __future__ import annotations

import argparse
import sys

import psycopg

from careroute.config import Settings


class CheckpointAdmin:
    _PRUNE_SQL = """
WITH stale AS (
    SELECT thread_id
    FROM checkpoints
    GROUP BY thread_id
    HAVING max((checkpoint->>'ts')::timestamptz)
           < now() - make_interval(secs => %(idle_s)s)
    LIMIT %(batch)s
),
w AS (DELETE FROM checkpoint_writes x USING stale s WHERE x.thread_id = s.thread_id),
b AS (DELETE FROM checkpoint_blobs  x USING stale s WHERE x.thread_id = s.thread_id)
DELETE FROM checkpoints x USING stale s
WHERE x.thread_id = s.thread_id
RETURNING x.thread_id
"""

    def __init__(self, database_url: str):
        if not database_url:
            raise ValueError("DATABASE_URL is not set — nothing to operate on.")
        self.database_url = database_url

    def migrate(self) -> None:
        from langgraph.checkpoint.postgres import PostgresSaver
        with PostgresSaver.from_conn_string(self.database_url) as saver:
            saver.setup()
        print("[checkpoints] schema up to date")

    def prune(self, idle_hours: float, batch: int = 500) -> int:
        threads = 0
        with psycopg.connect(self.database_url, autocommit=True) as conn:
            while True:
                rows = conn.execute(self._PRUNE_SQL, {"idle_s": idle_hours * 3600,
                                                      "batch": batch}).fetchall()
                n = len({r[0] for r in rows})
                threads += n
                if n == 0:
                    break
        print(f"[checkpoints] pruned {threads} idle conversation(s) (idle > {idle_hours} h)")
        return threads

    def stats(self) -> dict:
        with psycopg.connect(self.database_url) as conn:
            threads, cps = conn.execute(
                "SELECT count(DISTINCT thread_id), count(*) FROM checkpoints").fetchone()
            size = conn.execute(
                "SELECT pg_size_pretty(sum(pg_total_relation_size(c::regclass))) "
                "FROM unnest(ARRAY['checkpoints','checkpoint_blobs',"
                "'checkpoint_writes']) c").fetchone()[0]
        print(f"[checkpoints] {threads} conversations, {cps} checkpoints, {size}")
        return {"threads": threads, "checkpoints": cps, "size": size}


def main(argv: list[str] | None = None) -> None:
    settings = Settings.from_env()
    if not settings.use_postgres:
        sys.exit("DATABASE_URL is not set — nothing to operate on.")
    ap = argparse.ArgumentParser(prog="python -m careroute.storage.checkpoints")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate")
    sub.add_parser("stats")
    p = sub.add_parser("prune")
    p.add_argument("--idle-hours", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=500)
    a = ap.parse_args(argv)
    admin = CheckpointAdmin(settings.database_url)
    if a.cmd == "migrate":
        admin.migrate()
    elif a.cmd == "stats":
        admin.stats()
    else:
        admin.prune(a.idle_hours, a.batch)


if __name__ == "__main__":
    main()
