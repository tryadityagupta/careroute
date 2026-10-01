"""
checkpoints.py — operate the Postgres conversation store.

    python checkpoints.py migrate                 # create/upgrade tables (once per deploy)
    python checkpoints.py prune [--idle-hours 2]  # delete idle conversations
    python checkpoints.py stats                   # how big is it?

WHY PRUNE: Redis forgets an idle session by itself (TTL), but Postgres keeps
every checkpoint forever. Once a session has expired, nobody can ever resume
its thread, so its checkpoints are dead weight — and they contain symptoms and
medications, so keeping them is also a privacy liability (data minimisation).
Run `prune` on a schedule, e.g. an Azure Container Apps Job every 15 minutes,
with --idle-hours comfortably above CAREROUTE_SESSION_TTL.

Deletes go in batches so a big backlog never becomes one giant transaction
that locks the tables live traffic is writing to.
"""

import argparse
import sys

import psycopg

import shared_state

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


def migrate() -> None:
    from langgraph.checkpoint.postgres import PostgresSaver
    with PostgresSaver.from_conn_string(shared_state.DATABASE_URL) as saver:
        saver.setup()
    print("[checkpoints] schema up to date")


def prune(idle_hours: float, batch: int = 500) -> int:
    threads = 0
    with psycopg.connect(shared_state.DATABASE_URL, autocommit=True) as conn:
        while True:
            rows = conn.execute(_PRUNE_SQL, {"idle_s": idle_hours * 3600,
                                             "batch": batch}).fetchall()
            n = len({r[0] for r in rows})
            threads += n
            if n == 0:
                break
    print(f"[checkpoints] pruned {threads} idle conversation(s) "
          f"(idle > {idle_hours} h)")
    return threads


def stats() -> None:
    with psycopg.connect(shared_state.DATABASE_URL) as conn:
        threads, cps = conn.execute(
            "SELECT count(DISTINCT thread_id), count(*) FROM checkpoints"
        ).fetchone()
        size = conn.execute(
            "SELECT pg_size_pretty(sum(pg_total_relation_size(c::regclass))) "
            "FROM unnest(ARRAY['checkpoints','checkpoint_blobs',"
            "'checkpoint_writes']) c").fetchone()[0]
    print(f"[checkpoints] {threads} conversations, {cps} checkpoints, {size}")


if __name__ == "__main__":
    if not shared_state.USE_POSTGRES:
        sys.exit("DATABASE_URL is not set — nothing to operate on.")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate")
    sub.add_parser("stats")
    p = sub.add_parser("prune")
    p.add_argument("--idle-hours", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=500)
    a = ap.parse_args()
    if a.cmd == "migrate":
        migrate()
    elif a.cmd == "stats":
        stats()
    else:
        prune(a.idle_hours, a.batch)
