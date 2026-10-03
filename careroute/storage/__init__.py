"""Shared state and persistence.

    redis_client.py   one Redis connection per process + key naming
    sessions.py       SessionStore: memory | redis (+ the per-conversation turn lock)
    patient_store.py  PatientStore: live patient records, memory | redis
    checkpoints.py    operate the Postgres conversation store (migrate/prune/stats)
    seed_data.py      static demo data from ./data or Azure Blob
"""
