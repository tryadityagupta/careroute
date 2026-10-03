"""
storage/seed_data.py — where CareRoute's static seed data comes from.

One code path, two behaviours, chosen by configuration (12-factor):
  * no BLOB_ACCOUNT_URL -> read ./data/*.json from disk (dev)
  * BLOB_ACCOUNT_URL    -> read the same files from Azure Blob (prod), with the
                           Container App's managed identity (no passwords);
                           locally DefaultAzureCredential uses `az login`.
"""

from __future__ import annotations

import json
from pathlib import Path


class SeedDataSource:
    def __init__(self, data_dir: Path, *, blob_account_url: str = "",
                 blob_container: str = "patient-data"):
        self.data_dir = Path(data_dir)
        self.blob_account_url = blob_account_url
        self.blob_container = blob_container

    def _load_local(self, filename: str):
        with open(self.data_dir / filename, encoding="utf-8") as f:
            return json.load(f)

    def _load_blob(self, filename: str):
        # Lazy import: local dev doesn't need the azure libraries at all.
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient
        client = BlobServiceClient(account_url=self.blob_account_url,
                                   credential=DefaultAzureCredential())
        data = client.get_container_client(self.blob_container).download_blob(filename).readall()
        return json.loads(data)

    def load(self, filename: str):
        if self.blob_account_url:
            print(f"[seed_data] loading {filename} from Blob ({self.blob_account_url})")
            return self._load_blob(filename)
        print(f"[seed_data] loading {filename} from local dir ({self.data_dir})")
        return self._load_local(filename)

    def patients(self) -> dict:
        return self.load("patients.json")

    def providers(self) -> list:
        return self.load("providers.json")
