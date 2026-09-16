"""
data_source.py — where CareRoute's seed data comes from.

ONE code path, TWO behaviours, chosen by environment variables:

  * Local dev  (no BLOB_ACCOUNT_URL set)  -> read ./data/*.json from disk
  * Production (BLOB_ACCOUNT_URL is set)   -> read the same files from Azure Blob

This is the "12-factor" idea: the code never changes between your laptop and
prod; only configuration does. Nothing here holds a password — in Azure the
Container App's managed identity authenticates automatically, and on your
laptop DefaultAzureCredential falls back to your `az login` session.

Env vars (set the first two only in Azure):
  BLOB_ACCOUNT_URL   e.g. https://careroutedata.blob.core.windows.net
  BLOB_CONTAINER     default "patient-data"
  DATA_DIR           local folder for dev, default "./data"
"""

import json
import os

_ACCOUNT_URL = os.environ.get("BLOB_ACCOUNT_URL")
_CONTAINER = os.environ.get("BLOB_CONTAINER", "patient-data")
_DATA_DIR = os.environ.get("DATA_DIR", os.path.join(
    os.path.dirname(__file__), "data"))


def _load_local(filename):
    with open(os.path.join(_DATA_DIR, filename), encoding="utf-8") as f:
        return json.load(f)


def _load_blob(filename):
    # Imported lazily so local dev doesn't even need the azure libraries installed.
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    client = BlobServiceClient(
        account_url=_ACCOUNT_URL, credential=DefaultAzureCredential())
    container = client.get_container_client(_CONTAINER)
    data = container.download_blob(filename).readall()
    return json.loads(data)


def _load(filename):
    if _ACCOUNT_URL:
        print(f"[data_source] loading {filename} from Blob ({_ACCOUNT_URL})")
        return _load_blob(filename)
    print(f"[data_source] loading {filename} from local dir ({_DATA_DIR})")
    return _load_local(filename)


def load_patients():
    return _load("patients.json")


def load_providers():
    return _load("providers.json")
