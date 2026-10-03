"""
maps/cache.py — a small, thread-safe, BOUNDED in-process cache.

The geocode cache used to be a plain dict that grew forever (every distinct
string a user typed stayed in memory), and the route cache was cleared
wholesale when full. Under a load test both are problems: one leaks, the
other drops its hit rate to zero every 20k entries. An LRU evicts only the
least recently used entry.

Per-replica on purpose: these hold re-computable data, so a cold cache is
slower, never wrong. Shared state (sessions, records) lives in storage/.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Hashable

_MISSING = object()


class LRUCache:
    def __init__(self, max_items: int):
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self._max = max_items
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            value = self._data.get(key, _MISSING)
            if value is _MISSING:
                return default
            self._data.move_to_end(key)
            return value

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
