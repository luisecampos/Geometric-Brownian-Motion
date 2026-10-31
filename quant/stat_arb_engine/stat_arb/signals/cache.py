"""
signals/cache.py
----------------
Disk cache and process-pool helpers for the two expensive steps on a wide
universe: the Engle-Granger tests in walk-forward selection (~3,800 pairs x
~20 windows) and the Kalman MLE fit (once per pair-period).

Cache entries are keyed by a hash of the exact input numbers (prices in the
window, pair, window dates), so a cache hit can only return what the
computation would have produced on the same inputs. Delete the cache
directory to start fresh.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def fingerprint(*parts) -> str:
    """Stable short hash of arrays / strings / numbers."""
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p, dtype=float).tobytes())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:20]


class DiskCache:
    """A dict persisted as one pickle file. `None` dir = in-memory only."""

    def __init__(self, cache_dir: str | None, name: str):
        self.path = os.path.join(cache_dir, f"{name}.pkl") if cache_dir else None
        self.data: dict = {}
        self.dirty = False
        if self.path and os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.data = pickle.load(f)

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        self.dirty = True

    def save(self):
        if self.path and self.dirty:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self.path)
            self.dirty = False


def parallel_map(fn, items: list, n_jobs: int = 1, chunksize: int = 1) -> list:
    """map(fn, items) in a process pool (order preserved); serial if n_jobs <= 1."""
    if n_jobs is None or n_jobs <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(fn, items, chunksize=chunksize))


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 1)
