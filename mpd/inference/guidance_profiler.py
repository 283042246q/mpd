"""Low-overhead profiling and active-set statistics for MPD guidance."""

from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from pathlib import Path

import torch


class GuidanceProfiler:
    def __init__(self, enabled=False, record_active_statistics=False, sync_cuda=True):
        self.enabled = bool(enabled)
        self.record_active_statistics = bool(record_active_statistics)
        self.sync_cuda = bool(sync_cuda)
        self.records = []
        self._current = None

    def _sync(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def begin_call(self, **metadata):
        if not (self.enabled or self.record_active_statistics):
            return
        self._current = dict(metadata)

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            if self._current is not None:
                key = f"time_{name}_s"
                self._current[key] = self._current.get(key, 0.0) + time.perf_counter() - start

    def update(self, **statistics):
        if self._current is not None and (self.enabled or self.record_active_statistics):
            self._current.update(statistics)

    def end_call(self, warmup=False):
        if self._current is None:
            return
        if not warmup:
            self.records.append(self._current)
        self._current = None

    def snapshot(self):
        return [dict(record) for record in self.records]

    def clear(self):
        self.records.clear()
        self._current = None

    def write_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for record in self.records for key in record})
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
