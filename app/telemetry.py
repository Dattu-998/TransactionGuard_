"""In-process service metrics (reset on restart; enough for a single-instance demo)."""

from __future__ import annotations

import time
from collections import Counter, deque
from threading import Lock

import numpy as np


class Telemetry:
    def __init__(self, window: int = 1000):
        self.started = time.time()
        self._lock = Lock()
        self.requests = Counter()
        self.decisions = Counter()
        self.versions = Counter()
        self.errors = 0
        self.latencies: deque[float] = deque(maxlen=window)

    @property
    def uptime(self) -> float:
        return time.time() - self.started

    def record_request(self, path: str, status: int, elapsed_ms: float) -> None:
        with self._lock:
            self.requests[path] += 1
            if status >= 500:
                self.errors += 1
            if path.startswith("/api/predict"):
                self.latencies.append(elapsed_ms)

    def record_decisions(self, version: str, decisions) -> None:
        with self._lock:
            for decision in decisions:
                self.decisions[str(decision)] += 1
                self.versions[version] += 1

    def snapshot(self) -> dict:
        with self._lock:
            latencies = np.asarray(self.latencies) if self.latencies else np.zeros(1)
            scored = sum(self.decisions.values())
            return {
                "uptime_seconds": self.uptime,
                "requests_total": sum(self.requests.values()),
                "requests_by_path": dict(self.requests.most_common(12)),
                "errors_total": self.errors,
                "transactions_scored": scored,
                "decisions": {d: self.decisions.get(d, 0) for d in ("approve", "review", "block")},
                "by_version": dict(self.versions),
                "latency_ms": {
                    "p50": float(np.percentile(latencies, 50)),
                    "p95": float(np.percentile(latencies, 95)),
                    "p99": float(np.percentile(latencies, 99)),
                },
            }
