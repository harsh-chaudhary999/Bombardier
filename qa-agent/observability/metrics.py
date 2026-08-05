"""
Lightweight observability metrics for the QA Intelligence Engine.

Provides counters, histograms, and gauges without external dependencies.
Exposes data via /metrics (JSON) and /metrics/prometheus (text exposition).

Usage:
    from observability.metrics import metrics

    metrics.inc("sync_tests_total")
    metrics.observe("search_latency_seconds", 0.123)
    with metrics.timer("analysis_duration_seconds"):
        ...  # timed block
"""
import time
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Any


class Metrics:
    """Thread-safe in-process metrics collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}

    # ─── Counters ─────────────────────────────────────────────────────────────

    def inc(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment a counter."""
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] += value

    # ─── Gauges ───────────────────────────────────────────────────────────────

    def set(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge to a specific value."""
        key = self._key(name, labels)
        with self._lock:
            self._gauges[key] = value

    # ─── Histograms ───────────────────────────────────────────────────────────

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record an observation in a histogram."""
        key = self._key(name, labels)
        with self._lock:
            self._histograms[key].append(value)
            # Keep last 10000 observations to bound memory
            if len(self._histograms[key]) > 10000:
                self._histograms[key] = self._histograms[key][-5000:]

    @contextmanager
    def timer(self, name: str, labels: dict[str, str] | None = None):
        """Context manager that records elapsed time as a histogram observation."""
        start = time.monotonic()
        try:
            yield
        finally:
            self.observe(name, time.monotonic() - start, labels)

    # ─── Export ────────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of all metrics."""
        with self._lock:
            result: dict[str, Any] = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
            }
            for key, values in self._histograms.items():
                if values:
                    sorted_v = sorted(values)
                    n = len(sorted_v)
                    p95_idx = min(int(n * 0.95), n - 1)
                    p99_idx = min(int(n * 0.99), n - 1)
                    result["histograms"][key] = {
                        "count": n,
                        "sum": round(sum(sorted_v), 6),
                        "min": round(sorted_v[0], 6),
                        "max": round(sorted_v[-1], 6),
                        "avg": round(sum(sorted_v) / n, 6),
                        "p50": round(sorted_v[n // 2], 6),
                        "p95": round(sorted_v[p95_idx], 6) if n >= 20 else round(sorted_v[-1], 6),
                        "p99": round(sorted_v[p99_idx], 6) if n >= 100 else round(sorted_v[-1], 6),
                    }
            return result

    def prometheus_text(self) -> str:
        """Export metrics in Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            emitted_counter_types: set[str] = set()
            for key, value in sorted(self._counters.items()):
                name, labels_str = self._parse_key(key)
                if name not in emitted_counter_types:
                    lines.append(f"# TYPE {name} counter")
                    emitted_counter_types.add(name)
                lines.append(f"{name}{labels_str} {value}")

            emitted_gauge_types: set[str] = set()
            for key, value in sorted(self._gauges.items()):
                name, labels_str = self._parse_key(key)
                if name not in emitted_gauge_types:
                    lines.append(f"# TYPE {name} gauge")
                    emitted_gauge_types.add(name)
                lines.append(f"{name}{labels_str} {value}")

            emitted_hist_types: set[str] = set()
            for key, values in sorted(self._histograms.items()):
                if not values:
                    continue
                name, labels_str = self._parse_key(key)
                sorted_v = sorted(values)
                n = len(sorted_v)
                if name not in emitted_hist_types:
                    lines.append(f"# TYPE {name} summary")
                    emitted_hist_types.add(name)
                lines.append(f"{name}_count{labels_str} {n}")
                lines.append(f"{name}_sum{labels_str} {sum(sorted_v):.6f}")
                if n >= 2:
                    p50 = sorted_v[n // 2]
                    p95 = sorted_v[min(int(n * 0.95), n - 1)]
                    p99 = sorted_v[min(int(n * 0.99), n - 1)]
                    lines.append(f'{name}{{quantile="0.5"}} {p50:.6f}')
                    lines.append(f'{name}{{quantile="0.95"}} {p95:.6f}')
                    lines.append(f'{name}{{quantile="0.99"}} {p99:.6f}')

        return "\n".join(lines) + "\n"

    # ─── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    @staticmethod
    def _parse_key(key: str) -> tuple[str, str]:
        if "{" in key:
            name, rest = key.split("{", 1)
            return name, "{" + rest
        return key, ""


# Singleton instance — import and use from anywhere
metrics = Metrics()
