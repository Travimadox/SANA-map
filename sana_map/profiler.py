# ═══════════════════════════════════════════════════════════════════════
# PROJECT: SANA-map
# FILE: profiler.py
# DESCRIPTION: Lightweight per-stage pipeline profiler for the
#              Semantic SLAM system.  Measures wall-clock latency for
#              every processing stage and reports FPS / % of frame time.
# AUTHOR: Travimadox Webb
# ═══════════════════════════════════════════════════════════════════════

import time
import csv
import math
from pathlib import Path
from contextlib import contextmanager

import torch


class PipelineProfiler:
    """Accumulates per-stage timing statistics and prints / exports them.

    Usage
    -----
    >>> prof = PipelineProfiler(cuda_sync=True, print_interval=100)
    >>> with prof.stage("detection"):
    ...     run_detection()
    >>> prof.summary()          # pretty-printed table
    >>> prof.to_csv("out.csv")  # raw data export
    """

    def __init__(self, cuda_sync: bool = True, print_interval: int = 100,
                 enabled: bool = True):
        self.cuda_sync = cuda_sync and torch.cuda.is_available()
        self.print_interval = print_interval
        self.enabled = enabled

        # {stage_name: {"count": int, "total": float,
        #               "min": float, "max": float, "history": [float]}}
        self._stats: dict[str, dict] = {}
        self._stage_order: list[str] = []   # preserve insertion order
        self._active: dict[str, float] = {}
        self._frame_count = 0
        self._run_start: float | None = None

    # ── public API ───────────────────────────────────────────────────

    def tick_frame(self):
        """Call once per frame to trigger periodic live reports."""
        if not self.enabled:
            return
        if self._run_start is None:
            self._run_start = time.perf_counter()
        self._frame_count += 1
        if self._frame_count % self.print_interval == 0:
            self._print_live()

    @contextmanager
    def stage(self, name: str):
        """Context manager that times a named pipeline stage."""
        if not self.enabled:
            yield
            return
        if self.cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if self.cuda_sync:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        self._record(name, elapsed)

    def start(self, name: str):
        """Manual start for a named stage (pair with stop)."""
        if not self.enabled:
            return
        if self.cuda_sync:
            torch.cuda.synchronize()
        self._active[name] = time.perf_counter()

    def stop(self, name: str):
        """Manual stop for a named stage (pair with start)."""
        if not self.enabled:
            return
        if self.cuda_sync:
            torch.cuda.synchronize()
        t0 = self._active.pop(name, None)
        if t0 is None:
            return
        self._record(name, time.perf_counter() - t0)

    # ── output ───────────────────────────────────────────────────────

    def summary(self):
        """Print a formatted summary table to stdout."""
        if not self.enabled or not self._stats:
            return
        total_wall = time.perf_counter() - (self._run_start or 0.0)
        n = self._frame_count or 1

        full_avg_ms = (self._stats.get("full_frame", {}).get("total", 0.0)
                       / max(self._stats.get("full_frame", {}).get("count", 1), 1)
                       * 1000.0)

        overall_fps = n / total_wall if total_wall > 0 else 0.0

        hdr = (f"  Frames: {n}  │  Wall time: {total_wall:.1f}s  "
               f"│  Overall FPS: {overall_fps:.1f}")

        # ── Column layout ────────────────────────────────────────────
        # Each row has 6 columns separated by 7 border chars (║):
        #   C1 = 22  (stage name)
        #   C2..C5 = 8 each  (Avg ms / Min ms / Max ms / FPS)
        #   C6 = w - 61  (% Frame)
        # Total: 7 + 22 + 4×8 + C6 = 61 + C6 = w  →  C6 = w - 61
        C1 = 22
        CN = 8   # width of each numeric column
        w  = max(72, len(hdr) + 4)
        C6 = w - 61

        def _sep(left: str, mid: str, right: str) -> str:
            """Build a full-width horizontal separator row."""
            return left + "═" * C1 + mid + ("═" * CN + mid) * 4 + "═" * C6 + right

        def _row(stage: str, avg: float, mn: float, mx: float,
                 fps_str: str, pct_str: str) -> str:
            """Build one data row with consistent column widths."""
            return ("║" + f" {stage}".ljust(C1)
                    + "║" + f" {avg:7.1f}".ljust(CN)
                    + "║" + f" {mn:7.1f}".ljust(CN)
                    + "║" + f" {mx:7.1f}".ljust(CN)
                    + "║" + f" {fps_str:>7s}".ljust(CN)
                    + "║" + f" {pct_str:>9s}".ljust(C6)
                    + "║")

        print()
        print("╔" + "═" * (w - 2) + "╗")
        print("║" + " PIPELINE PERFORMANCE PROFILE".center(w - 2) + "║")
        print("║" + hdr.center(w - 2) + "║")
        print(_sep("╠", "╦", "╣"))
        print("║" + " Stage".ljust(C1)
              + "║" + " Avg ms ".ljust(CN)
              + "║" + " Min ms ".ljust(CN)
              + "║" + " Max ms ".ljust(CN)
              + "║" + "   FPS  ".ljust(CN)
              + "║" + " % Frame ".ljust(C6)
              + "║")
        print(_sep("╠", "╬", "╣"))

        for name in self._stage_order:
            s = self._stats[name]
            cnt = s["count"]
            avg = s["total"] / cnt * 1000.0 if cnt else 0.0
            mn = s["min"] * 1000.0
            mx = s["max"] * 1000.0
            fps = 1000.0 / avg if avg > 0 else math.inf
            pct = (avg / full_avg_ms * 100.0) if full_avg_ms > 0 else 0.0

            fps_str = f"{fps:.1f}" if fps < 1e6 else "-"
            pct_str = f"{pct:.1f}%" if name != "full_frame" else "100.0%"
            if name == "visualization":
                pct_str = "(periodic)"

            print(_row(name, avg, mn, mx, fps_str, pct_str))

        print(_sep("╚", "╩", "╝"))
        print()

    def to_csv(self, path: str):
        """Export per-frame timing history to a CSV file."""
        if not self.enabled or not self._stats:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        max_len = max(len(s["history"]) for s in self._stats.values())
        with open(p, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame"] + self._stage_order)
            for i in range(max_len):
                row = [i]
                for name in self._stage_order:
                    h = self._stats[name]["history"]
                    row.append(f"{h[i] * 1000.0:.3f}" if i < len(h) else "")
                writer.writerow(row)
        print(f"Profiling CSV saved to: {p}")

    # ── internals ────────────────────────────────────────────────────

    def _record(self, name: str, elapsed: float):
        if name not in self._stats:
            self._stats[name] = {
                "count": 0, "total": 0.0,
                "min": float("inf"), "max": 0.0,
                "history": [],
            }
            self._stage_order.append(name)

        s = self._stats[name]
        s["count"] += 1
        s["total"] += elapsed
        s["min"] = min(s["min"], elapsed)
        s["max"] = max(s["max"], elapsed)
        s["history"].append(elapsed)

    def _print_live(self):
        """Print a compact one-line live report."""
        parts = []
        for name in self._stage_order:
            s = self._stats[name]
            if s["count"] == 0:
                continue
            avg_ms = s["total"] / s["count"] * 1000.0
            parts.append(f"{name}={avg_ms:.1f}ms")

        overall = self._frame_count / (time.perf_counter() - self._run_start) \
            if self._run_start else 0.0
        header = f"[Profile F{self._frame_count} | {overall:.1f} FPS]"
        print(f"{header}  {' │ '.join(parts)}")