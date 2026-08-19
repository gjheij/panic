"""Lightweight diagnostics for long-running process-based searchlights.

The monitor is intentionally independent of joblib. Workers write a single
atomic heartbeat JSON file per PID into node-local runtime storage. The parent
maintains an exact completion bitmap and periodically copies a compact snapshot
(including current worker heartbeats) to a persistent diagnostics directory.
"""

from __future__ import annotations

import faulthandler
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


_WORKER_TASK_COUNTS: Dict[int, int] = {}
_WORKER_COMPLETION_FILES: Dict[int, Any] = {}
_FAULTHANDLER_INSTALLED = False


def install_faulthandler() -> None:
    """Enable traceback dumping and register SIGUSR1 when available."""
    global _FAULTHANDLER_INSTALLED
    if _FAULTHANDLER_INSTALLED:
        return

    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    if hasattr(signal, "SIGUSR1"):
        try:
            faulthandler.register(signal.SIGUSR1, all_threads=True)
        except (OSError, RuntimeError, ValueError):
            pass

    _FAULTHANDLER_INSTALLED = True


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fobj:
        json.dump(payload, fobj, sort_keys=True)
        fobj.flush()
    os.replace(tmp, path)


def worker_heartbeat(
    runtime_dir: Optional[str],
    ix: int,
    stage: str,
    **extra: Any,
) -> None:
    """Record the current center/stage for this worker PID.

    ``runtime_dir`` should preferably be node-local storage. Each worker only
    overwrites ``worker_<pid>.json`` so file count remains O(n_workers).
    """
    if not runtime_dir:
        return

    pid = os.getpid()
    payload: Dict[str, Any] = {
        "pid": pid,
        "ppid": os.getppid(),
        "ix": int(ix),
        "stage": str(stage),
        "time": time.time(),
    }
    payload.update(extra)

    try:
        _atomic_json(Path(runtime_dir) / f"worker_{pid}.json", payload)
    except OSError:
        # Diagnostics must never make a searchlight task fail.
        pass



def worker_mark_completed(runtime_dir: Optional[str], ix: int) -> None:
    """Append a completed center index to this worker's node-local journal."""
    if not runtime_dir:
        return

    pid = os.getpid()
    try:
        fobj = _WORKER_COMPLETION_FILES.get(pid)
        if fobj is None or fobj.closed:
            path = Path(runtime_dir) / f"completed_{pid}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            fobj = open(path, "a", encoding="ascii", buffering=1)
            _WORKER_COMPLETION_FILES[pid] = fobj
        fobj.write(f"{int(ix)}\n")
        fobj.flush()
    except OSError:
        pass

def worker_task_started() -> int:
    """Increment and return this process's searchlight-task counter."""
    pid = os.getpid()
    count = _WORKER_TASK_COUNTS.get(pid, 0) + 1
    _WORKER_TASK_COUNTS[pid] = count
    return count


def process_resource_snapshot(x_path: Optional[str] = None) -> Dict[str, Any]:
    """Return Linux process resource counters useful for leak diagnosis."""
    pid = os.getpid()
    out: Dict[str, Any] = {"pid": pid}

    try:
        out["fd_count"] = len(os.listdir("/proc/self/fd"))
    except OSError:
        out["fd_count"] = None

    maps = []
    try:
        with open("/proc/self/maps", "r", encoding="utf-8", errors="replace") as fobj:
            maps = fobj.readlines()
        out["map_count"] = len(maps)
    except OSError:
        out["map_count"] = None

    if x_path and maps:
        needle = os.path.basename(os.fspath(x_path))
        out["x_map_count"] = sum(needle in line for line in maps)
    else:
        out["x_map_count"] = None

    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fobj:
            for line in fobj:
                if line.startswith("VmRSS:"):
                    out["rss_kb"] = int(line.split()[1])
                elif line.startswith("VmSize:"):
                    out["vmsize_kb"] = int(line.split()[1])
                elif line.startswith("Threads:"):
                    out["threads"] = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass

    return out


def maybe_log_worker_resources(
    logger,
    task_count: int,
    every: int,
    x_path: Optional[str] = None,
) -> None:
    """Log a worker resource snapshot every ``every`` tasks."""
    if every <= 0 or task_count % every:
        return

    snap = process_resource_snapshot(x_path=x_path)
    logger.warning(
        "SEARCHLIGHT_WORKER_RESOURCE task_count=%d snapshot=%s",
        task_count,
        snap,
    )


class SearchlightMonitor:
    """Parent-side exact completion tracker with periodic watchdog snapshots."""

    def __init__(
        self,
        total: int,
        *,
        runtime_dir: str,
        snapshot_dir: str,
        logger,
        interval: float = 60.0,
        log_every: int = 5000,
    ) -> None:
        self.total = int(total)
        self.runtime_dir = Path(runtime_dir)
        self.snapshot_dir = Path(snapshot_dir)
        self.logger = logger
        self.interval = max(float(interval), 1.0)
        self.log_every = max(int(log_every), 0)

        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._completed = np.zeros(self.total, dtype=np.bool_)
        self._n_completed = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._completion_offsets: Dict[str, int] = {}

    def start(self) -> "SearchlightMonitor":
        install_faulthandler()
        self.dump_snapshot(reason="start")
        self._thread = threading.Thread(
            target=self._watchdog,
            name="searchlight-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def mark_completed(self, ix: int) -> int:
        ix = int(ix)
        newly_completed = False
        with self._lock:
            if not self._completed[ix]:
                self._completed[ix] = True
                self._n_completed += 1
                newly_completed = True
            n_completed = self._n_completed

        if newly_completed and self.log_every and n_completed % self.log_every == 0:
            self.logger.info(
                "Searchlight exact progress: %d/%d (%.1f%%)",
                n_completed,
                self.total,
                100.0 * n_completed / self.total,
            )
            self.dump_snapshot(reason="progress")

        return n_completed

    def _sync_worker_completions(self) -> None:
        """Merge per-worker completion journals into the parent bitmap."""
        completed_ix = []
        for path in sorted(self.runtime_dir.glob("completed_*.log")):
            key = os.fspath(path)
            offset = self._completion_offsets.get(key, 0)
            try:
                with open(path, "r", encoding="ascii") as fobj:
                    fobj.seek(offset)
                    for line in fobj:
                        line = line.strip()
                        if line:
                            completed_ix.append(int(line))
                    self._completion_offsets[key] = fobj.tell()
            except (OSError, ValueError):
                continue

        if not completed_ix:
            return

        with self._lock:
            for ix in completed_ix:
                if 0 <= ix < self.total and not self._completed[ix]:
                    self._completed[ix] = True
                    self._n_completed += 1

    def outstanding(self) -> np.ndarray:
        self._sync_worker_completions()
        with self._lock:
            return np.flatnonzero(~self._completed).astype(np.int64, copy=False)

    def _read_heartbeats(self) -> list[Dict[str, Any]]:
        heartbeats = []
        for path in sorted(self.runtime_dir.glob("worker_*.json")):
            try:
                with open(path, "r", encoding="utf-8") as fobj:
                    heartbeats.append(json.load(fobj))
            except (OSError, json.JSONDecodeError):
                continue
        return heartbeats

    def dump_snapshot(self, reason: str = "watchdog") -> None:
        outstanding = self.outstanding()
        npy_tmp = self.snapshot_dir / ".searchlight_outstanding.tmp.npy"
        npy_final = self.snapshot_dir / "searchlight_outstanding.npy"
        try:
            with open(npy_tmp, "wb") as fobj:
                np.save(fobj, outstanding)
                fobj.flush()
                os.fsync(fobj.fileno())
            os.replace(npy_tmp, npy_final)
        except OSError:
            pass

        heartbeats = self._read_heartbeats()
        payload = {
            "time": time.time(),
            "reason": reason,
            "parent_pid": os.getpid(),
            "total": self.total,
            "completed": self.total - int(outstanding.size),
            "outstanding_count": int(outstanding.size),
            "outstanding_first": outstanding[:100].tolist(),
            "workers": heartbeats,
        }
        try:
            _atomic_json(self.snapshot_dir / "searchlight_monitor.json", payload)
        except OSError:
            pass

        if reason == "watchdog":
            self.logger.warning(
                "SEARCHLIGHT_WATCHDOG completed=%d/%d outstanding=%d first=%s workers=%s",
                payload["completed"],
                self.total,
                payload["outstanding_count"],
                payload["outstanding_first"][:30],
                [
                    {
                        "pid": hb.get("pid"),
                        "ix": hb.get("ix"),
                        "stage": hb.get("stage"),
                        "age_s": round(max(0.0, time.time() - hb.get("time", time.time())), 1),
                    }
                    for hb in heartbeats
                ],
            )

    def _watchdog(self) -> None:
        while not self._stop.wait(self.interval):
            self.dump_snapshot(reason="watchdog")

    def close(self, reason: str = "complete") -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(self.interval, 5.0))
        self.dump_snapshot(reason=reason)

    def __enter__(self) -> "SearchlightMonitor":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(reason="exception" if exc_type else "complete")
