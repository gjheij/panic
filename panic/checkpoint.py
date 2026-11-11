"""Atomic, parent-owned searchlight checkpoints (trusted local joblib files)."""

import os
import tempfile
import time

import joblib


class SearchlightCheckpoint:
    """Persist indexed results and seeds, rejecting incompatible analyses."""

    def __init__(self, path, fingerprint, seeds, n_centers):
        self.path = path
        self.fingerprint = fingerprint
        self.seeds = seeds
        self.results = [None] * n_centers
        self.failures = {}
        self.fit_guard_history = []
        self.loaded_version = None
        self.restored_count = 0
        if os.path.exists(path):
            state = joblib.load(path)
            if (state.get("version") not in (1, 2)
                    or state.get("fingerprint") != fingerprint
                    or len(state.get("results", [])) != n_centers
                    or len(state.get("seeds", [])) != n_centers):
                raise ValueError(
                    f"Incompatible searchlight checkpoint: {path}. "
                    "Use the original inputs/settings or remove the checkpoint "
                    "to start a new analysis."
                )
            self.results = state["results"]
            self.seeds = state["seeds"]
            self.failures = state.get("failures", {})
            self.fit_guard_history = state.get("fit_guard_history", [])
            self.loaded_version = state['version']
            self.restored_count = sum(row is not None for row in self.results)
            if not isinstance(self.failures, dict) or any(not isinstance(key,str) or not key.isdigit() or not 0 <= int(key) < n_centers for key in self.failures):
                raise ValueError('Invalid searchlight checkpoint failure index')
            if not isinstance(self.fit_guard_history, list):
                raise ValueError('Invalid searchlight checkpoint guard history')

    def record_guard(self, policy):
        self.fit_guard_history.append(dict(time=time.time(), policy=dict(policy),
            protocol_version=1, prior_failed_centers=len(self.failures),
            restored_centers=self.restored_count, loaded_version=self.loaded_version,
            legacy_fit_status_unverified=self.loaded_version == 1))

    def save(self):
        """Replace the last checkpoint only after a complete, flushed write."""
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                joblib.dump({
                    "version": 2,
                    "fingerprint": self.fingerprint,
                    "seeds": self.seeds,
                    "results": self.results,
                    "failures": self.failures,
                    "fit_guard_history": self.fit_guard_history,
                }, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
