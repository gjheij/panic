"""Atomic, parent-owned searchlight checkpoints (trusted local joblib files)."""

import os
import tempfile

import joblib


class SearchlightCheckpoint:
    """Persist indexed results and seeds, rejecting incompatible analyses."""

    def __init__(self, path, fingerprint, seeds, n_centers):
        self.path = path
        self.fingerprint = fingerprint
        self.seeds = seeds
        self.results = [None] * n_centers
        if os.path.exists(path):
            state = joblib.load(path)
            if (state.get("version") != 1
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

    def save(self):
        """Replace the last checkpoint only after a complete, flushed write."""
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                joblib.dump({
                    "version": 1,
                    "fingerprint": self.fingerprint,
                    "seeds": self.seeds,
                    "results": self.results,
                }, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
