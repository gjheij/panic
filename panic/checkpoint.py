"""Atomic, parent-owned searchlight checkpoints (trusted local joblib files)."""

import os
import logging
import sys
import tempfile
import time

import joblib

logger = logging.getLogger(__name__)


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
        destination = os.path.abspath(self.path)
        directory = os.path.dirname(destination)
        fd, temporary = None, None
        stage = 'log destination'
        try:
            logger.info("Saving searchlight checkpoint: %s", destination)
            stage = 'create checkpoint directory'
            os.makedirs(directory, exist_ok=True)
            stage = 'create temporary file'
            fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=directory)
            stage = 'open temporary file'
            stream = os.fdopen(fd, "wb")
            fd = None
            with stream:
                stage = 'serialization'
                joblib.dump({
                    "version": 2,
                    "fingerprint": self.fingerprint,
                    "seeds": self.seeds,
                    "results": self.results,
                    "failures": self.failures,
                    "fit_guard_history": self.fit_guard_history,
                }, stream)
                stage = 'flush'
                stream.flush()
                stage = 'fsync'
                os.fsync(stream.fileno())
                stage = 'close temporary file'
            stage = 'replacement'
            os.replace(temporary, destination)
        except OSError as error:
            message = (f"Checkpoint save failed during {stage}; temporary={temporary!r}; "
                       f"{error.strerror or str(error)}")
            try:
                logger.error("%s; destination=%s; errno=%s", message, destination, error.errno)
            except OSError:
                pass
            raise OSError(error.errno, message, destination) from error
        finally:
            primary_failure = sys.exc_info()[0] is not None
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                except OSError as error:
                    if not primary_failure:
                        raise OSError(error.errno, f"Checkpoint temporary cleanup failed; temporary={temporary!r}", destination) from error
                    try:
                        logger.warning("Checkpoint cleanup failed: destination=%s; temporary=%s; %s",
                                       destination, temporary, error)
                    except OSError:
                        pass
