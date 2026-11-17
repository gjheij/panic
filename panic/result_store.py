"""Concurrent, atomic updates to PANIC's shared participant tables (POSIX)."""
import csv
from contextlib import contextmanager
import fcntl
import io
import logging
import os
from pathlib import Path
import stat
import sys
import tempfile

import pandas as pd

logger = logging.getLogger(__name__)


@contextmanager
def atomic_csv_writer(path, *, prefix='.csv-write-', create_parents=False):
    """Write/fsync/close a sibling temporary file before replacing one CSV.

    Existing output survives serialization, flush, fsync, close, and replace
    failures. Atomicity is per file, not across the results/null table pair.
    Error.filename is always the intended destination; the temporary path and
    failing stage are retained in its message and original exception chain.
    """
    destination = Path(path).absolute()
    fd, temporary = None, None
    stage = 'log destination'
    try:
        logger.info("Writing CSV: %s", destination)
        stage = 'prepare destination'
        if create_parents:
            destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            mode = stat.S_IMODE(destination.stat().st_mode)
        except FileNotFoundError:
            mode = None
        stage = 'create temporary file'
        fd, temporary = tempfile.mkstemp(dir=str(destination.parent), prefix=prefix)
        if mode is not None:
            stage = 'preserve file permissions'
            os.fchmod(fd, mode)
        stage = 'open temporary file'
        stream = os.fdopen(fd, 'w', encoding='utf-8', newline='')
        fd = None  # stream now owns this descriptor, including close failures.
        with stream:
            stage = 'write data'
            yield stream
            stage = 'flush data'
            stream.flush()
            stage = 'fsync data'
            os.fsync(stream.fileno())
            stage = 'close temporary file'
        stage = 'replace destination'
        os.replace(temporary, destination)
    except OSError as error:
        message = (f"CSV write failed during {stage}; temporary={temporary!r}; "
                   f"{error.strerror or str(error)}")
        try:
            logger.error("%s; destination=%s; errno=%s", message, destination, error.errno)
        except OSError:
            pass  # A full log filesystem must not mask the destination-aware error.
        raise OSError(error.errno, message, str(destination)) from error
    finally:
        primary_failure = sys.exc_info()[0] is not None
        if fd is not None:
            try:
                os.close(fd)
            except OSError as error:
                try:
                    logger.warning("CSV descriptor cleanup failed: destination=%s; %s", destination, error)
                except OSError:
                    pass
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            except OSError as error:
                if not primary_failure:
                    raise OSError(error.errno, f"CSV temporary cleanup failed; temporary={temporary!r}", str(destination)) from error
                try:
                    logger.warning("CSV temporary cleanup failed: destination=%s; temporary=%s; %s",
                                   destination, temporary, error)
                except OSError:
                    pass


def write_dataframe_csv(frame, path, **kwargs):
    """Pandas CSV/TSV serialization with destination-aware atomic I/O."""
    # Append/exclusive/binary modes have different publication semantics.
    if kwargs.pop('mode', 'w') != 'w':
        raise ValueError('Atomic CSV writes support mode="w" only')
    if kwargs.get('compression') not in (None, 'infer'):
        raise ValueError('Atomic CSV writes do not support compressed outputs')
    destination = Path(path)
    if destination.suffix.lower() in ('.gz', '.bz2', '.zip', '.xz', '.zst', '.tar'):
        raise ValueError('Atomic CSV writes do not support compressed outputs')
    with atomic_csv_writer(destination) as stream:
        frame.to_csv(stream, **kwargs)


def _read(path):
    if not Path(path).exists() or not Path(path).stat().st_size:
        return [], []
    with open(path, newline='') as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def _frame(frame):
    stream = io.StringIO()
    frame.to_csv(stream, index=False, na_rep='NaN')
    stream.seek(0)
    reader = csv.DictReader(stream)
    return list(reader.fieldnames or []), list(reader)


def _atomic(path, columns, rows):
    with atomic_csv_writer(path) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def merge_result_tables(result_path, null_path, results, null, *, overwrite=False, hemi_key='hemi'):
    """Reread under one lock; replace only this run's ROI/model/source cells.

    Overwrite replaces the affected cells across old analysis IDs. Incremental
    writes replace only cells with the same analysis ID. All other models and
    ROIs are preserved. Null draws are replaced as a complete set for a cell.
    """
    columns, rows = _frame(results)
    null_columns, null_rows = _frame(null)
    keys = ['subject', 'analysis', 'analysis_type', 'source', 'method', 'roi', hemi_key]
    if not overwrite:
        keys.append('analysis_id')
    scope = {tuple(row.get(key, '') for key in keys) for row in rows}
    parent = Path(result_path).parent
    parent.mkdir(parents=True, exist_ok=True)
    with (parent/'.tables.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path, new_columns, new_rows in [(result_path, columns, rows), (null_path, null_columns, null_rows)]:
            old_columns, old_rows = _read(path)
            all_columns = list(dict.fromkeys(old_columns + new_columns))
            if not all_columns:
                all_columns = columns + ['perm', 'acc']
            retained = [row for row in old_rows if tuple(row.get(key, '') for key in keys) not in scope]
            merged = retained + new_rows
            _atomic(path, all_columns, merged)
        return pd.read_csv(result_path, dtype={key: str for key in
            ("subject", "roi", "hemi", "source", "method", "analysis", "analysis_type", "analysis_id", "batch_job_id")})


def write_qc_rows(path, rows, hemi_key='hemi'):
    """Update per-run QC without touching decoding results or null draws."""
    keys = ['subject', 'analysis', 'method', 'source', 'roi', hemi_key]
    scope = {tuple(row.get(key, '') for key in keys) for row in rows}
    path = Path(path)
    with (path.parent/'.tables.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        columns, existing = _read(path)
        columns = list(dict.fromkeys(columns + [key for row in rows for key in row]))
        retained = [row for row in existing if tuple(row.get(key, '') for key in keys) not in scope]
        _atomic(path, columns, retained + rows)
