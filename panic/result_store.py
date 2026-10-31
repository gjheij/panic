"""Concurrent, atomic updates to PANIC's shared participant tables (POSIX)."""
import csv
import fcntl
import io
import os
from pathlib import Path
import tempfile

import pandas as pd


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
    fd, temporary = tempfile.mkstemp(dir=str(Path(path).parent))
    try:
        with os.fdopen(fd, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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
