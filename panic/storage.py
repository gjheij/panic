"""Execution-only temporary storage policy; never use for checkpoints."""
import os
from pathlib import Path
import tempfile


class ResolvedTempDir(str):
    """An already resolved and checked directory, reusable within one run."""


def resolve_tmp_dir(explicit=None):
    """Explicit setting -> TMPDIR -> Python default; never silently fall back."""
    if isinstance(explicit, ResolvedTempDir):
        return explicit
    selected = explicit if explicit is not None else (os.environ.get('TMPDIR') or tempfile.gettempdir())
    if not isinstance(selected, (str, os.PathLike)) or not os.fspath(selected).strip():
        raise ValueError('Temporary directory must be a nonempty path or null')
    path = Path(selected).expanduser().absolute()
    try:
        path.mkdir(parents=True, exist_ok=True)
        # Validate the selected directory itself, without fallback to another disk.
        with tempfile.TemporaryFile(dir=path):
            pass
    except OSError as error:
        raise OSError(error.errno, 'Cannot use selected temporary directory: '+str(error), str(path)) from error
    return ResolvedTempDir(str(path))
