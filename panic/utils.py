# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations

import os
import sys
import yaml
import json
import hashlib
import numpy as np
import nibabel as nib
from joblib import load
from pathlib import Path
from importlib.resources import files, as_file
from typing import Any, Dict, Optional, Union, Tuple

import panic
from panic.logger import get_logger

logger = get_logger(__name__)
opj = os.path.join

ArrayLike = Union[np.ndarray, Any]
PathLike = Union[str, os.PathLike[str], Path]

def get_config_path(filename="config.yml") -> Path:
    resource = files("panic").joinpath(filename)

    if resource.is_file():
        with as_file(resource) as path:
            return Path(path)

    search_locations = getattr(panic.__spec__, "submodule_search_locations", []) or []

    candidates = []

    for loc in search_locations:
        loc = Path(loc).resolve()
        candidates.extend([
            loc / filename,
            loc / "panic" / filename,
        ])

    panic_file = getattr(panic, "__file__", None)
    if panic_file:
        candidates.append(Path(panic_file).resolve().parent / filename)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find {filename!r}. Tried:\n"
        + "\n".join(str(c) for c in candidates)
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def tqdm_disabled():
    return (not sys.stderr.isatty()) # or bool(os.environ.get("PYTEST_CURRENT_TEST"))


def load_feature_matrix(
        X_path: Union[PathLike, ArrayLike],
        *,
        cols: Optional[np.ndarray] = None,
    ) -> np.ndarray:
    """
    Load and optionally subset a feature matrix.

    Parameters
    ----------
    X_path : str, path-like, or array-like
        Either a path to a joblib-dumped feature matrix or an already-loaded
        matrix. The matrix is expected to have shape ``(n_samples, n_features)``.
    cols : array-like of int, optional
        Feature columns to select. Used for searchlight neighbourhoods.

    Returns
    -------
    numpy.ndarray
        Materialized feature matrix. If ``cols`` is provided, the returned array
        has shape ``(n_samples, len(cols))``.
    """
    if isinstance(X_path, (str, os.PathLike, Path)):
        X_mm = load(X_path, mmap_mode="r")
    else:
        X_mm = X_path

    if cols is not None:
        return np.asarray(X_mm[:, cols])

    return np.asarray(X_mm)


def load_mask(
        mask: Any,
        return_affine: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Load a mask from a path, NIfTI image, or array-like object."""
    affine = None

    if isinstance(mask, (str, os.PathLike)):
        mask = nib.load(mask)

    if isinstance(mask, nib.spatialimages.SpatialImage):
        affine = np.asarray(mask.affine)
        mask = mask.get_fdata()

    mask_array = np.asarray(mask)

    if return_affine:
        return mask_array, affine

    return mask_array


def make_analysis_id(
    *,
    analysis_name,
    analysis_type,
    source,
    method,
    standardize,
    length=12,
):
    """Create a deterministic identifier for a PANIC analysis.

    The identifier captures the high-level analysis identity and permutation
    strategy, while deliberately ignoring lower-level estimator, feature
    selection, and cross-validation configuration.

    Parameters
    ----------
    analysis_name : str
        Human-readable analysis name.

    analysis_type : str
        Analysis/plugin type.

    source : str
        Source of the beta estimates.

    method : str
        Beta-estimation method, such as ``"LSA"`` or ``"LSS"``.

    standardize : str, bool, or None
        Standardization strategy applied to the beta estimates.

    dec_settings : mapping
        Decoding configuration containing permutation settings.

    length : int, default=12
        Number of hexadecimal SHA-256 characters retained.

    Returns
    -------
    str
        Deterministic analysis identifier.
    """
    payload = {
        "analysis_name": str(analysis_name),
        "analysis_type": str(analysis_type),
        "source": str(source),
        "method": str(method),
        "standardize": standardize
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()[:length]
