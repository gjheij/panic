# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from panic.pipeline import _cv_mean_score

opj = os.path.join

ArrayLike = Union[np.ndarray, Any]
PathLike = Union[str, os.PathLike[str], Path]
ArtifactDict = Dict[str, np.ndarray]
PluginReturn = Union[float, Tuple[float, ArtifactDict]]


def decoding_plugin(
        X_path: Union[PathLike, ArrayLike],
        labels: ArrayLike,
        cfg: Mapping[str, Any],
        *,
        folds: Optional[list] = None,
        cols: Optional[np.ndarray] = None,
        groups: Optional[ArrayLike] = None,
        permute: bool = False,
        rng: Optional[np.random.Generator] = None,
        return_artifacts: bool = False,
        **kwargs: Any,
    ) -> PluginReturn:
    """
    Perform cross-validated decoding using the standard PANIC pipeline.

    This plugin is a thin wrapper around :func:`panic.pipeline._cv_mean_score`
    and serves as the default analysis backend for ROI and searchlight
    decoding. It computes the mean cross-validation score across the supplied
    folds and optionally returns an empty artifact dictionary to conform to the
    generic plugin interface.

    Parameters
    ----------
    X_path : str, path-like, or array-like
        Feature matrix or path to a joblib-dumped feature matrix with shape
        ``(n_samples, n_features)``.
    labels : array-like of shape (n_samples,)
        Target labels aligned to rows of ``X``.
    cfg : mapping
        Decoding configuration dictionary used to construct the estimator,
        preprocessing pipeline, scoring function, and cross-validation
        behavior.
    folds : list of tuple(ndarray, ndarray), optional
        Outer cross-validation folds as ``(train_idx, test_idx)`` pairs.
    cols : array-like of int, optional
        Optional feature subset. Primarily used for searchlight analyses to
        restrict evaluation to a local voxel neighborhood.
    groups : array-like, optional
        Group labels (e.g., runs or subjects) used for grouped
        cross-validation and/or within-group permutations.
    permute : bool, default=False
        If True, labels are permuted according to the decoding configuration
        before fitting and scoring.
    rng : numpy.random.Generator, optional
        Random number generator controlling permutations and estimator
        random states.
    return_artifacts : bool, default=False
        Included for compatibility with the plugin API. Decoding artifacts
        are handled separately through ``_save_pipeline``; therefore this
        plugin returns an empty artifact dictionary when enabled.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`panic.pipeline._cv_mean_score`.

    Returns
    -------
    float or tuple
        Mean cross-validated decoding score.

        If ``return_artifacts=False``:

        >>> score

        If ``return_artifacts=True``:

        >>> score, {}

        The empty artifact dictionary is returned because decoding-specific
        outputs (weights, predictions, contributions, fitted estimators, etc.)
        are saved separately via :func:`panic.pipeline._save_pipeline`.

    Notes
    -----
    This plugin is intentionally lightweight and delegates all decoding logic
    to :func:`panic.pipeline._cv_mean_score`. It exists primarily to provide a
    consistent plugin interface alongside representational analyses such as
    dimensionality estimation and CS–US similarity.

    See Also
    --------
    panic.pipeline._cv_mean_score
        Core cross-validated decoding implementation.
    panic.pipeline._save_pipeline
        Saves fitted estimators and decoding artifacts.
    dimensionality_plugin
        PCA participation-ratio dimensionality analysis.
    cosine_similarity_plugin
        CS-to-US representational similarity analysis.
    """
    score = _cv_mean_score(
        X_path,
        labels,
        folds,
        cfg,
        cols=cols,
        groups=groups,
        permute=permute,
        rng=rng,
        **kwargs,
    )

    if return_artifacts:
        return float(score), {}

    return float(score)
