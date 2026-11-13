# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from sklearn.decomposition import PCA
from panic.utils import load_feature_matrix


ArrayLike = Union[np.ndarray, Any]
PathLike = Union[str, os.PathLike[str], Path]
ArtifactDict = Dict[str, np.ndarray]
PluginReturn = Union[float, Tuple[float, ArtifactDict]]

def dimensionality_plugin(
        X_path: Union[PathLike, ArrayLike],
        labels: ArrayLike,
        cfg: Mapping[str, Any],
        *,
        cols: Optional[np.ndarray] = None,
        groups: Optional[ArrayLike] = None,
        permute: bool = False,
        rng: Optional[np.random.Generator] = None,
        condition: Optional[Any] = None,
        return_artifacts: bool = False,
        **kwargs: Any,
    ) -> PluginReturn:
    """
    Compute effective dimensionality using the PCA participation ratio.

    The score is:

    ``(sum(lambda) ** 2) / sum(lambda ** 2)``

    where ``lambda`` are the positive PCA eigenvalues. If ``condition`` is
    provided, dimensionality is computed only for samples with that label.

    Parameters
    ----------
    X_path : str, path-like, or array-like
        Feature matrix or joblib path with shape ``(n_samples, n_features)``.
    labels : array-like
        Labels aligned to rows of ``X``.
    cfg : mapping
        Analysis configuration. Included for interface consistency; not used
        directly by this plugin.
    cols : array-like of int, optional
        Optional feature subset for searchlight evaluation.
    groups : array-like, optional
        Accepted for interface compatibility. Not used.
    permute : bool, default=False
        Accepted for interface compatibility. Not used; dimensionality does not
        require label permutation unless a custom null is implemented.
    rng : numpy.random.Generator, optional
        Accepted for interface compatibility. Not used.
    condition : object, optional
        Restrict dimensionality to samples with this label.
    return_artifacts : bool, default=False
        If True, return PCA artifacts: mean pattern, explained variance,
        explained variance ratio, PCA components, and sample scores.
    **kwargs
        Ignored extra keyword arguments accepted for plugin compatibility.

    Returns
    -------
    float or tuple
        Participation-ratio dimensionality. If ``return_artifacts=True``,
        returns ``(score, artifacts)``.
    """
    X = load_feature_matrix(X_path, cols=cols)
    y = np.asarray(labels)

    if condition is not None:
        X = X[y == condition]

    if X.shape[0] < 3 or X.shape[1] < 2:
        return (float("nan"), {}) if return_artifacts else float("nan")

    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean

    pca = PCA().fit(Xc)
    lam = pca.explained_variance_
    lam_pos = lam[lam > 0]

    score = (
        float((lam_pos.sum() ** 2) / np.sum(lam_pos ** 2))
        if lam_pos.size
        else float("nan")
    )

    if not return_artifacts:
        return score

    artifacts: ArtifactDict = {
        "mean_pattern": np.asarray(X_mean.squeeze()),
        "explained_variance": np.asarray(pca.explained_variance_),
        "explained_variance_ratio": np.asarray(pca.explained_variance_ratio_),
        "components": np.asarray(pca.components_),
        "scores": np.asarray(pca.transform(Xc)),
    }
    return score, artifacts
