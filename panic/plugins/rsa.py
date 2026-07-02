# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr

from panic.utils import load_feature_matrix

ArrayLike = Union[np.ndarray, Any]
PathLike = Union[str, os.PathLike[str], Path]
ArtifactDict = Dict[str, np.ndarray]
PluginReturn = Union[float, Tuple[float, ArtifactDict]]


def _rdm_vector(rdm: np.ndarray) -> np.ndarray:
    """Return upper-triangle vector of a square RDM."""
    rdm = np.asarray(rdm, dtype=float)
    iu = np.triu_indices_from(rdm, k=1)
    return rdm[iu]


def _validate_rdm(
        rdm: np.ndarray,
        n: int,
        name: str = "model_rdm"
    ) -> np.ndarray:
    """Validate and return a square model RDM."""
    rdm = np.asarray(rdm, dtype=float)

    if rdm.shape != (n, n):
        raise ValueError(
            f"{name} has shape {rdm.shape}, expected {(n, n)}."
        )

    if not np.allclose(rdm, rdm.T, equal_nan=True):
        raise ValueError(f"{name} must be symmetric.")

    if not np.allclose(np.diag(rdm), 0, equal_nan=True):
        raise ValueError(f"{name} diagonal must be zero.")

    return rdm


def _corr_vectors(
        a: np.ndarray,
        b: np.ndarray,
        method: str = "spearman"
    ) -> float:
    """Correlate two RDM upper-triangle vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")

    a = a[ok]
    b = b[ok]

    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")

    if method == "spearman":
        return float(spearmanr(a, b).statistic)
    if method == "pearson":
        return float(pearsonr(a, b).statistic)

    raise ValueError(f"Unknown RSA comparison method: {method!r}")


def rsa_plugin(
        X_path: Union[PathLike, ArrayLike],
        labels: ArrayLike,
        cfg: Mapping[str, Any],
        *,
        cols: Optional[np.ndarray] = None,
        groups: Optional[ArrayLike] = None,
        permute: bool = False,
        rng: Optional[np.random.Generator] = None,
        return_artifacts: bool = False,
        conditions: Optional[list[Any]] = None,
        metric: str = "correlation",
        compare: str = "spearman",
        model_rdms: Optional[Mapping[str, Any]] = None,
        target_model: Optional[str] = None,
        **kwargs: Any,
    ) -> PluginReturn:
    """
    Compute representational similarity analysis for an ROI or searchlight.

    This plugin computes a neural representational dissimilarity matrix (RDM)
    from condition-mean activity patterns, then compares that neural RDM to one
    or more model/design RDMs.

    Parameters
    ----------
    X_path : str, path-like, or array-like
        Feature matrix or joblib path with shape ``(n_samples, n_features)``.
    labels : array-like
        Integer condition labels aligned to rows of ``X``.
    cfg : mapping
        Analysis configuration. Included for plugin compatibility.
    cols : array-like of int, optional
        Optional feature subset, used by searchlight.
    groups : array-like, optional
        Accepted for plugin compatibility. Not used by default.
    permute : bool, default=False
        If True, labels are permuted before condition means are estimated.
        This gives a simple label-shuffle null for RSA.
    rng : numpy.random.Generator, optional
        Random generator used when ``permute=True``.
    return_artifacts : bool, default=False
        If True, return neural RDM, model scores, model RDMs, conditions, and
        condition-mean patterns.
    conditions : list, optional
        Ordered condition labels to include. If None, uses sorted unique labels.
        If config label resolution is enabled, condition names such as ``USp``
        can be specified and will be converted to label IDs.
    metric : str, default="correlation"
        Distance metric passed to ``scipy.spatial.distance.pdist`` for the
        neural RDM. Common options are ``correlation``, ``cosine``, and
        ``euclidean``.
    compare : {"spearman", "pearson"}, default="spearman"
        Correlation method for comparing neural and model RDM vectors.
    model_rdms : mapping, optional
        Dictionary of model RDM definitions. Each entry may either be a raw
        matrix or a dictionary with a ``matrix`` field.
    target_model : str, optional
        Name of the model RDM whose score should be returned as the plugin
        scalar. If omitted and only one model is supplied, that model is used.
        If omitted and multiple models are supplied, the first model is used.

    Returns
    -------
    float or tuple
        RSA model fit score. If ``return_artifacts=True``, returns
        ``(score, artifacts)``.

    Notes
    -----
    The returned scalar is a searchlight-compatible summary: high values mean
    that the local representational geometry resembles the selected design RDM.
    """
    X = load_feature_matrix(X_path, cols=cols)
    y = np.asarray(labels).reshape(-1)

    if permute:
        if rng is None:
            rng = np.random.default_rng()
        y = rng.permutation(y)

    if conditions is None:
        conds = list(np.unique(y))
    else:
        conds = list(conditions)

    # condition mean patterns
    patterns = []
    valid_conds = []
    for c in conds:
        idx = y == c
        if np.any(idx):
            patterns.append(X[idx].mean(axis=0))
            valid_conds.append(c)

    if len(patterns) < 3:
        return (float("nan"), {}) if return_artifacts else float("nan")

    patterns = np.vstack(patterns)
    conds = valid_conds
    n_cond = len(conds)

    neural_rdm = squareform(pdist(patterns, metric=metric))
    neural_vec = _rdm_vector(neural_rdm)

    model_scores: Dict[str, float] = {}
    model_mats: Dict[str, np.ndarray] = {}

    if not model_rdms:
        # If no model is supplied, return mean neural dissimilarity as a scalar.
        score = float(np.nanmean(neural_vec))
    else:
        for model_name, model_spec in model_rdms.items():
            mat = model_spec.get("matrix", model_spec) if isinstance(model_spec, Mapping) else model_spec
            mat = _validate_rdm(
                np.asarray(mat, dtype=float),
                n_cond,
                name=model_name
            )

            model_mats[model_name] = mat
            model_scores[model_name] = _corr_vectors(
                neural_vec,
                _rdm_vector(mat),
                method=compare,
            )

        if target_model is None:
            target_model = next(iter(model_scores))

        if target_model not in model_scores:
            raise ValueError(
                f"target_model={target_model!r} not found in model_rdms. "
                f"Available: {list(model_scores)}"
            )

        score = float(model_scores[target_model])

    if not return_artifacts:
        return score

    artifacts: ArtifactDict = {
        "rsa_conditions": np.asarray(conds),
        "rsa_patterns": np.asarray(patterns),
        "rsa_neural_rdm": np.asarray(neural_rdm),
        "rsa_neural_rdm_vec": np.asarray(neural_vec),
    }

    if model_scores:
        artifacts["rsa_model_names"] = np.asarray(list(model_scores.keys()), dtype=object)
        artifacts["rsa_model_scores"] = np.asarray(list(model_scores.values()), dtype=float)

        for name, mat in model_mats.items():
            artifacts[f"rsa_model_rdm_{name}"] = np.asarray(mat)

    return score, artifacts
