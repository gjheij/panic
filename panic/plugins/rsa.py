# -*- coding: utf-8 -*-
"""Representational similarity analysis plugin for PANIC.

The plugin accepts a subject's sample-by-feature matrix and produces a neural
representational dissimilarity matrix (RDM) for an ROI or searchlight.

Supported neural distances
--------------------------
* SciPy ``pdist`` distances (e.g. ``correlation``, ``cosine``, ``euclidean``)
  between condition-average patterns, pooled over all samples.
* ``crossnobis``: crossvalidated squared pattern distances between *different*
  partitions (normally runs). ``noise_normalization='identity'`` produces
  crossvalidated squared Euclidean distances; ``'diagonal'`` or ``'shrinkage'``
  requires independent GLM noise residuals in the same feature order and yields
  noise-normalized crossvalidated squared Mahalanobis distances.

Crossvalidated distances may be negative. They are not clipped. Condition pairs
with fewer than ``min_partitions`` independent partitions are set to NaN.
Missing conditions are never represented by all-zero patterns. All requested
conditions must be present somewhere in the subject's data so model-RDM indices
remain unambiguous.

This module deliberately does not infer GLM residual covariance from a handful
of condition-average beta patterns. Such covariance is generally not a valid
estimate of the voxelwise measurement-noise covariance.
"""

from __future__ import annotations

import os
import logging
from time import perf_counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.covariance import LedoitWolf

from panic.utils import load_feature_matrix
from .result import PluginResult

# PANIC configures root handlers and subject context centrally.
# get_logger() would reinitialize those handlers, so do not call it here.
logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, Any]
PathLike = Union[str, os.PathLike[str], Path]
ArtifactDict = Dict[str, np.ndarray]
PluginReturn = Union[float, Tuple[float, ArtifactDict], PluginResult]


def estimate_beta_noise_covariance(
    X,
    y,
    groups,
    *,
    shrinkage=True,
):
    """
    Estimate voxel noise covariance from trial-wise beta deviations.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_voxels)
        Single-trial GLMsingle beta patterns.
    y : ndarray, shape (n_trials,)
        Outcome condition labels.
    groups : ndarray, shape (n_trials,)
        Independent run identifiers.
    shrinkage : bool, default=True
        Use Ledoit-Wolf covariance shrinkage.

    Returns
    -------
    covariance : ndarray, shape (n_voxels, n_voxels)
        Estimated covariance of within-run, within-condition
        beta deviations.

    Notes
    -----
    This estimates trial-level beta variability, not the
    covariance of the original GLM time-series residuals.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    groups = np.asarray(groups)

    residuals = []
    for run in np.unique(groups):
        for condition in np.unique(y):
            mask = (groups == run) & (y == condition)

            if mask.sum() < 2:
                continue

            trials = X[mask]
            residuals.append(
                trials - trials.mean(axis=0, keepdims=True)
            )

    if not residuals:
        raise ValueError(
            "Not enough repeated trials within run-condition cells."
        )

    residuals = np.vstack(residuals)

    if shrinkage:
        return LedoitWolf(store_precision=False).fit(residuals).covariance_

    return np.cov(residuals, rowvar=False)


def _rdm_vector(rdm: np.ndarray) -> np.ndarray:
    """Return the strictly upper-triangular entries of a square RDM."""
    rdm = np.asarray(rdm, dtype=float)
    if rdm.ndim != 2 or rdm.shape[0] != rdm.shape[1]:
        raise ValueError("RDM must be a square 2D matrix.")
    return rdm[np.triu_indices(rdm.shape[0], k=1)]


def _validate_rdm(rdm: np.ndarray, n: int, name: str = "model_rdm") -> np.ndarray:
    """Validate a model RDM's shape, symmetry, diagonal, and entries."""
    rdm = np.asarray(rdm, dtype=float)
    if rdm.shape != (n, n):
        raise ValueError(f"{name} has shape {rdm.shape}; expected {(n, n)}.")
    if not np.isfinite(rdm).all():
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(rdm, rdm.T):
        raise ValueError(f"{name} must be symmetric.")
    if not np.allclose(np.diag(rdm), 0):
        raise ValueError(f"{name} diagonal must be zero.")
    if np.any(rdm < -1e-10):
        raise ValueError(f"{name} must have nonnegative dissimilarities.")
    return rdm


def _corr_vectors(a: np.ndarray, b: np.ndarray, method: str = "spearman",
                  min_pairs: int = 3) -> float:
    """Correlate RDM vectors using only shared finite entries.

    Returns NaN if fewer than ``min_pairs`` entries are usable or if either
    vector is constant. The result is descriptive, not a p-value.
    """
    if method not in {"spearman", "pearson"}:
        raise ValueError(f"Unknown RSA comparison method: {method!r}")
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("RDM vectors must have identical lengths.")
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < min_pairs:
        return float("nan")
    a, b = a[valid], b[valid]
    if np.ptp(a) <= 1e-14 or np.ptp(b) <= 1e-14:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(a, b).statistic)
    return float(pearsonr(a, b).statistic)


def _load_noise_residuals(noise_residuals: Any, cols: Optional[np.ndarray],
                          n_features: int) -> np.ndarray:
    """Load residual samples, respecting searchlight feature selection.

    Array input or ``.npy`` paths are supported. A 2D matrix already subset to
    the current features is accepted; a full-feature matrix is subset by
    ``cols``. Paths are loaded with ``allow_pickle=False``.
    """
    if isinstance(noise_residuals, (str, os.PathLike, Path)):
        path = Path(noise_residuals)
        if path.suffix != ".npy":
            raise ValueError("noise_residuals path must point to a .npy file.")
        residuals = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        residuals = np.asarray(noise_residuals)
    if residuals.ndim != 2:
        raise ValueError("noise_residuals must have shape (observations, features).")
    if residuals.shape[1] != n_features and cols is not None:
        residuals = residuals[:, np.asarray(cols)]
    residuals = np.asarray(residuals, dtype=float)
    if residuals.shape[1] != n_features:
        raise ValueError("Residual feature count does not match X after selection.")
    if residuals.shape[0] < 3 or not np.isfinite(residuals).all():
        raise ValueError("Residuals require >=3 finite observations.")
    return residuals


def _precision_matrix(n_features: int, noise_normalization: str,
                      noise_residuals: Optional[np.ndarray]) -> np.ndarray:
    """Construct voxel precision for crossvalidated pattern distances.

    ``identity`` requires no residuals; ``diagonal`` uses inverse residual
    variances; ``shrinkage`` uses Ledoit-Wolf covariance regularization.
    Residuals should be independently obtained from the GLM, not inferred from
    the condition means used to form the neural RDM.
    """
    if noise_normalization == "identity":
        return np.eye(n_features, dtype=float)
    if noise_normalization not in {"diagonal", "shrinkage"}:
        raise ValueError("noise_normalization must be identity, diagonal, or shrinkage.")
    if noise_residuals is None:
        raise ValueError(
            f"noise_normalization={noise_normalization!r} requires "
            "noise_residuals from the first-level GLM."
        )
    if noise_normalization == "diagonal":
        variances = np.var(noise_residuals, axis=0, ddof=1)
        if not np.isfinite(variances).all() or np.any(variances <= 0):
            raise ValueError("Residual variances must be finite and positive.")
        return np.diag(1.0 / variances)
    started = perf_counter()
    logger.debug("RSA Ledoit-Wolf fitting: residuals=%s", noise_residuals.shape)
    estimator = LedoitWolf(assume_centered=True, store_precision=False).fit(noise_residuals)
    covariance = estimator.covariance_
    logger.debug("RSA covariance fitted: shrinkage=%.6g elapsed=%.3fs",
                 estimator.shrinkage_, perf_counter() - started)
    try:
        started = perf_counter()
        precision = np.linalg.inv(covariance)
        logger.debug("RSA precision inversion: features=%d elapsed=%.3fs",
                     n_features, perf_counter() - started)
        return precision
    except np.linalg.LinAlgError as exc:
        raise ValueError("Shrinkage noise covariance is not invertible.") from exc


def crossnobis_rdm(
    X: np.ndarray,
    labels: ArrayLike,
    groups: ArrayLike,
    conditions: list[Any],
    *,
    noise_residuals: Optional[np.ndarray] = None,
    noise_normalization: str = "identity",
    min_partitions: int = 2,
    normalize_by_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a crossvalidated squared pattern-distance RDM.

    Each condition's samples are averaged *within each run*. For each pair of
    conditions, differences from distinct runs are cross-multiplied with a
    shared feature precision matrix. The average of all ordered cross-run
    products is returned, optionally divided by feature count.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Trial-wise beta patterns or independent run-wise condition patterns.
    labels : array-like, shape (n_samples,)
        Condition IDs aligned with ``X`` rows.
    groups : array-like, shape (n_samples,)
        Independent partition IDs, typically acquisition runs. Must be supplied.
    conditions : list
        Fixed model-RDM condition order. Every condition must be observed in at
        least one partition; individual partitions may omit conditions.
    noise_residuals : ndarray, optional
        First-level GLM residual observations, aligned with ``X`` features.
        Required for ``diagonal`` and ``shrinkage`` normalization.
    noise_normalization : {'identity', 'diagonal', 'shrinkage'}
        ``identity`` yields crossvalidated squared Euclidean distance, *not*
        noise-normalized Mahalanobis. The other modes whiten using residual
        variance or a regularized residual covariance.
    min_partitions : int, default=2
        Minimum number of partitions containing *both* conditions.
    normalize_by_features : bool, default=True
        Divide the cross-product by the number of features.

    Returns
    -------
    rdm : ndarray, shape (n_conditions, n_conditions)
        Symmetric RDM, zero diagonal, NaN for insufficient cross-run pairs.
        Finite off-diagonal estimates can be negative.
    pair_counts : ndarray, shape (n_conditions, n_conditions)
        Number of ordered independent-run products per pair, ``m*(m-1)`` for
        ``m`` runs containing both conditions.
    trial_counts : ndarray, shape (n_runs, n_conditions)
        Number of samples contributing to each run-specific condition mean.
    run_patterns : ndarray, shape (n_runs, n_conditions, n_features)
        Condition means by run, NaN when missing.

    Notes
    -----
    Cross-run products assume independent measurement errors across runs. If
    the input beta estimates have correlated errors across partitions, the
    estimator need not be unbiased. Noise normalization is a separate choice
    from crossvalidation and should use GLM residuals in matching voxel order.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(labels).reshape(-1)
    g = np.asarray(groups).reshape(-1)
    conds = list(conditions)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X must be a nonempty 2D sample-by-feature matrix.")
    if len(X) != len(y) or len(X) != len(g):
        raise ValueError("X, labels, and groups must have equal sample counts.")
    if not np.isfinite(X).all():
        raise ValueError("X contains nonfinite values.")
    if not 2 <= min_partitions:
        raise ValueError("min_partitions must be >= 2.")
    if len(conds) < 2 or len(set(conds)) != len(conds):
        raise ValueError("conditions must contain at least two unique IDs.")
    absent = [c for c in conds if not np.any(y == c)]
    if absent:
        raise ValueError(f"Requested conditions absent from all data: {absent}")
    runs = np.unique(g)
    if len(runs) < 2:
        raise ValueError("Crossnobis requires at least two independent partitions.")

    n_runs, n_cond, n_features = len(runs), len(conds), X.shape[1]
    run_patterns = np.full((n_runs, n_cond, n_features), np.nan)
    trial_counts = np.zeros((n_runs, n_cond), dtype=int)
    for ri, run in enumerate(runs):
        for ci, cond in enumerate(conds):
            mask = (g == run) & (y == cond)
            trial_counts[ri, ci] = int(mask.sum())
            if mask.any():
                run_patterns[ri, ci] = X[mask].mean(axis=0)

    logger.debug("RSA run-condition counts: runs=%s conditions=%s counts=%s",
                 runs.tolist(), conds, trial_counts.tolist())
    # Use residual-derived precision, never infer it from the target betas.
    precision = _precision_matrix(n_features, noise_normalization, noise_residuals)

    rdm = np.full((n_cond, n_cond), np.nan)
    pair_counts = np.zeros((n_cond, n_cond), dtype=int)
    np.fill_diagonal(rdm, 0.0)
    for i in range(n_cond):
        for j in range(i + 1, n_cond):
            valid = (trial_counts[:, i] > 0) & (trial_counts[:, j] > 0)
            differences = run_patterns[valid, i] - run_patterns[valid, j]
            m = len(differences)
            if m < min_partitions:
                continue
            cross = differences @ precision @ differences.T
            # Efficient mean of all ordered cross-partition products.
            total = cross.sum() - np.trace(cross)
            distance = total / (m * (m - 1))
            if normalize_by_features:
                distance /= n_features
            rdm[i, j] = rdm[j, i] = distance
            pair_counts[i, j] = pair_counts[j, i] = m * (m - 1)
    return rdm, pair_counts, trial_counts, run_patterns


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
    noise_normalization: str = "identity",
    noise_residuals: Optional[Any] = None,
    min_partitions: int = 2,
    normalize_by_features: bool = True,
    min_rdm_pairs: int = 3,
    **kwargs: Any,
) -> PluginReturn:
    """Compute ROI/searchlight RSA and compare the neural RDM with model RDMs.

    Parameters
    ----------
    X_path : array-like or joblib path
        Feature matrix with shape ``(n_samples, n_features)``. PANIC's loader
        selects ``cols`` when supplied.
    labels : array-like
        Condition IDs for each sample. Names in YAML ``conditions`` may be
        resolved by PANIC's plugin dispatcher using ``label_dict``.
    cfg : mapping
        PANIC configuration, retained for plugin API compatibility.
    cols : array-like, optional
        Feature indices, normally supplied for searchlight analysis.
    groups : array-like, optional
        Independent run IDs. Required for ``metric='crossnobis'``.
    permute : bool, default=False
        Shuffle condition labels *within each run* when groups are available,
        preserving run-specific condition counts. Without groups, shuffle
        globally. Exchangeability must still be justified for the design.
    rng : numpy.random.Generator, optional
        Random generator for label shuffling.
    return_artifacts : bool, default=False
        Return ``(score, artifacts)`` instead of only the scalar score.
    conditions : list, optional
        Fixed order of condition IDs in neural and model RDMs. If omitted,
        sorted unique labels are used. Every requested condition must occur
        at least once somewhere in the supplied data.
    metric : str, default='correlation'
        SciPy ``pdist`` metric for pooled condition-mean patterns, or
        ``'crossnobis'`` for crossvalidated squared distances.
    compare : {'spearman', 'pearson'}, default='spearman'
        Correlation between finite neural and model RDM upper-triangle entries.
    model_rdms : mapping, optional
        Model name to a square matrix, or ``{'matrix': matrix}``. Matrices must
        use exactly the ``conditions`` order. If omitted, return the mean of
        finite neural RDM distances as the scalar.
    target_model : str, optional
        Model whose RSA correlation is returned as the scalar. Defaults to the
        first model when several are supplied.
    noise_normalization : {'identity', 'diagonal', 'shrinkage'}
        For crossnobis only. ``identity`` is crossvalidated squared Euclidean;
        ``diagonal`` and ``shrinkage`` require GLM noise residuals.
    noise_residuals : ndarray or .npy path, optional
        First-level GLM residual samples in matching feature order. For
        searchlight use, the full residual matrix can be passed and ``cols``
        will be applied, or a preselected residual matrix can be supplied.
    min_partitions : int, default=2
        Minimum number of runs containing both conditions for each distance.
    normalize_by_features : bool, default=True
        Divide crossvalidated squared distances by number of features.
    min_rdm_pairs : int, default=3
        Minimum finite condition pairs needed for model-RDM correlation.
    **kwargs
        Ignored additional PANIC plugin arguments for compatibility.

    Returns
    -------
    score : float
        Selected model-RDM correlation, or mean neural dissimilarity if no
        model is supplied. NaN when insufficient information is available.
    (score, artifacts) : tuple, if return_artifacts=True
        Artifacts include ``rsa_conditions``, ``rsa_patterns`` (pooled condition
        means), ``rsa_neural_rdm``, ``rsa_neural_rdm_vec``, and model scores.
        For crossnobis they additionally include ``rsa_pair_counts``,
        ``rsa_trial_counts``, and ``rsa_run_patterns``.

    Notes
    -----
    * Crossnobis estimates can be negative; never truncate them to zero.
    * ``groups`` must represent independent acquisition partitions, not CV
      fold labels that reuse the same observations.
    * Per-run missing conditions produce NaN distances only for affected pairs.
      Conditions absent from *all* samples raise ValueError rather than silently
      misaligning model-RDM indices.
    * For crossnobis, ``noise_normalization='identity'`` is a useful initial
      pilot but is not covariance-whitened Mahalanobis distance.
    * Model fits are descriptive correlations. Their statistical comparison
      should be conducted across independent subjects.
    """
    started = perf_counter()
    # Artifact-producing observed calls are normally ROI analyses; keep
    # permutation and searchlight calls quiet at INFO level.
    log_level = logging.INFO if return_artifacts and not permute and cols is None else logging.DEBUG
    X = np.asarray(load_feature_matrix(X_path, cols=cols), dtype=float)
    y = np.asarray(labels).reshape(-1).copy()
    if X.ndim != 2 or X.shape[0] != len(y) or X.shape[1] == 0:
        raise ValueError("X must be 2D and aligned with labels.")
    if not np.isfinite(X).all():
        raise ValueError("X contains nonfinite values.")
    if compare not in {"spearman", "pearson"}:
        raise ValueError(f"Unknown RSA comparison method: {compare!r}")
    if min_rdm_pairs < 3:
        raise ValueError("min_rdm_pairs must be >= 3.")
    g = None if groups is None else np.asarray(groups).reshape(-1)
    if g is not None and len(g) != len(y):
        raise ValueError("groups must have one entry per sample.")
    if permute:
        rng = np.random.default_rng() if rng is None else rng
        if g is None:
            y = rng.permutation(y)
        else:
            for run in np.unique(g):
                idx = np.flatnonzero(g == run)
                y[idx] = rng.permutation(y[idx])

    conds = list(np.unique(y)) if conditions is None else list(conditions)
    if len(conds) < 3 or len(set(conds)) != len(conds):
        raise ValueError("RSA requires at least three distinct conditions.")
    absent = [c for c in conds if not np.any(y == c)]
    if absent:
        raise ValueError(f"Requested RSA conditions absent from data: {absent}")
    patterns = np.vstack([X[y == c].mean(axis=0) for c in conds])
    logger.log(log_level, "RSA input: X=%s metric=%s compare=%s runs=%s conditions=%s permute=%s",
               X.shape, metric, compare, None if g is None else np.unique(g).tolist(), conds, permute)

    pair_counts = trial_counts = run_patterns = None
    if metric == "crossnobis":
        if g is None:
            raise ValueError("metric='crossnobis' requires run-wise groups.")
        residuals = None
        if noise_residuals is not None and noise_normalization != "identity":
            residuals = _load_noise_residuals(noise_residuals, cols, X.shape[1])
        logger.log(log_level, "RSA crossnobis: normalization=%s | residuals=%s | min_partitions=%d | normalize_by_features=%s",
                   noise_normalization, None if residuals is None else residuals.shape,
                   min_partitions, normalize_by_features)
        neural_rdm, pair_counts, trial_counts, run_patterns = crossnobis_rdm(
            X, y, g, conds,
            noise_residuals=residuals,
            noise_normalization=noise_normalization,
            min_partitions=min_partitions,
            normalize_by_features=normalize_by_features,
        )
    else:
        if noise_normalization != "identity":
            raise ValueError("noise_normalization is only used for crossnobis.")
        neural_rdm = squareform(pdist(patterns, metric=metric))

    neural_vec = _rdm_vector(neural_rdm)
    if logger.isEnabledFor(log_level):
        logger.log(log_level, "RSA RDM: finite_pairs=%d/%d | negative_pairs=%d",
                   np.isfinite(neural_vec).sum(), neural_vec.size, (neural_vec < 0).sum())
        if trial_counts is not None:
            logger.log(log_level, "RSA partition QC: missing_run_condition_cells=%d/%d | ordered_pair_counts=%s",
                       (trial_counts == 0).sum(), trial_counts.size,
                       np.unique(_rdm_vector(pair_counts)).astype(int).tolist())
    model_scores: Dict[str, float] = {}
    model_mats: Dict[str, np.ndarray] = {}
    if not model_rdms:
        finite = neural_vec[np.isfinite(neural_vec)]
        score = float(finite.mean()) if len(finite) else float("nan")
    else:
        for model_name, model_spec in model_rdms.items():
            mat = (model_spec["matrix"] if isinstance(model_spec, Mapping)
                   and "matrix" in model_spec else model_spec)
            mat = _validate_rdm(mat, len(conds), name=model_name)
            model_mats[model_name] = mat
            model_scores[model_name] = _corr_vectors(
                neural_vec, _rdm_vector(mat), compare, min_pairs=min_rdm_pairs
            )
        if target_model is None:
            target_model = next(iter(model_scores))
        if target_model not in model_scores:
            raise ValueError(
                f"target_model={target_model!r} not in model_rdms: "
                f"{list(model_scores)}"
            )
        score = float(model_scores[target_model])

    logger.log(log_level, "RSA complete: target=%s | score=%.6g | models=%s | elapsed=%.3fs",
               target_model, score, model_scores, perf_counter() - started)
    if not return_artifacts:
        return score
    artifacts: ArtifactDict = {
        "rsa_conditions": np.asarray(conds),
        "rsa_patterns": patterns,
        "rsa_neural_rdm": neural_rdm,
        "rsa_neural_rdm_vec": neural_vec,
    }
    if pair_counts is not None:
        artifacts["rsa_pair_counts"] = pair_counts
        artifacts["rsa_trial_counts"] = trial_counts
        artifacts["rsa_run_patterns"] = run_patterns
    if model_scores:
        artifacts["rsa_model_names"] = np.asarray(list(model_scores), dtype=object)
        artifacts["rsa_model_scores"] = np.asarray(list(model_scores.values()), dtype=float)
        for name, mat in model_mats.items():
            artifacts[f"rsa_model_rdm_{name}"] = mat
    if not model_scores:
        return score, artifacts
    rows = [dict(model=str(name), observed=float(value), metric=metric,
                 compare=compare, noise_normalization=noise_normalization,
                 rsa_n_finite_pairs=int(np.isfinite(neural_vec).sum()),
                 rsa_n_negative_pairs=int((neural_vec < 0).sum()))
            for name, value in model_scores.items()]
    return PluginResult(score, artifacts, rows, ("model",), (str(target_model),))
