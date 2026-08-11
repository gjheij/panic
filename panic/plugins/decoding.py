# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from panic.pipeline import (
    sklearn_pipeline_score,
    load_feature_matrix,
    cv_mean_function_score
)


from .spatially_informed import (
    vanilla_nearest_centroid_score,
    n_region_nearest_centroid_score,
    fixed_six_region_generative_score,
)


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

    X = load_feature_matrix(X_path, cols=cols)
    score = cv_mean_function_score(
        X,
        labels,
        folds,
        sklearn_pipeline_score,
        groups=groups,
        permute=permute,
        rng=rng,
        cfg=cfg,
        **kwargs,
    )

    return (score, {}) if return_artifacts else score


def vanilla_nearest_centroid_plugin(
        X_path,
        labels,
        cfg,
        *,
        folds=None,
        groups=None,
        permute=False,
        rng=None,
        return_artifacts=False,
        **kwargs,
    ):
    """
    Perform cross-validated voxel-level nearest-centroid decoding.

    This plugin evaluates :func:`vanilla_nearest_centroid_score` independently
    on each supplied train/test fold and returns the mean balanced accuracy
    across folds.

    Parameters
    ----------
    X_path : array-like
        Trial-by-feature matrix with shape ``(n_samples, n_features)``.
    labels : array-like of shape (n_samples,)
        Binary class labels aligned to rows of ``X_path``.
    cfg : mapping
        Analysis configuration. Included for plugin-interface compatibility.
    folds : iterable of tuple(ndarray, ndarray)
        Cross-validation folds supplied as ``(train_idx, test_idx)`` pairs.
    cols : array-like of int, optional
        Included for plugin-interface compatibility. Currently unused.
    groups : array-like, optional
        Included for plugin-interface compatibility. Currently unused.
    permute : bool, default=False
        Included for plugin-interface compatibility. Label permutation is
        expected to be handled externally.
    rng : numpy.random.Generator, optional
        Included for plugin-interface compatibility. Currently unused.
    return_artifacts : bool, default=False
        If True, return ``(score, {})`` instead of only the scalar score.
    **kwargs
        Additional plugin arguments. Currently unused.

    Returns
    -------
    float or tuple
        Mean balanced accuracy across the supplied folds. If
        ``return_artifacts=True``, returns ``(score, {})``.

    See Also
    --------
    vanilla_nearest_centroid_score
        Scores a single train/test split.
    """
    if folds is None:
        raise ValueError("vanilla_nearest_centroid requires folds.")

    X = load_feature_matrix(X_path)

    score = cv_mean_function_score(
        X,
        labels,
        folds,
        vanilla_nearest_centroid_score,
        groups=groups,
        permute=permute,
        rng=rng,
        **kwargs,
    )

    return (score, {}) if return_artifacts else score


def n_region_nearest_centroid_plugin(
        X_path,
        labels,
        *,
        folds=None,
        cols=None,
        groups=None,
        permute=False,
        rng=None,
        return_artifacts=False,
        mask=None,
        **kwargs,
    ):
    """
    Perform cross-validated nearest-centroid decoding on spatial regions.

    The ROI voxels are reduced to contiguous spatial regions within each fold
    by :func:`n_region_nearest_centroid_score`. The resulting fold-wise
    balanced accuracies are averaged to produce the plugin score.

    Parameters
    ----------
    X_path : array-like
        Trial-by-voxel feature matrix with shape
        ``(n_samples, n_voxels)``.
    labels : array-like of shape (n_samples,)
        Binary class labels aligned to rows of ``X_path``.
    cfg : mapping
        Analysis configuration. Included for plugin-interface compatibility.
    folds : iterable of tuple(ndarray, ndarray)
        Cross-validation folds supplied as ``(train_idx, test_idx)`` pairs.
    cols : array-like of int, optional
        Feature subset used by searchlight analyses. This decoder does not
        currently support ``cols`` because ``mask`` must correspond exactly
        to the columns of ``X_path``.
    groups : array-like, optional
        Included for plugin-interface compatibility. Currently unused.
    permute : bool, default=False
        Included for plugin-interface compatibility. Label permutation is
        expected to be handled externally.
    rng : numpy.random.Generator, optional
        Included for plugin-interface compatibility. Currently unused.
    return_artifacts : bool, default=False
        If True, return ``(score, {})`` instead of only the scalar score.
    mask : str or path-like
        ROI NIfTI mask whose nonzero voxels correspond to the feature columns
        in ``X_path``.
    **kwargs
        Additional plugin arguments. Currently unused.

    Returns
    -------
    float or tuple
        Mean balanced accuracy across the supplied folds. If
        ``return_artifacts=True``, returns ``(score, {})``.

    Raises
    ------
    ValueError
        If ``mask`` or ``folds`` is not supplied, or if ``cols`` is used.

    See Also
    --------
    n_region_nearest_centroid_score
        Scores one train/test split after spatial region reduction.
    """

    if folds is None:
        raise ValueError("n_region_nearest_centroid_plugin requires folds.")

    if mask is None:
        raise ValueError(
            "n_region_nearest_centroid_plugin requires mask=<ROI mask>."
        )

    if cols is not None:
        raise ValueError(
            "n_region_nearest_centroid_plugin cannot currently be used with "
            "cols/searchlight because the mask must correspond exactly "
            "to the feature columns."
        )

    X = load_feature_matrix(X_path)

    score = cv_mean_function_score(
        X,
        labels,
        folds,
        n_region_nearest_centroid_score,
        groups=groups,
        permute=permute,
        rng=rng,
        mask_path=mask,
        **kwargs,
    )

    return (score, {}) if return_artifacts else score


def fixed_n_region_generative_plugin(
        X_path,
        labels,
        *,
        folds=None,
        cols=None,
        groups=None,
        permute=False,
        rng=None,
        return_artifacts=False,
        mask=None,
        trial_order_path=None,
        **kwargs,
    ):
    """
    Perform cross-validated spatially informed generative decoding.

    This plugin evaluates :func:`fixed_six_region_generative_score` on each
    supplied train/test fold and returns the mean balanced accuracy. The
    original labels are retained separately so the feature rows can remain
    aligned with the chronological trial-order metadata when fitting with
    permuted labels.

    Parameters
    ----------
    X_path : array-like
        Trial-by-voxel feature matrix with shape
        ``(n_samples, n_voxels)``.
    labels : array-like of shape (n_samples,)
        Binary labels used for model fitting and scoring.
    cfg : mapping
        Analysis configuration. Included for plugin-interface compatibility.
    folds : iterable of tuple(ndarray, ndarray)
        Cross-validation folds supplied as ``(train_idx, test_idx)`` pairs.
    cols : array-like of int, optional
        Feature subset used by searchlight analyses. This decoder does not
        currently support ``cols`` because ``mask`` must correspond exactly
        to the columns of ``X_path``.
    groups : array-like, optional
        Included for plugin-interface compatibility. Currently unused.
    permute : bool, default=False
        Included for plugin-interface compatibility. Permuted labels may be
        supplied through ``labels`` while their original ordering is retained
        for trial-order alignment.
    rng : numpy.random.Generator, optional
        Included for plugin-interface compatibility. Currently unused.
    return_artifacts : bool, default=False
        If True, return ``(score, {})`` instead of only the scalar score.
    mask : str or path-like
        ROI NIfTI mask whose nonzero voxels correspond to the feature columns
        in ``X_path``.
    trial_order_path : str or path-like
        CSV containing the chronological trial metadata required by the
        generative decoder.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`fixed_six_region_generative_score`.

    Returns
    -------
    float or tuple
        Mean balanced accuracy across the supplied folds. If
        ``return_artifacts=True``, returns ``(score, {})``.

    Raises
    ------
    ValueError
        If ``mask``, ``trial_order_path``, or ``folds`` is not supplied, or
        if ``cols`` is used.

    Notes
    -----
    ``original_labels`` is preserved separately from the fitting labels and
    passed as ``alignment_labels``. This allows permutation analyses to alter
    the labels used by the model without losing the original correspondence
    between feature rows and the chronological trial-order CSV.

    See Also
    --------
    fixed_six_region_generative_score
        Scores one train/test split with the spatial generative model.
    """

    if folds is None:
        raise ValueError("fixed_n_region_generative_plugin requires folds.")

    if mask is None:
        raise ValueError(
            "fixed_n_region_generative_plugin requires mask=<ROI mask>."
        )

    if cols is not None:
        raise ValueError(
            "fixed_n_region_generative_plugin cannot currently be used with "
            "cols/searchlight because the mask must correspond exactly "
            "to the feature columns."
        )

    # IMPORTANT: retain the original labels separately.
    original_labels = np.asarray(labels).ravel().copy()
    X = load_feature_matrix(X_path)

    score = cv_mean_function_score(
        X,
        labels,
        folds,
        fixed_six_region_generative_score,
        groups=groups,
        permute=permute,
        rng=rng,
        mask_path=mask,
        trial_order_path=trial_order_path,
        alignment_labels=original_labels,
        **kwargs,
    )

    return (score, {}) if return_artifacts else score
