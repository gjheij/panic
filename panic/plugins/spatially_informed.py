# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""
Single-fold fMRI decoding functions
===================================

These functions DO NOT perform cross-validation.

Each function receives ONE already-defined train/test split via:

    train_idx
    test_idx

and returns ONE balanced-accuracy score for that split.

This is intended for integration into an external loop such as:

    for train_idx, test_idx in folds:
        score = model_score(
            X,
            y,
            train_idx,
            test_idx,
            ...
        )

Public functions
----------------
1. vanilla_nearest_centroid_score
2. n_region_nearest_centroid_score
3. fixed_six_region_generative_score
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, eye as sparse_eye, lil_matrix
from scipy.sparse.linalg import spsolve
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from panic.utils import load_mask

ArrayLike = Union[np.ndarray, Any]
PathLike = Union[str, os.PathLike[str], Path]
MaskLike = Union[
    PathLike,
    nib.spatialimages.SpatialImage,
    np.ndarray,
]
ModelDict = Dict[str, Any]


def vanilla_nearest_centroid_score(
        X: ArrayLike,
        y: ArrayLike,
        train_idx: ArrayLike,
        test_idx: ArrayLike,
        class_0_label: int = 0,
        class_1_label: int = 1,
        standardize: bool = True,
        **kwargs: Any,
    ) -> float:
    """Score one train/test split with voxel-level nearest-centroid decoding.

The classifier is fit on the original voxel features. Features are standardized
using statistics estimated from the training split only, and balanced accuracy
is computed on the supplied test split. Cross-validation is intentionally handled
outside this function.

Parameters
----------
X : array-like of shape (n_trials, n_voxels)
    Trial-by-voxel feature matrix.
y : array-like of shape (n_trials,)
    Binary class labels.
train_idx, test_idx : array-like of int
    Row indices defining the training and test partitions for this fold.
class_0_label, class_1_label : int, defaults=0, 1
    Numeric values identifying the two classes.
standardize : bool, default=True
    Whether to z-score features using training-split statistics.
**kwargs
    Additional plugin/configuration arguments. They are accepted for interface
    compatibility and ignored by this scorer.

Returns
-------
float
    Balanced accuracy for the supplied test split.
"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel().astype(int)
    if class_0_label == class_1_label:
        raise ValueError('class_0_label and class_1_label must be different.')
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    centroid_0 = X_train[y_train == class_0_label].mean(axis=0)
    centroid_1 = X_train[y_train == class_1_label].mean(axis=0)
    distance_0 = np.sum((X_test - centroid_0[None, :]) ** 2, axis=1)
    distance_1 = np.sum((X_test - centroid_1[None, :]) ** 2, axis=1)
    predictions = np.where(distance_1 < distance_0, class_1_label, class_0_label)
    return float(balanced_accuracy_score(y_test, predictions))


def n_region_nearest_centroid_score(
        X: ArrayLike,
        y: ArrayLike,
        train_idx: ArrayLike,
        test_idx: ArrayLike,
        mask_path: MaskLike,
        n_regions: int = 6,
        linkage: str = "ward",
        class_0_label: int = 0,
        class_1_label: int = 1,
        standardize: bool = True,
        **kwargs: Any,
    ) -> float:
    """Score one split after reducing an ROI to contiguous spatial regions.

Nonzero voxels in ``mask_path`` are clustered in world-coordinate space using
spatially constrained agglomerative clustering. Each trial is represented by the
mean response within each region, followed by training-only standardization and
nearest-centroid classification.

Parameters
----------
X : array-like of shape (n_trials, n_voxels)
    Trial-by-voxel feature matrix.
y : array-like of shape (n_trials,)
    Binary class labels.
train_idx, test_idx : array-like of int
    Row indices defining the training and test partitions for this fold.
mask_path : str, nib.Nifti1Image-like, or path-like
    ROI NIfTI mask. Its number of nonzero voxels must equal ``X.shape[1]``.
n_regions : int, default=6
    Number of contiguous spatial regions to construct. May be supplied through
    ``analysis.args`` in the decoding configuration.
linkage : {"ward", "complete", "average", "single"}, default="ward"
    Agglomerative-clustering linkage criterion. May be supplied through
    ``analysis.args``.
class_0_label, class_1_label : int, defaults=0, 1
    Numeric values identifying the two classes.
standardize : bool, default=True
    Whether to z-score regional features using training-split statistics.
**kwargs
    Additional plugin/configuration arguments. They are accepted for interface
    compatibility and ignored by this scorer.

Returns
-------
float
    Balanced accuracy for the supplied test split.
"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel().astype(int)

    if class_0_label == class_1_label:
        raise ValueError('class_0_label and class_1_label must be different.')
    
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    n_regions = int(n_regions)
    if not 1 <= n_regions <= X.shape[1]:
        raise ValueError(
            f"n_regions must be between 1 and {X.shape[1]}, got {n_regions}."
        )
    
    mask_image, affine = load_mask(mask_path, return_affine=True)
    mask_boolean = np.asarray(mask_image) > 0

    n_mask_voxels = int(mask_boolean.sum())
    n_features = X.shape[1]

    if n_mask_voxels != n_features:
        raise ValueError(
            "Mask and feature matrix have incompatible shapes: "
            f"mask shape={mask_boolean.shape} contains {n_mask_voxels} "
            f"nonzero voxels, but X.shape={X.shape} contains {n_features} "
            "features. The number of nonzero mask voxels must equal X.shape[1]."
        )

    voxel_indices = np.argwhere(mask_boolean)
    world_coordinates = nib.affines.apply_affine(affine, voxel_indices)
    coordinate_to_feature = {
        tuple(coordinate): feature_index
        for feature_index, coordinate in enumerate(voxel_indices)
    }
    neighbour_offsets = np.asarray(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=int,
    )

    connectivity = lil_matrix((X.shape[1], X.shape[1]), dtype=np.int8)
    for feature_index, coordinate in enumerate(voxel_indices):
        for offset in neighbour_offsets:
            neighbour = tuple(coordinate + offset)
            neighbour_feature = coordinate_to_feature.get(neighbour)
            if neighbour_feature is not None:
                connectivity[feature_index, neighbour_feature] = 1

    clustering = AgglomerativeClustering(
        n_clusters=n_regions,
        linkage=linkage,
        connectivity=connectivity.tocsr(),
    )

    original_group_ids = clustering.fit_predict(world_coordinates)
    original_centres = np.asarray(
        [
            world_coordinates[original_group_ids == region_id].mean(axis=0)
            for region_id in range(n_regions)
        ]
    )

    spatial_order = np.lexsort(
        (original_centres[:, 0], original_centres[:, 2], original_centres[:, 1])
    )

    old_to_new = {
        int(old_region): int(new_region)
        for new_region, old_region in enumerate(spatial_order)
    }

    group_ids = np.asarray(
        [old_to_new[int(region_id)] for region_id in original_group_ids],
        dtype=int,
    )

    X_regional = np.column_stack(
        [
            X[:, group_ids == region_id].mean(axis=1)
            for region_id in range(n_regions)
        ]
    )

    X_train = X_regional[train_idx]
    y_train = y[train_idx]
    X_test = X_regional[test_idx]
    y_test = y[test_idx]
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    centroid_0 = X_train[y_train == class_0_label].mean(axis=0)
    centroid_1 = X_train[y_train == class_1_label].mean(axis=0)
    distance_0 = np.sum((X_test - centroid_0[None, :]) ** 2, axis=1)
    distance_1 = np.sum((X_test - centroid_1[None, :]) ** 2, axis=1)
    predictions = np.where(distance_1 < distance_0, class_1_label, class_0_label)
    return float(balanced_accuracy_score(y_test, predictions))


def fixed_six_region_generative_score(
        X: ArrayLike,
        y: ArrayLike,
        train_idx: ArrayLike,
        test_idx: ArrayLike,
        mask_path: MaskLike,
        trial_order_path: PathLike,
        alignment_labels: Optional[ArrayLike] = None,
        n_regions: int = 6,
        linkage: str = "ward",
        class_0_label: int = 0,
        class_1_label: int = 1,
        standardize: bool = True,
        drift_phi: float = 0.9,
        run_offset_ridge: float = 10.0,
        condition_signal_ridge: float = 1.0,
        drift_penalty: float = 10.0,
        min_noise_variance: float = 0.001,
        jitter: float = 1e-8,
        max_alternations: int = 30,
        n_random_initializations: int = 4,
        random_state: int = 42,
        run_seed_stride: int = 1000,
        **kwargs: Any,
    ) -> float:
    """Score one split with the spatially informed generative decoder.

The ROI is reduced to contiguous spatial regions and modeled as baseline +
run-specific offset + condition signal + AR(1)-regularized drift + residual noise.
The function operates on one externally supplied train/test split; it does not
perform cross-validation or hyperparameter selection internally.

``alignment_labels`` is intentionally separate from ``y`` so permutation tests can
shuffle the labels used for fitting while preserving the original row-to-trial
alignment against the chronological trial-order CSV.

Parameters
----------
X : array-like of shape (n_trials, n_voxels)
    Trial-by-voxel feature matrix.
y : array-like of shape (n_trials,)
    Binary labels used for model fitting and scoring.
train_idx, test_idx : array-like of int
    Row indices defining the training and test partitions for this fold.
mask_path : str, nib.Nifti1Image-like, or path-like
    ROI NIfTI mask whose nonzero voxel count must equal ``X.shape[1]``.
trial_order_path : str or path-like
    CSV containing at least ``run_id``, ``onset``, and ``label``.
alignment_labels : array-like, optional
    Original unpermuted labels used only to align ``X`` with the chronological CSV.
    Defaults to ``y`` for ordinary observed analyses.
n_regions : int, default=6
    Number of contiguous spatial regions.
linkage : {"ward", "complete", "average", "single"}, default="ward"
    Linkage criterion for spatial agglomerative clustering.
class_0_label, class_1_label : int, defaults=0, 1
    Numeric values identifying the two conditions/classes.
standardize : bool, default=True
    Whether to z-score regional features from the training split.
drift_phi : float, default=0.90
    AR(1) coefficient used in the drift penalty.
run_offset_ridge : float, default=10.0
    Ridge penalty applied to run-specific offsets.
condition_signal_ridge : float, default=1.0
    Ridge penalty applied to the condition-signal parameters.
drift_penalty : float, default=10.0
    Strength of the temporal drift penalty.
min_noise_variance : float, default=1e-3
    Lower bound for estimated regional residual variances.
jitter : float, default=1e-8
    Small diagonal regularizer used to stabilize linear solves.
max_alternations : int, default=30
    Maximum number of alternating nuisance/label updates per initialization.
n_random_initializations : int, default=4
    Number of additional random label initializations for test-run inference.
random_state : int, default=42
    Base random seed. Test runs receive deterministic offsets from this seed.
run_seed_stride : int, default=1000
    Deterministic seed offset applied between test runs.
**kwargs
    Additional plugin/configuration arguments. Accepted for interface compatibility
    and ignored unless consumed by future extensions.

Returns
-------
float
    Balanced accuracy for the supplied test split.
"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel().astype(int)
    if class_0_label == class_1_label:
        raise ValueError('class_0_label and class_1_label must be different.')
    if alignment_labels is None:
        alignment_labels = y.copy()
    else:
        alignment_labels = np.asarray(alignment_labels).ravel().astype(int)
    original_train_idx = np.asarray(train_idx, dtype=int)
    original_test_idx = np.asarray(test_idx, dtype=int)
    n_regions = int(n_regions)
    max_alternations = int(max_alternations)
    n_random_initializations = int(n_random_initializations)
    random_state = int(random_state)
    run_seed_stride = int(run_seed_stride)
    if not 1 <= n_regions <= X.shape[1]:
        raise ValueError(
            f"n_regions must be between 1 and {X.shape[1]}, got {n_regions}."
        )
    if not 0.0 <= drift_phi < 1.0:
        raise ValueError(
            f"drift_phi must satisfy 0 <= drift_phi < 1, got {drift_phi}."
        )
    if min(
        run_offset_ridge,
        condition_signal_ridge,
        drift_penalty,
        min_noise_variance,
        jitter,
    ) < 0:
        raise ValueError(
            "Regularization strengths, noise floor, and jitter must be non-negative."
        )
    if max_alternations < 1:
        raise ValueError('max_alternations must be at least 1.')
    if n_random_initializations < 0:
        raise ValueError('n_random_initializations cannot be negative.')
    if run_seed_stride < 0:
        raise ValueError('run_seed_stride cannot be negative.')
    trial_order = pd.read_csv(trial_order_path).reset_index(drop=True)
    required_columns = {'run_id', 'onset', 'label'}
    missing_columns = required_columns - set(trial_order.columns)
    if missing_columns:
        raise ValueError(
            f"Trial-order CSV is missing columns: {sorted(missing_columns)}"
        )

    mask_image, affine = load_mask(mask_path, return_affine=True)
    mask_boolean = np.asarray(mask_image) > 0

    if len(trial_order) != len(y):
        raise ValueError('trial_order CSV and X/y have different numbers of trials.')
    csv_labels = trial_order['label'].to_numpy(dtype=int)
    if np.array_equal(alignment_labels, csv_labels):
        train_idx = original_train_idx
        test_idx = original_test_idx
    else:
        source_counts = dict(zip(*np.unique(alignment_labels, return_counts=True)))
        csv_counts = dict(zip(*np.unique(csv_labels, return_counts=True)))
        if source_counts != csv_counts:
            raise ValueError('X/y are not in CSV order and class counts differ.')
        source_indices_by_class = {
            class_label: np.flatnonzero(alignment_labels == class_label)
            for class_label in sorted(source_counts)
        }
        next_position = {class_label: 0 for class_label in source_indices_by_class}
        source_rows = []
        for class_label in csv_labels:
            class_label = int(class_label)
            position = next_position[class_label]
            source_rows.append(int(source_indices_by_class[class_label][position]))
            next_position[class_label] += 1
        source_rows = np.asarray(source_rows, dtype=int)
        X = X[source_rows]
        y = y[source_rows]
        original_to_aligned = np.empty(len(y), dtype=int)
        original_to_aligned[source_rows] = np.arange(len(y))
        train_idx = original_to_aligned[original_train_idx]
        test_idx = original_to_aligned[original_test_idx]

        n_mask_voxels = int(mask_boolean.sum())
        n_features = X.shape[1]

        if n_mask_voxels != n_features:
            raise ValueError(
                "Mask and feature matrix have incompatible shapes: "
                f"mask shape={mask_boolean.shape} contains {n_mask_voxels} "
                f"nonzero voxels, but X.shape={X.shape} contains {n_features} "
                "features. The number of nonzero mask voxels must equal X.shape[1]."
            )

    voxel_indices = np.argwhere(mask_boolean)
    world_coordinates = nib.affines.apply_affine(affine, voxel_indices)
    coordinate_to_feature = {
        tuple(coordinate): feature_index
        for feature_index, coordinate in enumerate(voxel_indices)
    }
    neighbour_offsets = np.asarray(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=int,
    )
    connectivity = lil_matrix((X.shape[1], X.shape[1]), dtype=np.int8)
    for feature_index, coordinate in enumerate(voxel_indices):
        for offset in neighbour_offsets:
            neighbour = tuple(coordinate + offset)
            neighbour_feature = coordinate_to_feature.get(neighbour)
            if neighbour_feature is not None:
                connectivity[feature_index, neighbour_feature] = 1
    clustering = AgglomerativeClustering(
        n_clusters=n_regions,
        linkage=linkage,
        connectivity=connectivity.tocsr(),
    )
    original_group_ids = clustering.fit_predict(world_coordinates)
    original_centres = np.asarray(
        [
            world_coordinates[original_group_ids == region_id].mean(axis=0)
            for region_id in range(n_regions)
        ]
    )
    spatial_order = np.lexsort(
        (original_centres[:, 0], original_centres[:, 2], original_centres[:, 1])
    )
    old_to_new = {
        int(old_region): int(new_region)
        for new_region, old_region in enumerate(spatial_order)
    }
    group_ids = np.asarray(
        [old_to_new[int(region_id)] for region_id in original_group_ids],
        dtype=int,
    )
    X_regional = np.column_stack(
        [
            X[:, group_ids == region_id].mean(axis=1)
            for region_id in range(n_regions)
        ]
    )

    def condition_codes(labels: ArrayLike) -> np.ndarray:
        """Convert binary labels {0, 1} to symmetric condition codes {-1.0, +1.0}."""
        labels = np.asarray(labels)
        if np.any(~np.isin(labels, [class_0_label, class_1_label])):
            raise ValueError(
                "Labels contain values outside class_0_label/class_1_label."
            )
        return np.where(labels == class_1_label, 1.0, -1.0)

    def parameter_layout(
            number_trials: int,
            training_run_ids: ArrayLike,
        ) -> ModelDict:
        """Build slices and run mappings for the flattened model parameter vector."""
        unique_runs = np.asarray(sorted(np.unique(training_run_ids)))
        number_runs = len(unique_runs)
        baseline_start = 0
        baseline_stop = baseline_start + n_regions
        run_start = baseline_stop
        run_stop = run_start + number_runs * n_regions
        signal_start = run_stop
        signal_stop = signal_start + n_regions
        drift_start = signal_stop
        drift_stop = drift_start + number_trials * n_regions
        return {
            "unique_runs": unique_runs,
            "run_to_position": {
                run_id: position
                for position, run_id in enumerate(unique_runs)
            },
            "baseline_slice": slice(baseline_start, baseline_stop),
            "run_slice": slice(run_start, run_stop),
            "signal_slice": slice(signal_start, signal_stop),
            "drift_slice": slice(drift_start, drift_stop),
            "number_parameters": drift_stop,
        }

    def build_training_system(
            labels: ArrayLike,
            metadata: pd.DataFrame,
        ) -> Tuple[csc_matrix, csc_matrix, ModelDict, np.ndarray]:
        """Construct the sparse training design matrix and quadratic penalty matrix."""
        metadata = metadata.reset_index(drop=True)
        codes = condition_codes(labels)
        training_run_ids = metadata['run_id'].to_numpy()
        layout = parameter_layout(
            number_trials=len(labels),
            training_run_ids=training_run_ids,
        )
        rows = []
        columns = []
        values = []
        for trial_index in range(len(labels)):
            run_position = layout['run_to_position'][training_run_ids[trial_index]]
            for region_index in range(n_regions):
                observation_row = trial_index * n_regions + region_index
                rows.append(observation_row)
                columns.append(layout['baseline_slice'].start + region_index)
                values.append(1.0)
                rows.append(observation_row)
                columns.append(
                    layout["run_slice"].start
                    + run_position * n_regions
                    + region_index
                )
                values.append(1.0)
                rows.append(observation_row)
                columns.append(layout['signal_slice'].start + region_index)
                values.append(codes[trial_index])
                rows.append(observation_row)
                columns.append(
                    layout["drift_slice"].start
                    + trial_index * n_regions
                    + region_index
                )
                values.append(1.0)
        design = csc_matrix(
            (values, (rows, columns)),
            shape=(len(labels) * n_regions, layout["number_parameters"]),
            dtype=np.float64,
        )
        penalty = lil_matrix(
            (layout["number_parameters"], layout["number_parameters"]),
            dtype=np.float64,
        )
        for parameter_index in range(
            layout["baseline_slice"].start,
            layout["baseline_slice"].stop,
        ):
            penalty[parameter_index, parameter_index] += jitter
        for parameter_index in range(
            layout["run_slice"].start,
            layout["run_slice"].stop,
        ):
            penalty[parameter_index, parameter_index] += run_offset_ridge
        for parameter_index in range(
            layout["signal_slice"].start,
            layout["signal_slice"].stop,
        ):
            penalty[parameter_index, parameter_index] += condition_signal_ridge
        drift_start = layout['drift_slice'].start
        for _, run_table in metadata.groupby('run_id', sort=False):
            ordered_indices = run_table.sort_values('onset').index.to_numpy()
            first_trial = int(ordered_indices[0])
            for region_index in range(n_regions):
                first_parameter = drift_start + first_trial * n_regions + region_index
                penalty[first_parameter, first_parameter] += drift_penalty
            for position in range(1, len(ordered_indices)):
                previous_trial = int(ordered_indices[position - 1])
                current_trial = int(ordered_indices[position])
                for region_index in range(n_regions):
                    previous_parameter = (
                        drift_start + previous_trial * n_regions + region_index
                    )
                    current_parameter = (
                        drift_start + current_trial * n_regions + region_index
                    )
                    penalty[current_parameter, current_parameter] += drift_penalty
                    penalty[previous_parameter, previous_parameter] += (
                        drift_penalty * drift_phi ** 2
                    )
                    cross_value = -drift_penalty * drift_phi
                    penalty[current_parameter, previous_parameter] += cross_value
                    penalty[previous_parameter, current_parameter] += cross_value
        return (design, penalty.tocsc(), layout, training_run_ids)

    def solve_training(
            regional_data: np.ndarray,
            labels: ArrayLike,
            metadata: pd.DataFrame,
            noise_variance: Optional[np.ndarray] = None,
        ) -> ModelDict:
        """Solve the penalized weighted least-squares training system."""
        design, penalty, layout, training_run_ids = build_training_system(
            labels=labels,
            metadata=metadata,
        )
        observations = regional_data.reshape(-1)
        if noise_variance is None:
            row_scale = np.ones_like(observations)
        else:
            noise_variance = np.maximum(
                np.asarray(noise_variance, dtype=np.float64),
                min_noise_variance,
            )
            row_scale = np.tile(1.0 / np.sqrt(noise_variance), len(regional_data))
        weighted_design = design.multiply(row_scale[:, None])
        weighted_observations = observations * row_scale
        normal_matrix = (
            weighted_design.T @ weighted_design
            + penalty
            + jitter * sparse_eye(layout["number_parameters"], format="csc")
        )
        rhs = weighted_design.T @ weighted_observations
        parameters = spsolve(normal_matrix, rhs)
        baseline = parameters[layout['baseline_slice']]
        run_values = parameters[layout["run_slice"]].reshape(
            len(layout["unique_runs"]),
            n_regions,
        )
        run_offsets = {
            run_id: run_values[position]
            for position, run_id in enumerate(layout["unique_runs"])
        }
        condition_signal = parameters[layout['signal_slice']]
        drift = parameters[layout['drift_slice']].reshape(len(regional_data), n_regions)
        run_offset_matrix = np.vstack(
            [run_offsets[run_id] for run_id in training_run_ids]
        )
        condition_component = (
            condition_codes(labels)[:, None] * condition_signal[None, :]
        )
        fitted = baseline[None, :] + run_offset_matrix + condition_component + drift
        residual = regional_data - fitted
        return {
            "baseline": baseline,
            "condition_signal": condition_signal,
            "drift": drift,
            "residual": residual,
        }

    def fit_model(
            regional_train: np.ndarray,
            y_train: np.ndarray,
            train_metadata: pd.DataFrame,
        ) -> ModelDict:
        """Fit scaling, signal, nuisance terms, and residual noise estimates."""
        scaler = StandardScaler() if standardize else None
        standardized_train = (
            scaler.fit_transform(regional_train)
            if scaler is not None
            else np.asarray(regional_train, dtype=np.float64)
        )
        first_fit = solve_training(
            regional_data=standardized_train,
            labels=y_train,
            metadata=train_metadata,
            noise_variance=None,
        )
        first_noise_variance = np.maximum(
            np.var(first_fit["residual"], axis=0, ddof=1),
            min_noise_variance,
        )
        final_fit = solve_training(
            regional_data=standardized_train,
            labels=y_train,
            metadata=train_metadata,
            noise_variance=first_noise_variance,
        )
        final_fit["noise_variance"] = np.maximum(
            np.var(final_fit["residual"], axis=0, ddof=1),
            min_noise_variance,
        )
        final_fit['scaler'] = scaler
        final_fit["centroid_0"] = standardized_train[
            y_train == class_0_label
        ].mean(axis=0)
        final_fit["centroid_1"] = standardized_train[
            y_train == class_1_label
        ].mean(axis=0)
        return final_fit

    def test_nuisance_penalty(number_trials: int) -> np.ndarray:
        """Construct the run-offset and AR(1) drift penalty for one test run."""
        number_parameters = 1 + number_trials
        penalty = np.zeros((number_parameters, number_parameters), dtype=np.float64)
        penalty[0, 0] += run_offset_ridge
        penalty[1, 1] += drift_penalty
        for trial_index in range(1, number_trials):
            previous_parameter = trial_index
            current_parameter = 1 + trial_index
            penalty[current_parameter, current_parameter] += drift_penalty
            penalty[previous_parameter, previous_parameter] += (
                drift_penalty * drift_phi ** 2
            )
            cross_value = -drift_penalty * drift_phi
            penalty[current_parameter, previous_parameter] += cross_value
            penalty[previous_parameter, current_parameter] += cross_value
        return penalty

    def solve_test_nuisance(
            standardized_test_run: np.ndarray,
            provisional_labels: np.ndarray,
            model: ModelDict,
        ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Estimate test-run offset and drift for provisional condition labels."""
        condition_component = (
            condition_codes(provisional_labels)[:, None]
            * model["condition_signal"][None, :]
        )
        target = (
            standardized_test_run
            - model["baseline"][None, :]
            - condition_component
        )
        number_trials = len(standardized_test_run)
        design = np.zeros((number_trials, number_trials + 1), dtype=np.float64)
        design[:, 0] = 1.0
        design[np.arange(number_trials), 1 + np.arange(number_trials)] = 1.0
        penalty = test_nuisance_penalty(number_trials)
        run_offset = np.zeros(n_regions, dtype=np.float64)
        drift = np.zeros((number_trials, n_regions), dtype=np.float64)
        total_objective = 0.0
        for region_index in range(n_regions):
            variance = max(model['noise_variance'][region_index], min_noise_variance)
            precision = 1.0 / variance
            normal_matrix = (
                precision * (design.T @ design)
                + penalty
                + jitter * np.eye(number_trials + 1)
            )
            rhs = precision * design.T @ target[:, region_index]
            parameters = np.linalg.solve(normal_matrix, rhs)
            run_offset[region_index] = parameters[0]
            drift[:, region_index] = parameters[1:]
            residual = target[:, region_index] - design @ parameters
            total_objective += (
                precision * np.sum(residual ** 2)
                + parameters.T @ penalty @ parameters
            )
        return (run_offset, drift, float(total_objective))

    def classify_given_nuisance(
            standardized_test_run: np.ndarray,
            run_offset: np.ndarray,
            drift: np.ndarray,
            model: ModelDict,
        ) -> np.ndarray:
        """Classify trials after removing fitted baseline and nuisance components."""
        cleaned = (
            standardized_test_run
            - model["baseline"][None, :]
            - run_offset[None, :]
            - drift
        )
        signal = model['condition_signal']
        variance = model['noise_variance']
        cost_0 = np.sum((cleaned + signal[None, :]) ** 2 / variance[None, :], axis=1)
        cost_1 = np.sum((cleaned - signal[None, :]) ** 2 / variance[None, :], axis=1)
        return np.where(cost_1 < cost_0, class_1_label, class_0_label)

    def classify_test_run(
            regional_test_run: np.ndarray,
            test_run_metadata: pd.DataFrame,
            model: ModelDict,
            random_seed: int,
        ) -> np.ndarray:
        """Infer labels for one ordered test run from multiple initializations."""
        order = np.argsort(test_run_metadata['onset'].to_numpy(dtype=np.float64))
        inverse_order = np.argsort(order)
        sorted_raw = regional_test_run[order]
        standardized_test = (
            model["scaler"].transform(sorted_raw)
            if model["scaler"] is not None
            else np.asarray(sorted_raw, dtype=np.float64)
        )
        rng = np.random.default_rng(random_seed)
        distance_0 = np.sum(
            (standardized_test - model["centroid_0"][None, :]) ** 2,
            axis=1,
        )
        distance_1 = np.sum(
            (standardized_test - model["centroid_1"][None, :]) ** 2,
            axis=1,
        )
        centroid_init = np.where(distance_1 < distance_0, class_1_label, class_0_label)
        mean_0 = model['baseline'] - model['condition_signal']
        mean_1 = model['baseline'] + model['condition_signal']
        variance = model['noise_variance']
        cost_0 = np.sum(
            (standardized_test - mean_0[None, :]) ** 2 / variance[None, :],
            axis=1,
        )
        cost_1 = np.sum(
            (standardized_test - mean_1[None, :]) ** 2 / variance[None, :],
            axis=1,
        )
        condition_init = np.where(cost_1 < cost_0, class_1_label, class_0_label)
        initializations = [
            centroid_init,
            condition_init,
            np.full(len(standardized_test), class_0_label, dtype=int),
            np.full(len(standardized_test), class_1_label, dtype=int),
        ]
        for _ in range(n_random_initializations):
            initializations.append(
                rng.choice(
                    np.asarray([class_0_label, class_1_label], dtype=int),
                    size=len(standardized_test),
                )
            )
        unique_initializations = []
        seen_initializations = set()
        for labels_initial in initializations:
            key = tuple(labels_initial.tolist())
            if key not in seen_initializations:
                seen_initializations.add(key)
                unique_initializations.append(labels_initial)
        candidate_solutions = []
        for labels_initial in unique_initializations:
            labels_current = labels_initial.copy()
            seen_sequences = set()
            for _ in range(max_alternations):
                sequence_key = tuple(labels_current.tolist())
                if sequence_key in seen_sequences:
                    break
                seen_sequences.add(sequence_key)
                run_offset, drift, objective = solve_test_nuisance(
                    standardized_test_run=standardized_test,
                    provisional_labels=labels_current,
                    model=model,
                )
                labels_new = classify_given_nuisance(
                    standardized_test_run=standardized_test,
                    run_offset=run_offset,
                    drift=drift,
                    model=model,
                )
                if np.array_equal(labels_new, labels_current):
                    labels_current = labels_new
                    break
                labels_current = labels_new
            final_run_offset, final_drift, final_objective = solve_test_nuisance(
                standardized_test_run=standardized_test,
                provisional_labels=labels_current,
                model=model,
            )
            final_labels = classify_given_nuisance(
                standardized_test_run=standardized_test,
                run_offset=final_run_offset,
                drift=final_drift,
                model=model,
            )
            candidate_solutions.append((float(final_objective), final_labels))
        best_labels = min(candidate_solutions, key=lambda item: item[0])[1]
        return best_labels[inverse_order]

    train_metadata = trial_order.iloc[train_idx].reset_index(drop=True)
    model = fit_model(
        regional_train=X_regional[train_idx],
        y_train=y[train_idx],
        train_metadata=train_metadata,
    )
    y_test = y[test_idx]
    predictions = np.empty(len(test_idx), dtype=int)
    test_run_ids = trial_order.iloc[test_idx]['run_id'].to_numpy()
    for run_number, run_id in enumerate(np.unique(test_run_ids)):
        local_positions = np.flatnonzero(test_run_ids == run_id)
        global_indices = test_idx[local_positions]
        test_run_metadata = trial_order.iloc[global_indices].reset_index(drop=True)
        run_predictions = classify_test_run(
            regional_test_run=X_regional[global_indices],
            test_run_metadata=test_run_metadata,
            model=model,
            random_seed=random_state + run_seed_stride * run_number,
        )
        predictions[local_positions] = run_predictions
    return float(balanced_accuracy_score(y_test, predictions))
