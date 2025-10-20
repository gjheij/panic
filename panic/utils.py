# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import sys
import json
import yaml
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Any, Dict
from joblib import dump, load
from importlib.resources import files, as_file
from sklearn.utils.validation import has_fit_parameter
from sklearn.model_selection._search import BaseSearchCV

import panic
from panic.logger import get_logger
from panic import factory

from sklearn.feature_selection import (
    VarianceThreshold
)
from sklearn.pipeline import Pipeline

logger = get_logger(__name__)
opj = os.path.join

def get_config_path(filename="config.yml"):
    with as_file(files(panic) / filename) as p:
        return p

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)



def recon_contribution(roi, lin_roi, contrib_path, output_file=None):
    """
    Reconstruct per-trial and mean contribution maps into a 4D NIfTI image.

    This function takes voxelwise contribution values (e.g., classifier weights ×
    activation values) from a decoding analysis and reprojects them into the
    3D space of an ROI or brain mask. It outputs a 4D image with one volume per
    trial and an additional final volume representing the mean contribution map
    across all trials.

    :param str | nibabel.Nifti1Image roi:
        Reference NIfTI image or file path defining spatial dimensions and affine.
        Typically the ROI mask or the beta image from which voxel indices were derived.
    :param str lin_roi:
        Path to a ``.npy`` file containing the **linear voxel indices** of the ROI.
        These indices must correspond exactly to the columns of the contribution array.
    :param str contrib_path:
        Path to a ``.npy`` file containing the per-trial contributions matrix with shape
        ``(n_trials, n_voxels_in_roi)``.
    :param str | None output_file:
        Optional file path to save the reconstructed 4D NIfTI image.  
        If ``None``, the function returns a :class:`nibabel.Nifti1Image` object instead.

    :returns:
        The 4D reconstructed contribution image (if ``output_file`` is ``None``),
        otherwise ``None`` after saving to disk.

        The output image has shape ``(X, Y, Z, n_trials + 1)``, where the final
        volume corresponds to the voxelwise average across all trials.
    :rtype:
        nibabel.Nifti1Image or None

    **Computation Details**
        - Loads the ROI reference to determine the target 3D spatial shape and affine.
        - Loads ROI voxel indices and the contribution matrix.
        - Reconstructs one 3D map per trial by inserting contribution values into the
          corresponding ROI voxels.
        - Computes the mean contribution map and appends it as the final volume.
        - Assembles all volumes into a 4D NIfTI image.
        - Optionally saves the result to disk if ``output_file`` is provided.

    **Example**
        .. code-block:: python

            img = recon_contribution(
                roi="roi_mask.nii.gz",
                lin_roi="roi_linidx.npy",
                contrib_path="sub-01_contrib.npy"
            )
            nib.save(img, "sub-01_contrib_4d.nii.gz")

        .. code-block:: python

            # Direct save mode
            recon_contribution(
                roi="roi_mask.nii.gz",
                lin_roi="roi_linidx.npy",
                contrib_path="sub-01_contrib.npy",
                output_file="sub-01_contrib_4d.nii.gz"
            )

    .. note::
       - The ROI linear indices must exactly match the voxel order used when
         generating the contributions.
       - The function ensures at least 2D input for single-trial contributions.
       - The final 4th dimension is ordered as ``[trial_1, trial_2, ..., mean]``.
       - The affine and header are copied from the ROI reference image.
       - Designed for visualizing spatial contribution patterns in fMRI decoding.
    """


    # --- load reference image ---
    if isinstance(roi, str):
        img = nib.load(roi)
    elif isinstance(roi, nib.Nifti1Image):
        img = roi
    else:
        raise TypeError("`roi` must be a file path or Nifti1Image object.")

    affine = img.affine
    roi_shape = img.shape[:3]

    # --- load data ---
    roi_linidx = np.load(lin_roi)
    contrib = np.load(contrib_path)  # (n_trials, n_voxels_in_roi)

    if contrib.ndim == 1:
        contrib = contrib[None, :]  # ensure 2D

    n_trials, n_vox = contrib.shape
    assert len(roi_linidx) == n_vox, \
        f"ROI voxels ({len(roi_linidx)}) != contribution size ({n_vox})"

    # --- reconstruct each 3D volume ---
    vols = []
    for i in range(n_trials):
        vol = np.zeros(np.prod(roi_shape), dtype=np.float32)
        vol[roi_linidx] = contrib[i]
        vols.append(vol.reshape(roi_shape))

    # --- append the mean volume ---
    mean_vol = np.zeros(np.prod(roi_shape), dtype=np.float32)
    mean_vol[roi_linidx] = contrib.mean(axis=0)
    vols.append(mean_vol.reshape(roi_shape))

    img4d = np.stack(vols, axis=-1)
    w_img = nib.Nifti1Image(img4d, affine)

    # --- save or return ---
    if output_file is not None:
        nib.save(w_img, output_file)
        print(f"[INFO] Saved contribution map: {output_file}")
        return None
    else:
        return w_img

def recon_weights(roi, lin_roi, weights_path, output_file=None):
    """
    Reconstruct a voxelwise weight map (e.g., SVM coefficients) into a 3D NIfTI.

    Parameters
    ----------
    roi : str or nib.Nifti1Image
        Reference NIfTI (ROI or beta image) defining spatial dimensions and affine.
    lin_roi : str
        Path to .npy file with linear voxel indices of the ROI (as used during decoding).
    weights_path : str
        Path to .npy file with model weights, shape = (n_voxels_in_roi,).
    output_file : str, optional
        Path to save the reconstructed NIfTI file.
        If None, returns the nibabel Nifti1Image object.

    Returns
    -------
    nib.Nifti1Image or None
        The 3D reconstructed weight map.
    """

    # --- Load reference image ---
    if isinstance(roi, str):
        img = nib.load(roi)
    elif isinstance(roi, nib.Nifti1Image):
        img = roi
    else:
        raise TypeError("`roi` must be a file path or Nifti1Image object.")

    affine = img.affine
    roi_shape = img.shape[:3]

    # --- Load data ---
    roi_linidx = np.load(lin_roi)
    weights = np.load(weights_path)

    if weights.ndim != 1:
        raise ValueError(f"Expected 1D weight vector, got shape {weights.shape}")

    assert len(roi_linidx) == weights.shape[0], \
        f"ROI voxels ({len(roi_linidx)}) != weights size ({weights.shape[0]})"

    # --- Fill into 3D volume ---
    full_map = np.zeros(np.prod(roi_shape), dtype=np.float32)
    full_map[roi_linidx] = weights
    w_img = nib.Nifti1Image(full_map.reshape(roi_shape), affine)

    # --- Save or return ---
    if output_file is not None:
        nib.save(w_img, output_file)
        print(f"[INFO] Saved weight map: {output_file}")
        return None
    else:
        return w_img

def _pipeline(cfg, *, standardize=False, searchlight=False, locked_params=None, **kwargs):
    """
    Build and configure a machine learning pipeline (optionally with grid search)
    for decoding or classification tasks, based on a configuration dictionary.

    This function constructs a scikit-learn :class:`~sklearn.pipeline.Pipeline`
    using preprocessing, feature selection, and classification steps defined
    in ``cfg``. It supports both ROI-based decoding (with hyperparameter
    optimization) and searchlight analyses (where parameters are fixed).

    :param dict cfg:
        Configuration dictionary specifying components of the pipeline.

        Expected keys include:

        * ``"scaler"`` – configuration for the feature scaler (optional)
        * ``"cv"`` – configuration for the inner cross-validation splitter
        * ``"estimator"`` – configuration for the final classifier
        * ``"feature_selection"`` – configuration for feature selection (optional)
        * ``"variance_threshold"`` – float; minimum variance to retain a feature
        * ``"gridsearch"`` – configuration for hyperparameter search
          (only used if not in searchlight mode and no ``locked_params`` are given)
    :param bool standardize:
        Whether to include feature standardization in the pipeline.
        Defaults to ``False``.
    :param bool searchlight:
        If ``True``, disables feature selection and hyperparameter search,
        returning a fixed estimator pipeline. Defaults to ``False``.
    :param dict locked_params:
        Dictionary of fixed estimator parameters to set directly on the pipeline.
        When provided, the hyperparameter grid is skipped.
        This is typically used for searchlight analyses to ensure consistent
        parameters across voxels.
    :param kwargs:
        Additional keyword arguments for compatibility or future extensions.
        Currently unused.

    :returns:
        Either a configured pipeline or a grid search object depending on the mode.

        * :class:`~sklearn.pipeline.Pipeline` – if ``locked_params`` is provided
          or ``searchlight=True``
        * :class:`~sklearn.model_selection.GridSearchCV` – otherwise
    :rtype:
        :class:`~sklearn.pipeline.Pipeline` or :class:`~sklearn.model_selection.GridSearchCV`

    .. note::
       - The pipeline always begins with a :class:`~sklearn.feature_selection.VarianceThreshold`
         step to remove near-constant features.
       - Feature selection is automatically disabled in searchlight mode.
       - When grid search is enabled, ``cfg["gridsearch"]`` must define
         parameter grids and search settings.

    **Example:**

    .. code-block:: python

        cfg = {
            "scaler": {"type": "StandardScaler"},
            "cv": {"type": "KFold", "n_splits": 5},
            "estimator": {"type": "SVC"},
            "gridsearch": {"param_grid": {"clf__C": [0.1, 1, 10]}},
        }

        # ROI-based decoding (uses grid search)
        pipe = _pipeline(cfg, standardize=True)
        print(pipe)
        # GridSearchCV(...)

        # Searchlight decoding (fixed parameters)
        sl_pipe = _pipeline(cfg, searchlight=True, locked_params={"clf__C": 1.0})
        print(sl_pipe)
        # Pipeline(...)
    """

    scaler   = factory.scaler_from_config(cfg.get("scaler"))
    inner_cv = factory.cv_from_config(cfg["cv"])
    est      = factory.estimator_from_config(cfg["estimator"])

    # Turn off feature selection for searchlight
    selector = factory.selector_from_config(
        cfg.get("feature_selection") if not searchlight else None,
        estimator_factory=factory.estimator_from_config,
        task="classification",
        random_state=0,
    )

    thr = float(cfg.get("variance_threshold", 1e-12))
    pipe = Pipeline([
        ("var", VarianceThreshold(threshold=thr)),
        ("scaler", scaler if standardize else "passthrough"),
        ("select", selector if selector is not None else "passthrough"),
        ("clf", est),
    ])

    # If we have locked params OR we're in searchlight, skip the grid entirely.
    # (You can still keep grid for ROI decoding.)
    if locked_params is not None or searchlight:
        if locked_params:
            pipe.set_params(**locked_params)
        return pipe

    # Otherwise (ROI path), use your regular inner grid search
    grid = factory.search_from_config(pipe, inner_cv, cfg["gridsearch"])
    return grid


def _permute_within_groups(y, g, rng):
    """
    Permute labels independently within each group.

    This function shuffles the elements of ``y`` **within** each unique group in ``g``.
    It is commonly used in permutation testing or cross-validation contexts where
    label shuffling must respect grouping constraints (e.g., runs, sessions, or subjects).

    If ``g`` is ``None``, a global permutation of ``y`` is performed instead.

    :param array_like y:
        Array of labels or target values to be permuted.
    :param array_like g:
        Array of group identifiers of the same length as ``y``.
        Each unique value in ``g`` defines an independent group within which
        permutation occurs. If ``None``, all samples are treated as belonging
        to one global group.
    :param numpy.random.Generator rng:
        Random number generator instance used to perform the permutations.
        Should be a :class:`numpy.random.Generator` object for reproducibility.

    :returns:
        A permuted copy of ``y`` where labels have been shuffled within each group.
    :rtype:
        numpy.ndarray

    **Example:**

    .. code-block:: python

        import numpy as np

        y = np.array([0, 1, 0, 1, 0, 1])
        g = np.array([1, 1, 2, 2, 3, 3])
        rng = np.random.default_rng(42)

        y_perm = _permute_within_groups(y, g, rng)
        print(y_perm)
        # Output might look like: [1 0 1 0 0 1]

    .. note::
       - The function preserves the group structure: elements belonging to different
         groups are never mixed.
       - Each call produces a different permutation unless the RNG seed is fixed.
    """

    if g is None:
        return rng.permutation(y)
    y_perm = y.copy()
    for grp in np.unique(g):
        idx = np.where(g == grp)[0]
        y_perm[idx] = rng.permutation(y_perm[idx])
    return y_perm

def _cv_mean_score(
    X_path,
    labels,
    folds,
    cfg,
    *,
    groups=None,
    standardize=True,
    permute=False,
    permute_both_sets=True,
    permute_within_groups=True,
    rng=None,
    save_dir=None,
    roi_linidx=None,
    **kwargs
):
    """
    Compute the mean cross-validated score across provided outer folds using the
    same pipeline definition as :func:`_pipeline`. Supports both observed scoring
    and permutation-based null estimation.

    The function loads a memory-mapped feature matrix from ``X_path`` (via
    :func:`joblib.load` with ``mmap_mode="r"``), iterates over outer folds, fits
    the configured model, and scores on held-out data. When ``permute=True``,
    labels are permuted either globally or within groups (runs/sessions) to form
    a null distribution sample.

    :param str X_path:
        Path to a ``joblib`` dump of a memory-mapped feature matrix with shape
        ``(n_samples, n_features)``.
    :param array_like labels:
        Integer (or categorical) labels of shape ``(n_samples,)``.
    :param list folds:
        List of 2-tuples ``(train_idx, test_idx)`` providing outer splits (e.g.,
        LOGO). Each index array selects rows of ``X`` and ``labels``.
    :param dict cfg:
        Decoding configuration passed to :func:`_pipeline`.
    :param array_like groups:
        Optional group identifiers of shape ``(n_samples,)``. If provided, they
        can be used to pass ``groups`` to estimators that support it, and to
        constrain permutations when ``permute_within_groups=True``.
    :param bool standardize:
        If ``True``, enables the scaler step in :func:`_pipeline`. Default: ``True``.
    :param bool permute:
        If ``True``, perform label permutations to estimate a null score. Default: ``False``.
    :param bool permute_both_sets:
        If ``True``, permute both training and test labels within each fold (current behavior).
        If ``False``, permute **only** training labels and score against true test labels.
        Default: ``True``.
    :param bool permute_within_groups:
        If ``True`` and ``groups`` is provided, permute labels independently
        **within** each group using :func:`_permute_within_groups`. Otherwise,
        perform a global permutation. Default: ``True``.
    :param numpy.random.Generator rng:
        RNG used for permutations. If ``None`` and ``permute=True``, a default
        generator is created via ``np.random.default_rng()``.
    :param str | pathlib.Path | None save_dir:
        Optional directory in which to store per-fold artifacts for the observed
        (non-permutation) run. When set, a subdirectory per fold is created with
        the pattern ``fold-XX``; the trained pipeline and relevant metadata are
        saved via :func:`_save_pipeline`.
    :param numpy.ndarray roi_linidx:
        Optional array of voxel linear indices (1D→3D mapping within an ROI mask)
        passed through to :func:`_save_pipeline`.
    :param kwargs:
        Additional keyword arguments forwarded to :func:`_pipeline` (e.g., estimator-specific
        options). These do not override the behavior controlled by the named parameters above.

    :returns:
        Mean of the per-fold scores (float).
    :rtype:
        float

    **Details**
        - The model is created via :func:`_pipeline(cfg, **kwargs)`.
        - If the resulting object is a :class:`~sklearn.model_selection.BaseSearchCV`
          instance **or** its estimator supports a ``groups`` parameter (detected via
          :func:`sklearn.utils.validation.has_fit_parameter`), groups are passed to
          ``fit`` when available.
        - For permutations:
          * If ``permute_within_groups`` and ``groups`` are set, labels are shuffled
            **within** each group in both train and test sets if ``permute_both_sets=True``;
            otherwise only the training labels are permuted.
          * If no groups are provided or ``permute_within_groups=False``, labels are
            permuted globally using ``rng.permutation``.
        - When ``permute=False`` and ``save_dir`` is provided, per-fold artifacts are
          saved after scoring.

    **Example**
        .. code-block:: python

            folds = [(tr, te) for tr, te in logo.split(X_idx, y, groups)]
            mean_acc = _cv_mean_score(
                X_path="/path/to/X.dump",
                labels=y,
                folds=folds,
                cfg=cfg,
                groups=groups,
                standardize=True,
                permute=False,
                save_dir="results/run-01",
                roi_linidx=roi_idx,
            )
            print(f"Mean CV score: {mean_acc:.3f}")

    .. seealso::
       :func:`_pipeline`, :func:`_permute_within_groups`, :func:`_save_pipeline`

    .. note::
       - ``X_path`` must reference a ``joblib``-dumped, memory-mappable array.
       - Set a fixed seed on ``rng`` (or create one with a fixed seed) for
         reproducible permutations.
       - The function returns only the **mean** score; if you need per-fold
         scores or fitted estimators for analysis, consider extending the
         saving logic or returning additional outputs.
    """

    X_mm = load(X_path, mmap_mode="r")
    y = np.asarray(labels)
    g_full = None if groups is None else np.asarray(groups)
    if rng is None and permute:
        rng = np.random.default_rng()

    fold_scores = []
    for f_ix, (train_idx, test_idx) in enumerate(folds):
        
        fold_dir = None
        if isinstance(save_dir, str):
            fold_dir = opj(save_dir, f"fold-{str(f_ix).zfill(len(str(len(folds))))}")
            os.makedirs(fold_dir, exist_ok=True)

        y_tr = y[train_idx]
        y_te = y[test_idx]
        g_tr = None if g_full is None else g_full[train_idx]
        g_te = None if g_full is None else g_full[test_idx]

        if permute:
            if permute_within_groups and g_full is not None:
                y_tr_perm = _permute_within_groups(y_tr, g_tr, rng)
                y_te_perm = _permute_within_groups(y_te, g_te, rng) if permute_both_sets else y_te
            else:
                y_tr_perm = rng.permutation(y_tr)
                y_te_perm = rng.permutation(y_te) if permute_both_sets else y_te
        else:
            y_tr_perm = y_tr
            y_te_perm = y_te

        # define the pipeline
        update_dict = {
            "groups": groups,
            "standardize": standardize
        }
        
        model = _pipeline(
            cfg,
            **kwargs
        )

        supports_groups = isinstance(model, BaseSearchCV) or has_fit_parameter(model, "groups")
        if g_tr is not None and supports_groups:
            model.fit(X_mm[train_idx], y_tr_perm, groups=g_tr)
        else:
            model.fit(X_mm[train_idx], y_tr_perm)

        score = model.score(X_mm[test_idx], y_te_perm)
        fold_scores.append(float(score))

        if not permute:
            if isinstance(fold_dir, str):
                _save_pipeline(
                    model,
                    X_mm,
                    test_idx,
                    train_idx,
                    fold_dir,
                    roi_linidx=roi_linidx
                )

    return float(np.mean(fold_scores))

def tqdm_disabled():
    return (not sys.stderr.isatty()) # or bool(os.environ.get("PYTEST_CURRENT_TEST"))

def extract_from_pipeline(model, X, test_idx):
    """
    Extract key elements (weights, predictions, and contributions) from a fitted
    decoding pipeline for a given test fold.

    This function reconstructs feature-space indices, classifier weights, and
    decision values from a trained scikit-learn pipeline. It supports both
    plain pipelines and those wrapped inside a :class:`~sklearn.model_selection.GridSearchCV`
    (by automatically selecting ``best_estimator_``). The function also computes
    per-sample feature contributions in the original ROI feature space.

    :param model:
        Trained model, typically an instance of :class:`~sklearn.pipeline.Pipeline`
        or :class:`~sklearn.model_selection.GridSearchCV` that wraps one.
    :param numpy.ndarray X:
        Feature matrix of shape ``(n_samples, n_features)`` used to fit the model.
        Should correspond to the same ROI feature space used during training.
    :param array_like test_idx:
        Indices of the test samples (from the outer CV split) for which to extract
        predictions and contributions.

    :returns:
        A dictionary containing extracted information from the trained model:

        * ``"test_idx"`` – indices of the test samples (echo of ``test_idx``)
        * ``"orig_idx"`` – feature indices retained after variance thresholding
          and feature selection, mapping back to the ROI feature space
        * ``"weights"`` – classifier weights expanded to full feature space
          (``numpy.ndarray`` of shape ``(n_features,)``)
        * ``"contribution"`` – per-sample feature contributions in ROI space
          (``numpy.ndarray`` of shape ``(n_test, n_features)``)
        * ``"decision"`` – raw decision values for the test samples
          (output of ``decision_function``)
        * ``"y_pred"`` – predicted labels for the test samples

    :rtype:
        dict

    **Computation Details**
        - If the model was trained using :class:`~sklearn.feature_selection.VarianceThreshold`
          or another selector, the function reconstructs the chain of indices mapping
          selected features back to their original ROI coordinates.
        - The full-weight vector is reconstructed by inserting zeros for any features
          that were filtered out during preprocessing.
        - Per-sample contributions are computed as the elementwise product between the
          transformed test data (up to, but excluding, the classifier step)
          and the classifier’s linear weights.

    **Example**
        .. code-block:: python

            ddict = extract_from_pipeline(trained_model, X, test_idx)

            print(ddict["weights"].shape)
            # (n_features,)

            print(ddict["decision"][:5])
            # [ 1.23, -0.85, 0.54, ...]

    .. note::
       - The classifier must expose a ``coef_`` attribute (e.g., linear models such as
         SVMs or logistic regression).
       - The function assumes the pipeline steps are named
         ``"var"``, ``"scaler"``, ``"select"``, and ``"clf"``.
       - If a step was set to ``'passthrough'``, its indices are handled transparently.
       - For non-linear classifiers (e.g., RBF SVM), the computed feature
         contributions are **not** meaningful.
    """

    pipe = model.best_estimator_ if hasattr(model, "best_estimator_") else model

    var = pipe.named_steps.get("var")             # VarianceThreshold or 'passthrough'
    sel = pipe.named_steps.get("select")          # selector
    clf = pipe.named_steps["clf"]

    # indices mapping back to ROI feature space
    var_idx = np.arange(X.shape[1]) if var == "passthrough" else var.get_support(indices=True)
    sel_idx = np.arange(len(var_idx)) if sel == "passthrough" else sel.get_support(indices=True)
    orig_idx = var_idx[sel_idx]                   # indices in ROI feature space (columns of X)

    # weights in full ROI feature space
    w_sel = clf.coef_.ravel()
    w_full = np.zeros(X.shape[1], dtype=float); w_full[orig_idx] = w_sel

    # out-of-fold predictions
    dec = pipe.decision_function(X[test_idx])
    y_pred = pipe.predict(X[test_idx])

    # (optional) per-sample contributions
    Xt_test = pipe[:-1].transform(X[test_idx])    # var -> scaler -> select
    contrib_full = np.zeros((Xt_test.shape[0], X.shape[1]), dtype=float)
    contrib_full[:, orig_idx] = Xt_test * w_sel

    ddict = {
        "test_idx": test_idx,
        "orig_idx": orig_idx,
        "weights": w_full,
        "contribution": contrib_full,
        "decision": dec,
        "y_pred": y_pred
    }

    return ddict

def _save_pipeline(model, X, test_idx, train_idx, output_dir, roi_linidx=None):
    """
    Save the fitted model, its parameters, and fold-specific outputs to disk
    in a consistent format for both ROI and searchlight decoding paths.

    This utility function serializes everything needed to reproduce or
    inspect a given cross-validation fold. It works seamlessly whether the
    model is a tuned :class:`~sklearn.model_selection.GridSearchCV` instance
    or a plain :class:`~sklearn.pipeline.Pipeline` (e.g., from searchlight
    decoding with fixed hyperparameters).

    The function:
        1. Extracts key outputs using :func:`extract_from_pipeline`
        2. Saves them as ``.npy`` arrays
        3. Dumps the fitted estimator using :func:`joblib.dump`
        4. Writes model parameters to a JSON file
        5. Optionally stores ROI voxel mapping if provided

    :param model:
        Fitted model to be saved. Can be either a
        :class:`~sklearn.model_selection.GridSearchCV` (ROI path)
        or a :class:`~sklearn.pipeline.Pipeline` (searchlight path).
    :param numpy.ndarray X:
        Full feature matrix of shape ``(n_samples, n_features)`` used in training.
        Needed for extracting model weights and predictions.
    :param array_like test_idx:
        Indices of the test samples used in this fold.
    :param array_like train_idx:
        Indices of the training samples used in this fold.
        (Not directly used in saving, but retained for completeness and possible
        downstream extensions.)
    :param str | pathlib.Path output_dir:
        Path to the output directory where all artifacts will be saved.
        The directory is created if it does not already exist.
    :param numpy.ndarray roi_linidx:
        Optional 1D array mapping ROI feature indices to 3D voxel coordinates.
        Saved as ``roi_linidx.npy`` if provided.

    :returns:
        None. Artifacts are written to disk at ``output_dir``.

    **Saved Files**
        - ``fold_k_test_idx.npy`` – test sample indices
        - ``fold_k_orig_idx.npy`` – indices of retained ROI features
        - ``fold_k_weights.npy`` – classifier weights (in full feature space)
        - ``fold_k_contribution.npy`` – per-sample feature contributions
        - ``fold_k_decision.npy`` – decision function values
        - ``fold_k_y_pred.npy`` – predicted labels
        - ``fold_k_best_estimator.joblib`` – serialized fitted pipeline or estimator
        - ``fold_k_best_params.json`` – parameter dictionary, with an extra key
          ``"_searchlight_mode"`` indicating whether the model came from
          searchlight or ROI decoding
        - ``roi_linidx.npy`` – optional voxel index mapping (if provided)

    **Behavior**
        - If ``model`` has a ``best_estimator_`` attribute, it is treated as a
          :class:`~sklearn.model_selection.GridSearchCV` and the tuned estimator
          is extracted via ``model.best_estimator_``.
        - Otherwise, the function assumes a simple pipeline (searchlight path)
          and saves it directly.
        - Parameters are taken from ``model.best_params_`` if available; otherwise,
          from ``fitted.named_steps["clf"].get_params()`` or fallback to
          ``fitted.get_params(deep=False)``.
        - The parameter JSON is augmented with the flag ``"_searchlight_mode"``.

    **Example**
        .. code-block:: python

            _save_pipeline(
                model=trained_model,
                X=X_mm,
                test_idx=test_idx,
                train_idx=train_idx,
                output_dir="results/roi_01/fold-00",
                roi_linidx=roi_idx
            )

    .. note::
       - Each call overwrites existing files with the same names in ``output_dir``.
       - The saved parameter JSON is human-readable and suitable for tracking
         model provenance.
       - The directory structure should be managed by the caller (e.g., one folder
         per outer CV fold).
    """

    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # If ROI/GridSearchCV: pull tuned estimator; else keep the fitted pipeline
    is_grid = hasattr(model, "best_estimator_")
    fitted = model.best_estimator_ if is_grid else model

    # ---- 1) Save fold outputs (whatever your extractor returns)
    ddict = extract_from_pipeline(fitted, X, test_idx)
    for key, val in ddict.items():
        np.save(opj(output_dir, f"fold_k_{key}.npy"), val)

    # ---- 2) Save the fitted estimator
    dump(fitted, opj(output_dir, "fold_k_best_estimator.joblib"))

    # ---- 3) Save params JSON (robust to both cases)
    if is_grid:
        best_params = model.best_params_
    else:
        # Prefer classifier-only params to keep file small; fall back to pipeline params
        try:
            best_params = fitted.named_steps["clf"].get_params()
        except Exception:
            best_params = fitted.get_params(deep=False)

    # Optional: tag so you can tell which path produced this
    out_params = {"_searchlight_mode": (not is_grid), **best_params}

    with open(opj(output_dir, "fold_k_best_params.json"), "w") as f:
        json.dump(out_params, f, indent=2)

    # ---- 4) Save ROI mapping if provided
    if isinstance(roi_linidx, np.ndarray):
        np.save(opj(output_dir, "roi_linidx.npy"), roi_linidx)


def _folds_for_labels(cfg, labels, groups=None):
    """
    Generate outer cross-validation folds for decoding, mirroring the ROI path
    behavior while supporting both group-based and legacy split schemes.

    Depending on the decoding configuration, this function either:
    
    - Uses a configured cross-validation splitter (e.g., StratifiedKFold, LeaveOneGroupOut)
      if ``permute_within_groups=True`` in ``cfg``; or
    - Falls back to a legacy deterministic scheme that selects every *k*-th
      trial within each class as the test set.

    :param dict cfg:
        Decoding settings dictionary. Expected keys include:
        
        * ``"permute_within_groups"`` – bool indicating whether to use group-aware CV.
        * ``"outer_cv"`` – configuration passed to :func:`factory.cv_from_config`
          (only used when ``permute_within_groups=True``).
        * ``"fold_interval"`` – integer specifying the interval for legacy
          “every-third” folding (used when ``permute_within_groups=False``).
    :param array_like labels:
        Label vector of shape ``(n_samples,)`` containing class assignments.
        Must contain binary or categorical values.
    :param array_like groups:
        Optional group labels of shape ``(n_samples,)`` used when
        group-aware cross-validation is enabled (e.g., runs or subjects).

    :returns:
        A list of tuples ``(train_idx, test_idx)``, where each element contains
        the integer indices for the training and test sets of one outer fold.
    :rtype:
        list[tuple[numpy.ndarray, numpy.ndarray]]

    **Behavior**
        - If ``permute_within_groups=True``:
          - Uses :func:`factory.cv_from_config(cfg["outer_cv"])`
            to instantiate a cross-validation splitter.
          - Generates folds by calling ``split(dummy_X, labels, groups)`` where
            ``dummy_X`` is a placeholder array (since most splitters ignore features).
        - If ``permute_within_groups=False``:
          - Implements the legacy “every *k*-th within-class” strategy:
            samples in each class are partitioned such that every *k*-th
            sample (as determined by ``fold_interval``) is assigned to the
            test set in one fold.
          - Training indices are the complement of test indices.

    **Example**
        .. code-block:: python

            cfg = {
                "permute_within_groups": True,
                "outer_cv": {"type": "LeaveOneGroupOut"}
            }
            folds = _folds_for_labels(cfg, labels, groups=runs)
            print(f"{len(folds)} outer folds generated")

        .. code-block:: python

            # Legacy "every third" fallback
            cfg = {"permute_within_groups": False, "fold_interval": 3}
            folds = _folds_for_labels(cfg, labels)
            print(len(folds))  # → 3

    .. note::
       - When using the legacy folding mode, the labels are assumed to be
         binary and balanced across classes.
       - The group-aware path provides compatibility with the ROI decoding
         configuration logic (via :func:`factory.cv_from_config`).
       - The “every-third” approach ensures that each sample appears once
         in the test set across all folds, similar to manual k-fold CV.
    """

    if cfg.get("permute_within_groups"):
        outer = factory.cv_from_config(cfg["outer_cv"])
        dummy_X = np.zeros_like(labels)  # not used by splitter
        folds = [(tr, te) for tr, te in outer.split(dummy_X, labels, groups)]
    else:
        # legacy “every third within class” (your define_folds)
        labs = np.asarray(labels)
        n_trials = len(labs)
        z_idx = np.where(labs == 0)[0]
        o_idx = np.where(labs == 1)[0]
        folds = []
        k = int(cfg["fold_interval"])
        for off in range(k):
            te = np.concatenate([z_idx[off::k], o_idx[off::k]])
            tr = np.setdiff1d(np.arange(n_trials), te)
            folds.append((tr, te))
    return folds