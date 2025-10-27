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


def pipeline_from_config(
    cfg,
    *,
    searchlight: bool = False,
    standardize: bool = False,
    random_state=None,
    labels=None,
    scoring=None,
    locked=None,
):
    """
    Construct a decoding pipeline from configuration.

    This function builds a full scikit-learn :class:`~sklearn.pipeline.Pipeline`
    according to the specification in ``cfg``. The standard structure is:

        ``VarianceThreshold`` → (optional) ``Scaler`` → (optional) ``Feature Selector`` → ``Estimator``

    Optionally, the resulting pipeline can be wrapped in a model selection object
    (e.g., :class:`~sklearn.model_selection.GridSearchCV` or
    :class:`~sklearn.model_selection.RandomizedSearchCV`) for ROI-level optimization,
    or configured in a “locked” mode for searchlight decoding, where grid search is skipped
    and previously tuned parameters are applied directly.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary defining all pipeline components. Supported keys include:
        ``"variance_threshold"``, ``"scaler"``, ``"feature_selection"``,
        ``"estimator"``, ``"cv"``, and optionally ``"gridsearch"``.
    searchlight : bool, default=False
        If True, disables grid/random search and applies ``locked`` parameters directly.
        Used for voxelwise searchlight decoding where hyperparameter optimization is impractical.
    standardize : bool, default=False
        Whether to include a scaling step defined by ``cfg["scaler"]``.
        If False, the scaler stage is replaced by ``"passthrough"``.
    random_state : int or None, optional
        Seed controlling stochastic elements in estimators, cross-validation splits,
        or randomized searches. Typically derived from a higher-level RNG in
        permutation testing.
    labels : array-like of shape (n_samples,), optional
        Target labels used for label-aware estimator initialization or scoring behavior.
        When provided, they can inform:
        - The choice of appropriate scoring function (binary vs. multiclass).
        - Whether to enable probabilistic outputs (e.g., ``SVC(probability=True)``)
          when using AUC- or log-loss-based metrics.
        - Automatic adaptation of class balancing or weighting schemes
          (e.g., ``class_weight='balanced'``).
        Passing labels is recommended when the estimator or metric depends on
        the number of unique classes.
    scoring : str or callable, optional
        Metric used for model evaluation and hyperparameter tuning.
        If not provided, falls back to ``cfg["scoring"]`` or ``"balanced_accuracy"``.
        The same scoring function is propagated consistently across observed and
        permutation-based evaluations.
    locked : dict or None, optional
        Parameter dictionary applied directly via ``Pipeline.set_params``.
        Used in searchlight mode to reuse tuned hyperparameters from ROI-level analyses.

    Returns
    -------
    pipe : sklearn.pipeline.Pipeline or sklearn.model_selection.BaseSearchCV
        - **ROI mode** → A grid or randomized search object wrapping the pipeline.
        - **Searchlight mode** → A bare pipeline with fixed parameters (no search).

    Notes
    -----
    - The base pipeline always includes a
      :class:`~sklearn.feature_selection.VarianceThreshold` filter controlled by
      ``cfg["variance_threshold"]`` (default: 1e-12) to remove constant features.
    - If ``cfg["gridsearch"]`` is present and ``searchlight=False``, the pipeline
      is wrapped using :func:`factory.search_from_config`. The scoring metric is
      inherited from ``cfg["scoring"]`` unless explicitly overridden in the
      grid search arguments.
    - When ``searchlight=True`` or ``locked`` is provided, no hyperparameter search
      is performed and the locked parameters are applied directly.
    - A deterministic ``random_state`` ensures reproducibility across permutations
      and folds when passed down from higher-level RNGs.

    Examples
    --------
    ROI decoding with grid search:

    >>> pipe = pipeline_from_config(cfg, standardize=True, random_state=42)
    >>> pipe
    GridSearchCV(
        estimator=Pipeline(steps=[
            ('var', VarianceThreshold(threshold=1e-12)),
            ('scaler', StandardScaler()),
            ('select', SelectPercentile(percentile=10)),
            ('clf', SVC(class_weight='balanced', kernel='linear'))
        ]),
        param_grid={'select__percentile': [10, 20, 40, 100],
                    'clf__C': [0.01, 0.1, 1, 10]},
        scoring='balanced_accuracy'
    )

    Searchlight decoding with locked parameters:

    >>> locked = {'clf__C': 1.0, 'select__percentile': 20}
    >>> pipe = pipeline_from_config(cfg, searchlight=True, locked=locked)
    >>> type(pipe)
    <class 'sklearn.pipeline.Pipeline'>

    Binary AUC decoding example (label-aware scoring):

    >>> cfg['scoring'] = 'roc_auc_ovr'
    >>> pipe = pipeline_from_config(cfg, labels=y, random_state=42)
    >>> # Under the hood, SVC(probability=True) is enabled automatically.

    See Also
    --------
    factory.scaler_from_config : Construct scaling components (e.g., StandardScaler, MinMaxScaler).
    factory.selector_from_config : Build feature selection objects.
    factory.estimator_from_config : Create estimator objects (SVC, LDA, LogisticRegression, etc.).
    factory.search_from_config : Wrap pipelines in GridSearchCV or RandomizedSearchCV.
    """

    # 1) Scaler (always define it)
    scaler = factory.scaler_from_config(cfg.get("scaler")) if standardize else "passthrough"

    # 2) Feature selector
    selector = factory.selector_from_config(
        cfg.get("feature_selection") if not searchlight else None,
        estimator_factory=factory.estimator_from_config,
        random_state=random_state,
    )

    # 3) Estimator (labels/scoring-aware if you added those conveniences)
    est = factory.estimator_from_config(
        cfg.get("estimator"),
        random_state=random_state,
        labels=labels,
        scoring=scoring if scoring is not None else cfg.get("scoring", "balanced_accuracy"),
    )

    # 4) Assemble base pipeline
    thr = float(cfg.get("variance_threshold", 1e-12))
    pipe = Pipeline([
        ("var", VarianceThreshold(threshold=thr)),
        ("scaler", scaler if standardize else "passthrough"),
        ("select", selector if selector is not None else "passthrough"),
        ("clf", est),
    ])


    # 5) Apply locked params / searchlight early-exit
    if searchlight or locked:
        if locked:
            pipe.set_params(**locked)
        return pipe

    # 6) Optional grid/random search (ROI path only)
    gs_cfg = cfg.get("gridsearch")
    if gs_cfg:
        # Inner CV from cfg
        inner_cv = factory.cv_from_config(cfg.get("cv"))

        # If grid args don't specify scoring but top-level does, propagate it
        if "args" in gs_cfg and "scoring" not in gs_cfg["args"] and "scoring" in cfg:
            gs_cfg = {**gs_cfg, "args": {**gs_cfg["args"], "scoring": cfg["scoring"]}}

        pipe = factory.search_from_config(
            estimator=pipe,
            cv=inner_cv,
            cfg=gs_cfg,
        )
    
    return pipe


def _permute_within_groups(y, g, rng):
    """
    Permute labels independently within groups.

    This utility function shuffles the labels in ``y`` **within each unique group**
    defined by the corresponding entries in ``g``. It is typically used in
    permutation testing or cross-validation contexts where label shuffling must
    preserve the dependency structure among samples (e.g., within runs, sessions,
    or subjects).

    If ``g`` is ``None``, a global permutation of ``y`` is performed instead,
    equivalent to ``rng.permutation(y)``.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target labels or values to permute.
    g : array-like of shape (n_samples,), optional
        Group identifiers that define independent shuffling blocks. All samples
        sharing the same group label are permuted among themselves but never
        mixed with samples from other groups. If ``None``, all samples are
        treated as belonging to a single group.
    rng : numpy.random.Generator
        Random number generator instance used for permutation. Should be a
        :class:`numpy.random.Generator` created via :func:`numpy.random.default_rng`
        for reproducible permutations.

    Returns
    -------
    y_perm : numpy.ndarray of shape (n_samples,)
        A permuted copy of ``y`` where labels have been shuffled within each group.

    Notes
    -----
    - The group structure is strictly preserved: no label exchanges occur between
      different groups.
    - The output array is a **copy** of ``y``; the input is not modified in place.
    - Each call produces a different permutation unless the RNG is seeded with
      a fixed value.
    - The function is agnostic to label type; any 1D array-like input
      (integers, strings, floats) is supported.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 1, 0, 1, 0, 1])
    >>> g = np.array([1, 1, 2, 2, 3, 3])
    >>> rng = np.random.default_rng(42)
    >>> _permute_within_groups(y, g, rng)
    array([1, 0, 1, 0, 0, 1])

    When ``g`` is None, labels are permuted globally:

    >>> rng = np.random.default_rng(123)
    >>> _permute_within_groups(y, None, rng)
    array([0, 0, 1, 1, 0, 1])

    See Also
    --------
    _cv_mean_score : Uses this function to perform within-group permutations
        during cross-validation-based decoding.
    numpy.random.default_rng : Recommended constructor for modern RNGs.
    """
    if g is None:
        return rng.permutation(y)
    y_perm = np.copy(y)
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
    Compute the mean cross-validated decoding score across user-specified folds.

    This function performs outer-loop evaluation for ROI-based or searchlight decoding.
    For each fold, it builds a full decoding pipeline using
    :func:`pipeline_from_config`, fits on the training data, and scores on the
    test data. When ``permute=True``, it performs label permutations to estimate
    a null-distribution sample under the hypothesis of no label–feature
    relationship.

    The same scoring metric (configured via ``cfg['scoring']`` or
    :func:`factory.scorer_from_config`) is used for both observed and permuted
    evaluations, ensuring consistency between model optimization (e.g., during
    grid search) and final scoring.

    Parameters
    ----------
    X_path : str or Path
        Path to a ``joblib`` dump of a memory-mapped feature matrix
        ``(n_samples, n_features)``.
    labels : array-like of shape (n_samples,)
        Integer or categorical labels aligned with rows in ``X``.
    folds : list of tuple(ndarray, ndarray)
        List of (train_idx, test_idx) splits defining the outer cross-validation
        scheme (e.g., leave-one-run-out, leave-one-subject-out).
    cfg : dict
        Configuration dictionary controlling decoding parameters and pipeline
        construction. Passed to :func:`pipeline_from_config`.
    groups : array-like of shape (n_samples,), optional
        Optional grouping labels (e.g., run or subject IDs). Used both for
        stratified or grouped CV and for within-group label permutations.
    standardize : bool, default=True
        If True, enables the scaling step in the pipeline. When False, scaling
        is skipped (equivalent to a "passthrough" scaler).
    permute : bool, default=False
        If True, permutes labels in each fold to compute a null score.
    permute_both_sets : bool, default=True
        If True, permutes both training and test labels within each fold.
        If False, permutes only training labels (recommended for
        most decoding-based null distributions).
    permute_within_groups : bool, default=True
        If True and ``groups`` are provided, permutations are restricted
        to within-group shuffles via :func:`_permute_within_groups`.
        If False, labels are permuted globally using ``rng.permutation``.
    rng : numpy.random.Generator, optional
        Random number generator controlling label shuffling and per-model
        random_state seeds. If None and ``permute=True``, a default generator
        is created via :func:`numpy.random.default_rng()`.
    save_dir : str or Path, optional
        If provided and ``permute=False``, per-fold artifacts (fitted pipeline,
        metadata) are stored under ``save_dir/fold-XX`` using :func:`_save_pipeline`.
    roi_linidx : numpy.ndarray, optional
        Linear voxel indices corresponding to the ROI mask for saving fold outputs.
    **kwargs
        Additional keyword arguments forwarded to :func:`pipeline_from_config`
        (e.g., estimator- or searchlight-specific parameters). ``locked_params``
        may be used to disable grid search in searchlight mode.

    Returns
    -------
    float
        Mean cross-validation score across folds.

    Notes
    -----
    - A new pipeline is instantiated for each fold. If ``permute=True``, the
      same RNG is used to derive an integer ``random_state`` for that instance,
      ensuring reproducible results across permutations.
    - If the pipeline (or inner estimator) supports a ``groups`` argument
      (checked via :func:`sklearn.utils.validation.has_fit_parameter` or
      subclassing :class:`sklearn.model_selection.BaseSearchCV`), ``groups`` are
      passed to ``fit`` automatically.
    - Scoring consistency:
        * The same scorer is used for both observed and permuted evaluations.
        * Grid search (if configured) inherits the same scoring metric unless
          explicitly overridden in ``cfg['gridsearch']['args']``.

    Example
    -------
    .. code-block:: python

        folds = [(tr, te) for tr, te in logo.split(X_idx, y, groups)]
        mean_score = _cv_mean_score(
            X_path="/path/to/X.dump",
            labels=y,
            folds=folds,
            cfg=cfg,
            groups=groups,
            permute=False,
            save_dir="results/run-01",
            roi_linidx=roi_idx,
        )
        print(f"Mean CV {cfg['scoring']}: {mean_score:.3f}")

    See Also
    --------
    pipeline_from_config : Build scaler → selector → estimator pipeline.
    _permute_within_groups : Shuffle labels within group boundaries.
    _save_pipeline : Persist fitted models and metadata.
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

        # make scoring object
        scorer = factory.scorer_from_config(cfg.get("scoring", "balanced_accuracy"))
        
        # define the pipeline
        random_state = int(rng.integers(2**31 - 1)) if rng is not None else None
        clf = pipeline_from_config(
            cfg,
            random_state=random_state,
            labels=labels,
            scoring=scorer,
            **kwargs
        )

        supports_groups = isinstance(clf, BaseSearchCV) or has_fit_parameter(clf, "groups")
        if g_tr is not None and supports_groups:
            clf.fit(X_mm[train_idx], y_tr_perm, groups=g_tr)
        else:
            clf.fit(X_mm[train_idx], y_tr_perm)

        score = clf.score(X_mm[test_idx], y_te_perm)
        fold_scores.append(float(score))

        if not permute:
            if isinstance(fold_dir, str):
                _save_pipeline(
                    clf,
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
    Extract per-fold weights, decisions, predictions, and per-sample contributions
    from a fitted decoding pipeline, supporting both binary and multiclass classifiers.

    This function introspects a trained scikit-learn pipeline (optionally wrapped in
    a search object such as :class:`~sklearn.model_selection.GridSearchCV`) to recover:

      - the mapping of features retained after preprocessing back to the original ROI
        feature space,
      - the classifier’s linear weight vectors (expanded to full ROI length),
      - per-sample feature contributions,
      - decision values or probabilities, and
      - predicted labels for a specified test split.

    It supports both **binary** and **multiclass** linear estimators that expose
    ``coef_`` and either ``decision_function`` or ``predict_proba``.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline or sklearn.model_selection.BaseSearchCV
        Trained pipeline or search object. If wrapped in a cross-validation
        search (e.g., ``GridSearchCV``), the best estimator is extracted
        automatically via ``model.best_estimator_``.
        Expected step names:
        ``"var"`` (variance filter), ``"scaler"``, ``"select"`` (feature selector),
        and ``"clf"`` (classifier). Any of these may be set to ``'passthrough'``.
    X : numpy.ndarray of shape (n_samples, n_features)
        Feature matrix in the original ROI feature space used for model training.
    test_idx : array-like of int
        Indices of the test samples (outer cross-validation split) for which
        to compute predictions and feature contributions.

    Returns
    -------
    ddict : dict
        Dictionary containing extracted decoding information. Keys and shapes:

        - ``"test_idx"`` : numpy.ndarray
            Echo of the provided ``test_idx``.
        - ``"orig_idx"`` : numpy.ndarray of shape (n_selected_features,)
            Indices of features retained after variance thresholding and
            feature selection, mapped back to columns of the original ROI feature matrix.
        - ``"weights"`` :
            - **Binary:** numpy.ndarray of shape ``(n_features,)``
            - **Multiclass:** numpy.ndarray of shape ``(n_classes, n_features)``
            Full-length classifier weights expanded to the original ROI space
            (zeros for filtered-out features).
        - ``"contribution"`` :
            - **Binary:** numpy.ndarray of shape ``(n_test, n_features)``
            - **Multiclass:** numpy.ndarray of shape ``(n_test, n_features, n_classes)``
            Per-sample voxelwise contributions in ROI space, computed as the
            elementwise product between preprocessed feature activations
            (up to, but excluding, the classifier) and the corresponding
            class-specific weight vector.
        - ``"decision"`` : numpy.ndarray or None
            Output of the model’s ``decision_function`` or, if unavailable,
            ``predict_proba``. Shape depends on classifier type:
            - Binary: ``(n_test,)``
            - Multiclass: ``(n_test, n_classes)``.
        - ``"y_pred"`` : numpy.ndarray of shape (n_test,)
            Predicted labels for the test samples.

    Notes
    -----
    - The function assumes a **linear** model exposing ``coef_`` and a compatible
      decision interface. For nonlinear kernels (e.g., RBF SVM), the reported
      feature contributions are **not interpretable**.
    - When the classifier is multiclass (``coef_.shape == (n_classes, n_selected)``),
      weights and contributions are computed and returned per class.
    - Any preprocessing step absent from the pipeline (e.g., selector omitted)
      must appear as a named step with ``'passthrough'`` to ensure proper index mapping.
    - Feature indices are reconstructed by chaining the supports of
      ``VarianceThreshold`` and any selector step back to the original
      ROI feature ordering.
    - If neither ``decision_function`` nor ``predict_proba`` is implemented
      by the classifier, ``"decision"`` will be set to ``None``.

    Examples
    --------
    Binary decoding example:
    >>> ddict = extract_from_pipeline(trained_model, X, test_idx)
    >>> ddict["weights"].shape
    (X.shape[1],)
    >>> ddict["contribution"].shape
    (len(test_idx), X.shape[1])

    Multiclass decoding example:
    >>> ddict = extract_from_pipeline(trained_model, X, test_idx)
    >>> ddict["weights"].shape
    (n_classes, X.shape[1])
    >>> ddict["contribution"].shape
    (len(test_idx), X.shape[1], n_classes)

    See Also
    --------
    sklearn.pipeline.Pipeline
    sklearn.feature_selection.VarianceThreshold
    sklearn.linear_model.LogisticRegression
    sklearn.svm.LinearSVC
    """

    pipe = model.best_estimator_ if hasattr(model, "best_estimator_") else model

    var = pipe.named_steps.get("var", None)
    sel = pipe.named_steps.get("select", None)
    clf = pipe.named_steps["clf"]

    def _safe_get_support(step, n_feats):
        if step is None or step == "passthrough":
            return np.arange(n_feats)
        if hasattr(step, "get_support"):
            return step.get_support(indices=True)
        if hasattr(step, "support_"):
            sup = np.asarray(step.support_, dtype=bool)
            return np.flatnonzero(sup)
        # transformers like PCA have no support mask → identity here;
        # mapping to ROI is handled separately below for PCA.
        return np.arange(n_feats)

    n_features = X.shape[1]
    var_idx = _safe_get_support(var, n_features)

    # --- detect PCA in 'select' step
    is_pca = (sel is not None and sel != "passthrough" and hasattr(sel, "components_"))

    if not hasattr(clf, "coef_"):
        raise ValueError("Classifier does not expose `coef_`; cannot compute linear weights.")
    coef = clf.coef_
    if coef.ndim == 1:
        coef = coef[np.newaxis, :]  # (1, n_selected_or_pc)

    Xt = X[test_idx]  # original ROI space

    if is_pca:
        # 1) pre-PCA features (after var + scaler, before PCA)
        #    Build a small pipeline: var -> scaler (skip 'select' since it is PCA)
        #    We can apply steps one by one for clarity.
        Z = Xt
        if var is not None and var != "passthrough":
            # keep only var_idx features
            Z = Z[:, var_idx]
        scaler = pipe.named_steps.get("scaler", None)
        if scaler is not None and scaler != "passthrough":
            Z = scaler.transform(Z)  # (n_test, n_pre)

        # 2) map weights from PC space back to pre-PCA feature space
        #    sel.components_: (n_components, n_pre)
        comps = sel.components_  # rows are PCs
        # coef: (n_classes, n_components)
        # weights in pre-PCA feature space:
        w_pre = coef @ comps  # (n_classes, n_pre)

        # 3) expand to full ROI by placing pre-PCA weights at var_idx
        n_classes = w_pre.shape[0]
        w_full = np.zeros((n_classes, n_features), dtype=float)
        w_full[:, var_idx] = w_pre

        # 4) contributions in ROI space (per class)
        #    Z is (n_test, n_pre); contributions per feature = Z * w_pre (broadcast)
        contrib_pre = Z[:, None, :] * w_pre[None, :, :]  # (n_test, n_classes, n_pre)
        # expand to ROI space
        n_test = Z.shape[0]
        contrib_full = np.zeros((n_test, n_features, n_classes), dtype=float)
        for c in range(n_classes):
            contrib_full[:, var_idx, c] = contrib_pre[:, c, :]

        # 5) decisions / predictions from pipeline (already includes PCA step)
        if hasattr(pipe, "decision_function"):
            dec = pipe.decision_function(Xt)
        elif hasattr(pipe, "predict_proba"):
            dec = pipe.predict_proba(Xt)
        else:
            dec = None
        y_pred = pipe.predict(Xt)

        # 6) orig_idx for bookkeeping: with PCA we didn’t select a subset after var
        #    (all var_idx are used, just linearly remixed), so:
        orig_idx = var_idx

        # shapes as in the multiclass doc:
        if n_classes == 1:
            weights_out = w_full[0]
            contrib_out = contrib_full[:, :, 0]
        else:
            weights_out = w_full
            contrib_out = contrib_full

        return {
            "test_idx": np.asarray(test_idx),
            "orig_idx": orig_idx,
            "weights": weights_out,
            "contribution": contrib_out,
            "decision": dec,
            "y_pred": y_pred,
        }

    # --- non-PCA path (selectors with get_support or passthrough) -------------
    sel_idx = _safe_get_support(sel, len(var_idx))
    orig_idx = var_idx[sel_idx]

    n_selected = len(orig_idx)
    if coef.shape[1] != n_selected:
        raise ValueError(
            f"Classifier coef_ width ({coef.shape[1]}) != selected features ({n_selected})."
        )

    # expand weights back to ROI space
    n_classes = coef.shape[0]
    w_full = np.zeros((n_classes, n_features), dtype=float)
    w_full[:, orig_idx] = coef

    # decisions / predictions
    if hasattr(pipe, "decision_function"):
        dec = pipe.decision_function(Xt)
    elif hasattr(pipe, "predict_proba"):
        dec = pipe.predict_proba(Xt)
    else:
        dec = None
    y_pred = pipe.predict(Xt)

    # contributions in selected space
    Xt_test = pipe[:-1].transform(Xt)  # includes 'select' since it's not PCA
    contrib_sel = Xt_test[:, None, :] * coef[None, :, :]  # (n_test, n_classes, n_selected)

    # expand to ROI
    n_test = Xt_test.shape[0]
    contrib_full = np.zeros((n_test, n_features, n_classes), dtype=float)
    for c in range(n_classes):
        contrib_full[:, orig_idx, c] = contrib_sel[:, c, :]

    if n_classes == 1:
        weights_out = w_full[0]
        contrib_out = contrib_full[:, :, 0]
    else:
        weights_out = w_full
        contrib_out = contrib_full

    return {
        "test_idx": np.asarray(test_idx),
        "orig_idx": orig_idx,
        "weights": weights_out,
        "contribution": contrib_out,
        "decision": dec,
        "y_pred": y_pred,
    }

def _save_pipeline(model, X, test_idx, train_idx, output_dir, roi_linidx=None):
    """
    Save a fitted decoding model and fold-specific outputs to disk (ROI & searchlight).

    This utility serializes everything needed to inspect or reproduce a single
    outer-CV fold. It works whether `model` is a tuned
    :class:`~sklearn.model_selection.GridSearchCV` (ROI path) or a plain
    :class:`~sklearn.pipeline.Pipeline` (searchlight path with fixed hyperparameters).

    The function:
      1) extracts fold outputs via :func:`extract_from_pipeline` (now multiclass-aware),
      2) saves them as `.npy` arrays,
      3) dumps the fitted estimator with :func:`joblib.dump`,
      4) writes a JSON of the chosen parameters,
      5) optionally stores the ROI voxel mapping, and
      6) for classification, stores the classifier’s ``classes_`` ordering.

    Parameters
    ----------
    model : sklearn.model_selection.BaseSearchCV or sklearn.pipeline.Pipeline
        Fitted model to be saved. If it has ``best_estimator_``, that estimator
        is used for extraction and serialization.
    X : numpy.ndarray of shape (n_samples, n_features)
        Full feature matrix used during fitting (ROI feature space).
    test_idx : array-like of int
        Indices of the test samples for this outer fold.
    train_idx : array-like of int
        Indices of the training samples for this outer fold (not saved by default,
        but accepted for completeness and potential extensions).
    output_dir : str or pathlib.Path
        Directory to write all artifacts. Created if it does not exist.
    roi_linidx : numpy.ndarray, optional
        1D array mapping ROI feature columns back to voxel linear indices; saved
        as ``roi_linidx.npy`` when provided.

    Returns
    -------
    None
        Artifacts are written to ``output_dir``.

    Saved Files
    -----------
    - ``fold_k_test_idx.npy`` : test sample indices (echo of `test_idx`).
    - ``fold_k_orig_idx.npy`` : indices of retained ROI features after var/selector.
    - ``fold_k_weights.npy`` :
        * **Binary:** shape ``(n_features,)``
        * **Multiclass:** shape ``(n_classes, n_features)``
    - ``fold_k_contribution.npy`` :
        * **Binary:** shape ``(n_test, n_features)``
        * **Multiclass:** shape ``(n_test, n_features, n_classes)``
    - ``fold_k_decision.npy`` *(optional)* :
        Decision values if available:
        * **Binary:** shape ``(n_test,)``
        * **Multiclass:** shape ``(n_test, n_classes)``
        Omitted if the estimator exposes neither ``decision_function`` nor
        ``predict_proba``.
    - ``fold_k_y_pred.npy`` : predicted labels (shape ``(n_test,)``).
    - ``fold_k_classes.npy`` *(optional)* :
        Class label order used by the classifier (``clf.classes_``), saved when present.
        Important for interpreting columns of multiclass outputs.
    - ``fold_k_best_estimator.joblib`` : serialized fitted pipeline/estimator.
    - ``fold_k_best_params.json`` : chosen parameter dictionary with an extra
      flag ``"_searchlight_mode"`` indicating whether the model came from
      searchlight (True) or ROI (False).
    - ``roi_linidx.npy`` *(optional)* : voxel index mapping (if provided).

    Behavior & Notes
    ----------------
    - If ``model`` has ``best_estimator_``, it is treated as a search object and
      the tuned estimator is used. Otherwise, the pipeline is used directly.
    - Parameters are taken from ``model.best_params_`` when available; otherwise
      from ``fitted.named_steps["clf"].get_params()`` (fallback to
      ``fitted.get_params(deep=False)``).
    - Multiclass compatibility:
        * Weights and contributions are saved in per-class form when applicable.
        * The classifier’s ``classes_`` (if present) is saved to disambiguate class order.
    - Existing files with the same names are overwritten.

    Examples
    --------
    >>> _save_pipeline(
    ...     model=trained_model,
    ...     X=X_mm,
    ...     test_idx=test_idx,
    ...     train_idx=train_idx,
    ...     output_dir="results/roi_01/fold-00",
    ...     roi_linidx=roi_idx,
    ... )
    """
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # If ROI/GridSearchCV: pull tuned estimator; else keep the fitted pipeline
    is_grid = hasattr(model, "best_estimator_")
    fitted = model.best_estimator_ if is_grid else model

    # 1) Extract fold outputs (binary & multiclass supported)
    ddict = extract_from_pipeline(fitted, X, test_idx)

    # Save everything extractable, but skip keys with None values (e.g., decision=None)
    for key, val in ddict.items():
        if val is None:
            # skip saving absent artifacts (e.g., no decision_function/proba)
            continue
        np.save(opj(output_dir, f"fold_k_{key}.npy"), val)

    # 1a) Save classifier classes_ ordering if available (helps interpret multiclass)
    try:
        clf = fitted.named_steps["clf"]
        if hasattr(clf, "classes_"):
            np.save(opj(output_dir, "fold_k_classes.npy"), clf.classes_)
    except Exception:
        # pipeline might not have a "clf" step (unlikely here); ignore silently
        pass

    # 2) Save the fitted estimator
    dump(fitted, opj(output_dir, "fold_k_best_estimator.joblib"))

    # 3) Save params JSON (robust to both cases)
    if is_grid:
        best_params = model.best_params_
    else:
        # Prefer classifier-only params to keep file small; fall back to pipeline params
        try:
            best_params = fitted.named_steps["clf"].get_params()
        except Exception:
            best_params = fitted.get_params(deep=False)

    out_params = {"_searchlight_mode": (not is_grid), **best_params}
    with open(opj(output_dir, "fold_k_best_params.json"), "w") as f:
        json.dump(out_params, f, indent=2)

    # 4) Save ROI mapping if provided
    if isinstance(roi_linidx, np.ndarray):
        np.save(opj(output_dir, "roi_linidx.npy"), roi_linidx)


def _folds_for_labels(cfg, labels, groups=None):
    """
    Generate outer cross-validation folds for decoding, supporting both
    group-based and legacy within-class split schemes.

    This helper determines how outer folds are constructed for decoding analyses,
    depending on the configuration flags in ``cfg``. It mirrors the ROI decoding
    pipeline logic while maintaining backward compatibility with legacy
    “every-kth” splitting for simpler setups.

    Specifically:
      - When ``permute_within_groups=True``, group-aware CV is performed using
        a splitter constructed via :func:`factory.cv_from_config`.
      - When ``permute_within_groups=False``, a deterministic “every-kth”
        split scheme is used, in which samples are divided within each class.

    Parameters
    ----------
    cfg : dict
        Decoding configuration dictionary. Expected keys include:

        - ``"permute_within_groups"`` : bool
            Whether to perform group-aware outer CV (e.g., leave-one-run-out).
        - ``"outer_cv"`` : dict
            Cross-validation configuration passed to
            :func:`factory.cv_from_config` when group-based splitting is used.
        - ``"fold_interval"`` : int
            Interval controlling the legacy “every-kth” folding strategy,
            used when ``permute_within_groups=False``.
    labels : array-like of shape (n_samples,)
        Class labels for decoding (binary or multiclass). Must be aligned
        with the feature matrix used elsewhere in decoding.
    groups : array-like of shape (n_samples,), optional
        Group identifiers used for cross-validation when
        ``permute_within_groups=True``. Typically corresponds to runs,
        sessions, or subjects.

    Returns
    -------
    folds : list of tuple(ndarray, ndarray)
        List of (train_idx, test_idx) pairs defining each outer fold.
        Each index array contains integer sample indices for the corresponding split.

    Notes
    -----
    **Group-aware mode (``permute_within_groups=True``)**
        - Uses :func:`factory.cv_from_config(cfg["outer_cv"])` to instantiate
          a cross-validation splitter (e.g., StratifiedKFold, LeaveOneGroupOut,
          StratifiedGroupKFold).
        - Generates folds by calling ``split(dummy_X, labels, groups)``, where
          ``dummy_X`` is a placeholder array of zeros with length equal to ``labels``.
        - Ensures group-respecting splits compatible with permutation-based
          decoding when labels are shuffled within groups.

    **Legacy “every-kth” mode (``permute_within_groups=False``)**
        - Implements a deterministic split that partitions samples within
          each class label such that every *k*-th sample (as specified by
          ``fold_interval``) is used for testing in a distinct fold.
        - Training indices are all remaining samples not in the current test set.
        - The number of folds equals the value of ``fold_interval``.

    **Multiclass Behavior**
        - For multiclass data, the legacy path applies the same “every-kth”
          logic independently to each class label, ensuring each class
          contributes approximately equal samples per fold.
        - When ``permute_within_groups=True``, any valid scikit-learn CV
          splitter compatible with multiclass labels can be used.

    Examples
    --------
    Group-aware outer CV (recommended for ROI decoding):

    >>> cfg = {
    ...     "permute_within_groups": True,
    ...     "outer_cv": {"type": "LeaveOneGroupOut"}
    ... }
    >>> folds = _folds_for_labels(cfg, labels, groups=runs)
    >>> print(f"{len(folds)} outer folds generated")

    Legacy “every third” fallback (simple binary or multiclass):

    >>> cfg = {"permute_within_groups": False, "fold_interval": 3}
    >>> folds = _folds_for_labels(cfg, labels)
    >>> len(folds)
    3

    See Also
    --------
    factory.cv_from_config : Builds scikit-learn CV splitters from configuration.
    sklearn.model_selection.StratifiedKFold
    sklearn.model_selection.LeaveOneGroupOut
    sklearn.model_selection.StratifiedGroupKFold

    Notes
    -----
    - The legacy mode assumes roughly balanced class distributions.
      It is deterministic and ensures that each sample appears once
      in the test set across all folds.
    - The group-aware path is recommended for reproducibility and
      consistency with ROI-based decoding pipelines.
    - When labels are permuted within groups during permutation testing,
      group-aware CV ensures that group boundaries are respected.
    """

    labs = np.asarray(labels)
    n = len(labs)
    
    if cfg.get("permute_within_groups"):
        cv = factory.cv_from_config(cfg["outer_cv"])
        dummy_X = np.zeros((n, 1), dtype=int)
        folds = [(tr, te) for tr, te in (
            cv.split(dummy_X, labs, groups) if groups is not None else cv.split(dummy_X, labs)
        )]
        return folds

    # legacy “every third within class” (your define_folds)
    k = int(cfg.get("fold_interval", 3))
    classes = np.unique(labs)
    folds = []
    all_idx = np.arange(n)

    # Build test sets by taking every k-th index *within each class*, then union across classes
    per_class_indices = {c: np.where(labs == c)[0] for c in classes}
    for off in range(k):
        te_slices = [idxs[off::k] for idxs in per_class_indices.values()]
        te = np.concatenate(te_slices, dtype=int) if te_slices else np.empty(0, int)
        tr = np.setdiff1d(all_idx, te, assume_unique=False)
        folds.append((tr, te))

    return folds

def recon_contribution(roi, lin_roi, contrib_path, output_file=None):
    """
    Reconstruct per-trial and mean contribution maps into a 4D NIfTI image.

    This function takes a matrix of per-trial voxel contributions (e.g., model
    weights × activations) defined over an ROI and projects them back into the
    3D image space of a reference NIfTI. It returns (or saves) a 4D image with
    one volume per trial plus a final volume that is the voxelwise mean across
    trials.

    Parameters
    ----------
    roi : str or nibabel.Nifti1Image
        Reference image (or path to it) that supplies the target spatial shape
        and affine. Typically an ROI mask or a reference beta/mean image.
        Only ``img.shape[:3]`` and ``img.affine`` are used.
    lin_roi : str
        Path to a ``.npy`` file containing the **linear voxel indices** (1D) of
        the ROI in the reference image. These indices must match the column
        order of ``contrib`` and assume **NumPy C-order raveling**
        (i.e., consistent with ``np.ravel`` / ``reshape(..., order='C')``).
    contrib_path : str
        Path to a ``.npy`` file with shape ``(n_trials, n_voxels_in_roi)`` containing
        per-trial contribution values. A 1D array of shape ``(n_voxels_in_roi,)`` is
        also accepted and will be treated as a single trial.
    output_file : str or None, optional
        If provided, the reconstructed 4D NIfTI is written to this path and the
        function returns ``None``. If ``None`` (default), a
        :class:`nibabel.Nifti1Image` is returned instead.

    Returns
    -------
    img4d : nibabel.Nifti1Image or None
        If ``output_file`` is ``None``, returns a NIfTI image with shape
        ``(X, Y, Z, n_trials + 1)`` where the last volume is the mean across
        trials. If ``output_file`` is set, returns ``None`` after saving.

    Notes
    -----
    - Linear indices in ``lin_roi`` must reference the flattened reference grid
      in **C-order** (as produced by ``np.ravel``). If your indices were created
      with Fortran-order or a different flattening convention, the reconstruction
      will be spatially misaligned.
    - The output affine is copied from the reference image. The reference header
      is not propagated; modify as needed before saving if header fields matter.
    - Inputs are loaded via :func:`numpy.load` and must be saved as ``.npy``.
    - A copy-safe path is used: the function allocates a zero-filled 1D buffer
      per volume, assigns ROI values by index, then reshapes to 3D.

    Examples
    --------
    Return an in-memory 4D image:

    >>> img = recon_contribution(
    ...     roi="roi_mask.nii.gz",
    ...     lin_roi="roi_linidx.npy",
    ...     contrib_path="sub-01_contrib.npy"
    ... )
    >>> import nibabel as nib
    >>> nib.save(img, "sub-01_contrib_4d.nii.gz")

    Save directly to disk:

    >>> recon_contribution(
    ...     roi="roi_mask.nii.gz",
    ...     lin_roi="roi_linidx.npy",
    ...     contrib_path="sub-01_contrib.npy",
    ...     output_file="sub-01_contrib_4d.nii.gz"
    ... )

    See Also
    --------
    numpy.ravel, numpy.reshape
    nibabel.Nifti1Image
    """

    # load reference image ---
    if isinstance(roi, str):
        img = nib.load(roi)
    elif isinstance(roi, nib.Nifti1Image):
        img = roi
    else:
        raise TypeError("`roi` must be a file path or Nifti1Image object.")

    affine = img.affine
    roi_shape = img.shape[:3]

    # load data ---
    roi_linidx = np.load(lin_roi)
    contrib = np.load(contrib_path)  # (n_trials, n_voxels_in_roi)

    if contrib.ndim == 1:
        contrib = contrib[None, :]  # ensure 2D

    n_trials, n_vox = contrib.shape
    assert len(roi_linidx) == n_vox, \
        f"ROI voxels ({len(roi_linidx)}) != contribution size ({n_vox})"

    # reconstruct each 3D volume ---
    vols = []
    for i in range(n_trials):
        vol = np.zeros(np.prod(roi_shape), dtype=np.float32)
        vol[roi_linidx] = contrib[i]
        vols.append(vol.reshape(roi_shape))

    # append the mean volume ---
    mean_vol = np.zeros(np.prod(roi_shape), dtype=np.float32)
    mean_vol[roi_linidx] = contrib.mean(axis=0)
    vols.append(mean_vol.reshape(roi_shape))

    img4d = np.stack(vols, axis=-1)
    w_img = nib.Nifti1Image(img4d, affine)

    # save or return ---
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

    # Load reference image
    if isinstance(roi, str):
        img = nib.load(roi)
    elif isinstance(roi, nib.Nifti1Image):
        img = roi
    else:
        raise TypeError("`roi` must be a file path or Nifti1Image object.")

    affine = img.affine
    roi_shape = img.shape[:3]

    # Load data
    roi_linidx = np.load(lin_roi)
    weights = np.load(weights_path)

    if weights.ndim != 1:
        raise ValueError(f"Expected 1D weight vector, got shape {weights.shape}")

    assert len(roi_linidx) == weights.shape[0], \
        f"ROI voxels ({len(roi_linidx)}) != weights size ({weights.shape[0]})"

    # Fill into 3D volume
    full_map = np.zeros(np.prod(roi_shape), dtype=np.float32)
    full_map[roi_linidx] = weights
    w_img = nib.Nifti1Image(full_map.reshape(roi_shape), affine)

    # Save or return
    if output_file is not None:
        nib.save(w_img, output_file)
        print(f"[INFO] Saved weight map: {output_file}")
        return None
    else:
        return w_img    