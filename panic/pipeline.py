# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import json
import numpy as np
import nibabel as nib
from joblib import dump
from sklearn.utils.validation import has_fit_parameter
from sklearn.model_selection._search import BaseSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

from panic.logger import get_logger
from panic import (
    factory,
    errors
)

from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import (
    VarianceThreshold
)
from sklearn.pipeline import Pipeline

from typing import Any, Callable, Optional

import warnings

FoldScoreFunction = Callable[..., float]


logger = get_logger(__name__)
opj = os.path.join


class FailIfNoFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if X.shape[1] == 0:
            raise errors.NoFeaturesSelectedError("No features left after preprocessing.")
        return self
    def transform(self, X):
        if X.shape[1] == 0:
            raise errors.NoFeaturesSelectedError("No features left after preprocessing.")
        return X
    

def pipeline_from_config(
        cfg,
        *,
        searchlight: bool = False,
        standardize: bool = False,
        random_state=None,
        labels=None,
        scoring=None,
        locked=None,
        **kwargs
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
        ("check", FailIfNoFeatures()),
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
        inner_cv_cfg = cfg.get("inner_cv")

        if inner_cv_cfg is None:
            raise ValueError(
                "`inner_cv` must be configured when grid search is enabled."
            )

        inner_cv = factory.cv_from_config(inner_cv_cfg)

        gs_args = dict(gs_cfg.get("args", {}))

        if "scoring" not in gs_args and "scoring" in cfg:
            gs_args["scoring"] = cfg["scoring"]

        gs_args.setdefault("error_score", "raise")

        gs_cfg = {
            **gs_cfg,
            "args": gs_args,
        }

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


def sklearn_pipeline_score(
    X,
    labels,
    train_idx,
    test_idx,
    *,
    cfg,
    groups=None,
    rng=None,
    **kwargs,
) -> float:
    """Fit and score a configured sklearn pipeline on one CV split.

    Construct the sklearn estimator pipeline from the decoding
    configuration, fit it on the samples specified by ``train_idx``,
    and evaluate it on the held-out samples specified by ``test_idx``.

    This function represents a single outer cross-validation fold. The
    configured pipeline may itself contain an inner model-selection step,
    such as ``GridSearchCV``. In that case, hyperparameter optimization is
    performed using only the outer training data.

    If grouping information is provided and the estimator supports
    ``groups`` during fitting, the groups corresponding to the training
    samples are passed to ``fit``. This allows group-aware inner
    cross-validation without exposing information from the outer test set.

    A random state is derived from ``rng`` when provided and passed to the
    pipeline factory, allowing stochastic pipeline components to remain
    reproducible across observed and permutation analyses.

    Convergence warnings raised by estimators fitted internally by a
    ``BaseSearchCV`` object are suppressed. Individual hyperparameter
    candidates may occasionally reach their configured iteration limit,
    particularly for permuted data, without invalidating the surrounding
    model-selection procedure. Convergence warnings from estimators fitted
    outside a search object are not suppressed.

    If preprocessing or feature selection leaves no usable features,
    ``NoFeaturesSelectedError`` is converted to ``NaN``. This allows the
    calling cross-validation routine to identify folds for which no valid
    score could be obtained and handle them collectively.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix containing all samples available to the current
        analysis.

    labels : array-like of shape (n_samples,)
        Target labels corresponding to the rows of ``X``.

    train_idx : array-like of int
        Indices selecting the samples used to fit the pipeline.

    test_idx : array-like of int
        Indices selecting the held-out samples used for scoring.

    cfg : dict
        Decoding configuration used to construct the sklearn pipeline,
        scorer, estimator, feature-selection steps, and optional
        hyperparameter search.

    groups : array-like of shape (n_samples,), optional
        Group labels used by group-aware fitting or inner
        cross-validation. Only the groups corresponding to ``train_idx``
        are passed to the estimator.

    rng : numpy.random.Generator, optional
        Random-number generator used to derive the random state supplied
        to the pipeline. If ``None``, no random state is generated here.

    **kwargs
        Additional keyword arguments forwarded to
        ``pipeline_from_config``.

    Returns
    -------
    float
        Score of the fitted pipeline on the outer test set. Returns
        ``NaN`` when no features remain after preprocessing or feature
        selection.

    Notes
    -----
    The outer test samples are never used during pipeline fitting or
    hyperparameter selection. When ``clf`` is a ``BaseSearchCV`` object,
    its internal cross-validation operates exclusively on ``X_train`` and
    ``y_train``.

    Suppression of ``ConvergenceWarning`` is intentionally restricted to
    estimators fitted inside ``BaseSearchCV``. A finite estimator
    ``max_iter`` can therefore be retained as a safeguard against
    pathological candidate fits while avoiding excessive warnings from
    individual inner-CV fits.
    """
    random_state = (
        int(rng.integers(2**31 - 1))
        if rng is not None
        else None
    )

    scorer = factory.scorer_from_config(
        cfg.get("scoring", "balanced_accuracy")
    )

    clf = pipeline_from_config(
        cfg,
        random_state=random_state,
        labels=labels,
        scoring=scorer,
        **kwargs,
    )

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    groups_train = (
        None
        if groups is None
        else np.asarray(groups)[train_idx]
    )

    supports_groups = (
        isinstance(clf, BaseSearchCV)
        or has_fit_parameter(clf, "groups")
    )

    try:
        # Suppress convergence warnings only for inner search candidates.
        # Plain/final estimators still emit ConvergenceWarning normally.
        if isinstance(clf, BaseSearchCV):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=ConvergenceWarning,
                )

                if groups_train is not None and supports_groups:
                    clf.fit(
                        X_train,
                        y_train,
                        groups=groups_train,
                    )
                else:
                    clf.fit(
                        X_train,
                        y_train,
                    )

        else:
            if groups_train is not None and supports_groups:
                clf.fit(
                    X_train,
                    y_train,
                    groups=groups_train,
                )
            else:
                clf.fit(
                    X_train,
                    y_train,
                )

        return float(
            clf.score(
                X_test,
                y_test,
            )
        )

    except errors.NoFeaturesSelectedError:
        return float("nan")


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

    # non-PCA path (selectors with get_support or passthrough)
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
    Xt_test = Xt
    for name, step in pipe.steps[:-1]:
        if step is None or step == "passthrough":
            continue
        Xt_test = step.transform(Xt_test)

    contrib_sel = Xt_test[:, None, :] * coef[None, :, :]

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


def create_outer_folds(cfg, labels, groups=None):
    """Construct outer train/test folds from the decoding configuration.

    Parameters
    ----------
    cfg : dict
        Decoding configuration containing an ``outer_cv`` section.

        Standard scikit-learn splitters use::

            outer_cv:
                name: LeaveOneGroupOut
                args: {}

        Custom per-class n-th-trial splitting uses::

            outer_cv:
                mode: nth_trial
                args:
                    fold_interval: 3

        If ``mode`` is omitted, ``"sklearn"`` is assumed.

    labels : array-like of shape (n_samples,)
        Class labels aligned with the samples being decoded.

    groups : array-like of shape (n_samples,), optional
        Group identifiers aligned with ``labels``. Required by group-aware
        splitters such as ``LeaveOneGroupOut`` and ``GroupKFold``.

    Returns
    -------
    list[tuple[numpy.ndarray, numpy.ndarray]]
        List of ``(train_idx, test_idx)`` pairs.

    Raises
    ------
    ValueError
        If labels are empty, labels and groups have different lengths,
        a group-aware splitter is selected without providing ``groups``,
        ``fold_interval`` is invalid, a generated fold contains no test
        samples, or ``outer_cv.mode`` is unknown.

    Notes
    -----
    The outer CV folds are constructed once from the original labels and are
    kept fixed for the observed analysis and all permutation realizations.

    This is particularly important for ``outer_cv.mode == "nth_trial"``.
    That splitter assigns test samples by taking every k-th trial separately
    within each original class. The resulting train/test assignments therefore
    represent the fixed Bach-style CV design.

    Permutation inference changes only the association between samples and
    class labels; it does not reconstruct the CV folds from the permuted
    labels. Reconstructing folds for every permutation would simultaneously
    alter both the labels and the train/test partition and would therefore
    evaluate a different CV design from the observed statistic.

    The intended scheme is::

        original labels y
            |
            +----> create_outer_folds(y) ----> fixed folds F
            |                                   |
            |                                   +--> observed:
            |                                   |      CV(X, y, F)
            |                                   |
            +----> permutation 1: y_perm_1 -----+--> CV(X, y_perm_1, F)
            |
            +----> permutation 2: y_perm_2 -----+--> CV(X, y_perm_2, F)
            |
            ...
            |
            +----> permutation P: y_perm_P -----+--> CV(X, y_perm_P, F)

    Each ``y_perm`` is generated once and is used consistently across all
    folds. Thus each permutation produces one coherent cross-validated null
    score.

    When ``groups`` are supplied and ``permute_within_groups=True``, labels
    are shuffled within those exchangeability blocks before the fixed folds
    are evaluated.        
    """
    labs = np.asarray(labels)

    if labs.ndim != 1:
        labs = labs.ravel()

    n_samples = len(labs)

    if n_samples == 0:
        raise ValueError("`labels` must contain at least one sample.")

    cv_cfg = cfg.get("outer_cv", {})

    if not cv_cfg:
        raise ValueError(
            "No `outer_cv` configuration was provided."
        )

    mode = cv_cfg.get("mode", "sklearn")

    # --------------------------------------------------------------
    # Standard scikit-learn CV
    if mode == "sklearn":
        if "name" not in cv_cfg:
            raise ValueError(
                "`outer_cv.name` is required when mode is 'sklearn'."
            )

        sklearn_cv_cfg = {
            "name": cv_cfg["name"],
            "args": cv_cfg.get("args", {}),
        }

        logger.info(
            "Creating outer folds with scikit-learn: %s",
            sklearn_cv_cfg,
        )

        cv = factory.cv_from_config(sklearn_cv_cfg)

        dummy_X = np.zeros(
            (n_samples, 1),
            dtype=np.uint8,
        )

        # check if splitter requires groups (e.g., LeaveOneGroupOut)
        requires_groups = factory._splitter_accepts_groups(cv)
        groups_arr = None
        if requires_groups:

            # check if groups were passed
            if groups is not None:
                groups_arr = np.asarray(groups)

                if groups_arr.ndim != 1:
                    groups_arr = groups_arr.ravel()

                if len(groups_arr) != n_samples:
                    raise ValueError(
                        "Number of group labels must match number of samples: "
                        f"{len(groups_arr)} != {n_samples}."
                    )
            else:     
                raise ValueError(
                    f"{type(cv).__name__} requires `groups`, "
                    "but no group labels were provided."
                )

        try:
            if groups_arr is not None:
                folds = list(
                    cv.split(
                        dummy_X,
                        labs,
                        groups=groups_arr,
                    )
                )
            else:
                folds = list(
                    cv.split(
                        dummy_X,
                        labs,
                    )
                )

        except ValueError as exc:
            raise ValueError(
                f"Could not construct outer folds with "
                f"{type(cv).__name__}: {exc}"
            ) from exc

        return folds

    # --------------------------------------------------------------
    # Bach-style per-class n-th-trial CV
    if mode == "nth_trial":
        k = int(
            cv_cfg.get("args", {}).get(
                "fold_interval",
                3,
            )
        )

        if k < 2:
            raise ValueError(
                "`fold_interval` must be >= 2 for nth-trial CV."
            )

        logger.info(
            "Creating outer folds with per-class n-th trial: "
            "interval=%d (Bach et al. 2011)",
            k,
        )

        all_idx = np.arange(n_samples)

        per_class_indices = {
            class_label: np.flatnonzero(
                labs == class_label
            )
            for class_label in np.unique(labs)
        }

        # double check validity
        class_counts = {
            label: len(indices)
            for label, indices in per_class_indices.items()
        }

        too_small = {
            label: count
            for label, count in class_counts.items()
            if count < k
        }

        if too_small:
            raise ValueError(
                "`fold_interval` is larger than the number of "
                f"samples in one or more classes: {too_small}"
            )

        # create folds
        folds = []

        for offset in range(k):
            test_slices = [
                indices[offset::k]
                for indices in per_class_indices.values()
            ]

            test_idx = np.concatenate(test_slices)

            # Preserve original sample/trial order.
            test_idx = np.sort(test_idx)

            if test_idx.size == 0:
                raise ValueError(
                    f"Outer fold {offset} contains no test samples. "
                    f"`fold_interval={k}` may be too large."
                )

            train_mask = np.ones(
                n_samples,
                dtype=bool,
            )
            train_mask[test_idx] = False

            train_idx = all_idx[train_mask]

            folds.append(
                (train_idx, test_idx)
            )

        return folds


    raise ValueError(
        f"Unknown outer CV mode {mode!r}. "
        "Expected 'sklearn' or 'nth_trial'."
    )


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


def cv_mean_function_score(
    X,
    labels,
    folds,
    score_fn: FoldScoreFunction,
    *,
    groups=None,
    permute: bool = False,
    permute_within_groups: bool = True,
    rng: Optional[np.random.Generator] = None,
    **score_kwargs: Any,
) -> float:
    """
    Compute the mean cross-validated score for one observed or permuted dataset.

    Labels are permuted once per call and the resulting label vector is used
    consistently across all fixed CV folds.

    Before fitting, every training fold is checked to ensure that it still
    contains all classes represented in the original label vector. This is
    mainly a safeguard for permutation analyses, where a rare shuffle could
    otherwise produce a single-class training set and cause the classifier to
    fail during fitting.

    Scheme
    ------

    Observed::

        original y
            |
            +---- fixed fold 1 ----> score_1
            |
            +---- fixed fold 2 ----> score_2
            |
            ...
            |
            +---- fixed fold K ----> score_K
                                      |
                                      v
                                  mean score


    Permutation::

        original y
            |
            | permute ONCE
            v
         y_perm
            |
            | validate all training folds
            |
            +---- fixed fold 1 ----> score_1
            |
            +---- fixed fold 2 ----> score_2
            |
            ...
            |
            +---- fixed fold K ----> score_K
                                      |
                                      v
                               permutation score

    A complete permutation test repeats this function with independent random
    seeds, producing one cross-validated null score per permutation.

    Parameters
    ----------
    X : array-like
        Feature matrix of shape ``(n_samples, n_features)``.

    labels : array-like of shape (n_samples,)
        Original labels aligned with rows of ``X``.

    folds : iterable of tuple(ndarray, ndarray)
        Fixed outer-CV splits represented as ``(train_idx, test_idx)``.

    score_fn : FoldScoreFunction
        Function that fits and evaluates one train/test fold.

    groups : array-like, optional
        Exchangeability-group labels. When
        ``permute_within_groups=True``, labels are shuffled only within these
        groups.

    permute : bool, default=False
        If True, construct one permutation of ``labels`` before evaluating the
        fixed folds.

    permute_within_groups : bool, default=True
        Restrict permutation to groups when ``groups`` is provided.

    rng : numpy.random.Generator, optional
        Random-number generator used for permutation and downstream stochastic
        model operations.

    **score_kwargs
        Additional arguments forwarded to ``score_fn``.

    Returns
    -------
    float
        Mean score across valid CV folds.

    Raises
    ------
    ValueError
        If a training fold does not contain all classes represented in the
        original label vector.

    errors.NoFeaturesSelectedError
        If all folds produce NaN scores.
    """
    X = np.asarray(X)
    y = np.asarray(labels)

    groups = (
        None
        if groups is None
        else np.asarray(groups)
    )

    original_classes = np.unique(y)

    # ---------------------------------------------------------------
    # One coherent label vector for this complete CV realization.
    if permute:
        if rng is None:
            rng = np.random.default_rng()

        if permute_within_groups and groups is not None:
            y_eval = _permute_within_groups(
                y,
                groups,
                rng,
            )
        else:
            y_eval = rng.permutation(y)

    else:
        y_eval = y

    # ---------------------------------------------------------------
    # Validate training folds before fitting.
    #
    # This is especially useful for permutation testing: a pathological
    # shuffle should fail here with a clear error rather than inside SVC,
    # LogisticRegression, etc.
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_classes = np.unique(y_eval[train_idx])

        if not np.array_equal(
            np.sort(train_classes),
            np.sort(original_classes),
        ):
            raise ValueError(
                f"CV fold {fold_i} does not contain all classes in its "
                f"training set. Expected {original_classes.tolist()}, "
                f"found {train_classes.tolist()}."
            )

    # ---------------------------------------------------------------
    # Cross-validation.
    fold_scores = []

    for train_idx, test_idx in folds:
        score = score_fn(
            X,
            y_eval,
            train_idx,
            test_idx,
            groups=groups,
            rng=rng,
            **score_kwargs,
        )

        fold_scores.append(float(score))

    fold_scores = np.asarray(fold_scores, dtype=float)

    if not np.any(np.isfinite(fold_scores)):
        raise errors.NoFeaturesSelectedError(
            "All folds produced NaN scores; cannot compute mean score."
        )

    mean_score = float(np.nanmean(fold_scores))

    if np.isnan(mean_score):
        raise errors.NoFeaturesSelectedError(
            "All folds produced NaN scores; cannot compute mean score."
        )

    return mean_score
