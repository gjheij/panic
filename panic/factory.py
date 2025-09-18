# cv_factory.py
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit,
    GroupKFold, StratifiedGroupKFold, LeaveOneGroupOut, LeavePGroupsOut,
    TimeSeriesSplit, PredefinedSplit,
    GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
)

from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel, RFE, RFECV, VarianceThreshold,
    f_classif, f_regression, chi2, mutual_info_classif, mutual_info_regression
)

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    QuantileTransformer, PowerTransformer, Normalizer
)
from sklearn import svm, linear_model, ensemble
from sklearn.calibration import CalibratedClassifierCV
import inspect


_SCALER_REGISTRY = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,           # good for chi2 (keeps features ≥ 0)
    "RobustScaler": RobustScaler,
    "MaxAbsScaler": MaxAbsScaler,           # good for sparse data
    "QuantileTransformer": QuantileTransformer,
    "PowerTransformer": PowerTransformer,   # box-cox needs strictly positive
    "Normalizer": Normalizer,               # ℓ2 normalize samples
    "passthrough": "passthrough",
    "none": "passthrough",
}

_CV_REGISTRY = {
    "StratifiedKFold": StratifiedKFold,
    "RepeatedStratifiedKFold": RepeatedStratifiedKFold,
    "StratifiedShuffleSplit": StratifiedShuffleSplit,
    "GroupKFold": GroupKFold,
    "StratifiedGroupKFold": StratifiedGroupKFold,
    "LeaveOneGroupOut": LeaveOneGroupOut,
    "LeavePGroupsOut": LeavePGroupsOut,
    "TimeSeriesSplit": TimeSeriesSplit,
    "PredefinedSplit": PredefinedSplit,   # usually built per-outer-fold
}

_ESTIMATOR_REGISTRY = {
    # SVMs
    "SVC": svm.SVC,                      # supports probability=True
    "LinearSVC": svm.LinearSVC,          # no predict_proba
    "SVR": svm.SVR,
    "LinearSVR": svm.LinearSVR,

    # Common classifiers
    "LogisticRegression": linear_model.LogisticRegression,
    "LogisticRegressionCV": linear_model.LogisticRegressionCV,
    "SGDClassifier": linear_model.SGDClassifier,
    "RidgeClassifier": linear_model.RidgeClassifier,
    "RandomForestClassifier": ensemble.RandomForestClassifier,

    # Common regressors
    "RandomForestRegressor": ensemble.RandomForestRegressor,
}

_SEARCH_REGISTRY = {
    "GridSearchCV": GridSearchCV,
    "RandomizedSearchCV": RandomizedSearchCV,
    "HalvingGridSearchCV": HalvingGridSearchCV,
    "HalvingRandomSearchCV": HalvingRandomSearchCV,
}

_FS_REGISTRY = {
    "SelectKBest": SelectKBest,
    "SelectPercentile": SelectPercentile,
    "SelectFromModel": SelectFromModel,
    "RFE": RFE,
    "RFECV": RFECV,
    "VarianceThreshold": VarianceThreshold,
    "passthrough": "passthrough",
    "none": "passthrough",
}

_SCORE_FUNCS = {
    "f_classif": f_classif,
    "f_regression": f_regression,
    "chi2": chi2,  # requires non-negative features
    "mutual_info_classif": mutual_info_classif,
    "mutual_info_regression": mutual_info_regression,
}

# Optional: simple aliases
_ALIASES = {
    "SVM": "SVC",           # "SVM" -> SVC by default
    "LinearSVM": "LinearSVC"
}

def scaler_from_config(cfg: dict | None):
    """
    cfg example:
      scaler:
        name: MinMaxScaler
        args: { feature_range: [0, 1] }
    """
    if not cfg:
        return StandardScaler()  # sensible default

    name = cfg.get("name", "StandardScaler")
    if name not in _SCALER_REGISTRY:
        raise ValueError(f"Unknown scaler '{name}'. Allowed: {sorted(_SCALER_REGISTRY)}")
    if _SCALER_REGISTRY[name] == "passthrough":
        return "passthrough"

    cls = _SCALER_REGISTRY[name]
    args = dict(cfg.get("args", {}))

    # fail fast on unexpected kwargs
    sig = inspect.signature(cls.__init__)
    extra = set(args) - set(sig.parameters)
    if extra:
        raise TypeError(f"{name} got unexpected args: {sorted(extra)}")

    return cls(**args)

def cv_from_config(cfg):
    """cfg like {'name': 'StratifiedKFold', 'args': {'n_splits': 3, 'shuffle': False}}"""
    name = cfg["name"]
    args = dict(cfg.get("args", {}))
    if name not in _CV_REGISTRY:
        raise ValueError(f"Unknown CV splitter: {name}")
    cls = _CV_REGISTRY[name]

    # filter unknown kwargs to fail-fast with a clear error message
    sig = inspect.signature(cls.__init__)
    extra = set(args) - set(sig.parameters)
    if extra:
        raise TypeError(f"{name} got unexpected args: {sorted(extra)}")

    return cls(**args)

def estimator_from_config(cfg, *, task=None, random_state=None):
    """
    cfg example:
      estimator:
        name: SVC
        args: { kernel: linear, C: 1.0, class_weight: balanced, probability: false }
        calibrate: { enable: true, method: sigmoid, cv: 3 }   # optional (classification only)
    """
    name = cfg["name"]
    name = _ALIASES.get(name, name)

    if name not in _ESTIMATOR_REGISTRY:
        known = ", ".join(sorted(_ESTIMATOR_REGISTRY))
        raise ValueError(f"Unknown estimator '{name}'. Choose one of: {known}")

    cls = _ESTIMATOR_REGISTRY[name]
    args = dict(cfg.get("args", {}))

    # Fail fast on unexpected kwargs
    sig = inspect.signature(cls.__init__)
    extra = set(args) - set(sig.parameters)
    if extra:
        raise TypeError(f"{name} got unexpected args: {sorted(extra)}")

    # Inject random_state if supported and not provided
    if random_state is not None and "random_state" in sig.parameters and "random_state" not in args:
        args["random_state"] = random_state

    est = cls(**args)

    # Optional calibration for classifiers that lack predict_proba (e.g., LinearSVC)
    cal_cfg = cfg.get("calibrate", {})
    if cal_cfg.get("enable", False):
        if task and task.lower().startswith("regress"):
            raise ValueError("Calibration is classification-only.")
        has_proba = hasattr(est, "predict_proba")
        if not has_proba:
            method = cal_cfg.get("method", "sigmoid")   # 'sigmoid' or 'isotonic'
            cv = cal_cfg.get("cv", 3)
            est = CalibratedClassifierCV(estimator=est, method=method, cv=cv)
        # If the estimator already has predict_proba (e.g., SVC with probability=True), do nothing.

    return est

def search_from_config(estimator, cv, cfg):
    """
    cfg example:
      gridsearch:
        name: GridSearchCV
        args:
          param_grid: {...}
          scoring: balanced_accuracy
          n_jobs: 1
          refit: true
          error_score: raise
    """
    name = cfg.get("name", "GridSearchCV")
    args = dict(cfg.get("args", {}))
    if name not in _SEARCH_REGISTRY:
        raise ValueError(f"Unknown search '{name}'. Allowed: {sorted(_SEARCH_REGISTRY)}")
    cls = _SEARCH_REGISTRY[name]

    # Validate args against the class signature; reserve estimator/cv for our call
    sig = inspect.signature(cls.__init__)
    extra = set(args) - set(sig.parameters)
    if extra:
        raise TypeError(f"{name} got unexpected args: {sorted(extra)}")
    args.pop("estimator", None)
    args.pop("cv", None)

    # Light validation for the right parameter key
    if cls in (GridSearchCV, HalvingGridSearchCV) and "param_grid" not in args:
        raise ValueError(f"{name} requires 'param_grid'.")
    if cls in (RandomizedSearchCV, HalvingRandomSearchCV) and "param_distributions" not in args:
        raise ValueError(f"{name} requires 'param_distributions'.")

    return cls(estimator=estimator, cv=cv, **args)


def selector_from_config(cfg, estimator_factory=None, *, task=None, random_state=None):
    """
    cfg example:
      feature_selection:
        name: SelectKBest
        args:
          k: 300
          score_func: f_classif

    For SelectFromModel/RFE/RFECV, include:
        args:
          estimator: { name: LinearSVC, args: { C: 1.0, dual: false } }
    """
    name = cfg.get("name", "SelectKBest")
    if name not in _FS_REGISTRY:
        raise ValueError(f"Unknown feature selector '{name}'. Allowed: {sorted(_FS_REGISTRY)}")
    if _FS_REGISTRY[name] == "passthrough":
        return "passthrough"

    cls = _FS_REGISTRY[name]
    args = dict(cfg.get("args", {}))

    # Map score_func strings to callables
    if "score_func" in args and isinstance(args["score_func"], str):
        sf = args["score_func"]
        if sf not in _SCORE_FUNCS:
            raise ValueError(f"Unknown score_func '{sf}'. Allowed: {sorted(_SCORE_FUNCS)}")
        args["score_func"] = _SCORE_FUNCS[sf]

    # Inject estimator for wrappers that require it
    if name in ("SelectFromModel", "RFE", "RFECV"):
        est_cfg = args.pop("estimator", None)
        if est_cfg is None:
            raise ValueError(f"{name} requires an 'estimator' in args")
        if estimator_factory is None:
            raise ValueError(f"Provide estimator_factory to build the inner estimator for {name}")
        inner_est = estimator_factory(est_cfg, task=task, random_state=random_state)
        args["estimator"] = inner_est

    # Fail fast on unexpected kwargs
    sig = inspect.signature(cls.__init__)
    extra = set(args) - set(sig.parameters)
    if extra:
        raise TypeError(f"{name} got unexpected args: {sorted(extra)}")

    return cls(**args)