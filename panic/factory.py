# cv_factory.py
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    GroupKFold,
    StratifiedGroupKFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    TimeSeriesSplit,
    PredefinedSplit,
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV
)

from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    SelectFromModel,
    RFE,
    RFECV,
    VarianceThreshold,
    f_classif,
    f_regression,
    chi2,
    mutual_info_classif,
    mutual_info_regression
)

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    QuantileTransformer,
    PowerTransformer,
    Normalizer
)

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)

from sklearn.naive_bayes import (
    GaussianNB,
    BernoulliNB,
    MultinomialNB
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier, 
    BaggingClassifier
)

from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score
)

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier

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

    # Linear / generalized linear classifiers
    "LogisticRegression": linear_model.LogisticRegression,
    "LogisticRegressionCV": linear_model.LogisticRegressionCV,
    "RidgeClassifier": linear_model.RidgeClassifier,
    "SGDClassifier": linear_model.SGDClassifier,
    "Perceptron": linear_model.Perceptron,
    "PassiveAggressiveClassifier": linear_model.PassiveAggressiveClassifier,

    # Discriminant analysis
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,

    # Naive Bayes
    "GaussianNB": GaussianNB,
    "BernoulliNB": BernoulliNB,
    "MultinomialNB": MultinomialNB,

    # Neighbors
    "KNeighborsClassifier": KNeighborsClassifier,
    "NearestCentroid": NearestCentroid,

    # Trees & ensembles
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": ensemble.RandomForestClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
    "BaggingClassifier": BaggingClassifier,

    # Gaussian Process
    "GaussianProcessClassifier": GaussianProcessClassifier,

    # Neural nets
    "MLPClassifier": MLPClassifier,

    # Regressors you already expose
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
    "PCA": PCA,
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
    "SVM": "SVC",
    "LinearSVM": "LinearSVC",
    "LR": "LogisticRegression",
    "LDA": "LinearDiscriminantAnalysis",
    "QDA": "QuadraticDiscriminantAnalysis",
    "RF": "RandomForestClassifier",
    "ET": "ExtraTreesClassifier",
    "GPC": "GaussianProcessClassifier",
}

_SCORER_ALIASES = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "roc_auc_ovr": "roc_auc_ovr",
    "roc_auc_ovo": "roc_auc_ovo",
    "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
    "roc_auc_ovo_weighted": "roc_auc_ovo_weighted",
    "neg_log_loss": "neg_log_loss",
}

def scorer_from_config(sc_cfg, *, n_classes=None):
    """
    Accepts either a string ("balanced_accuracy", "roc_auc_ovr", ...) or
    a dict like {"name": "f1_macro"} and returns a scorer suitable for sklearn CV.

    Supported (recommended) names:
      - "accuracy", "balanced_accuracy"
      - "roc_auc_ovr", "roc_auc_ovo", "..._weighted"
      - "f1_macro"
      - "mcc" (Matthews corrcoef), "cohen_kappa"
      - "neg_log_loss" (requires predict_proba or decision_function)

    Returns: either a string scorer name or a callable from make_scorer(...).
    """
    if not sc_cfg:
        return "balanced_accuracy"

    # Allow plain string
    if isinstance(sc_cfg, str):
        sc_name = sc_cfg
    else:
        sc_name = sc_cfg.get("name", "balanced_accuracy")

    sc_name = sc_name.lower()

    # 1) direct pass-through (strings sklearn already understands)
    if sc_name in _SCORER_ALIASES:
        return _SCORER_ALIASES[sc_name]

    # 2) build a scorer with make_scorer for the classic functions
    if sc_name == "f1_macro":
        return make_scorer(f1_score, average="macro")
    if sc_name == "mcc":
        return make_scorer(matthews_corrcoef)
    if sc_name in ("cohen_kappa", "cohen_kappa_score", "kappa"):
        return make_scorer(cohen_kappa_score)

    if sc_name in ("accuracy_score", "accuracy"):
        return make_scorer(accuracy_score)
    if sc_name in ("balanced_accuracy_score", "balanced_accuracy"):
        return make_scorer(balanced_accuracy_score)

    raise ValueError(
        f"Unknown scoring '{sc_name}'. Try one of: "
        f"{sorted(set(_SCORER_ALIASES) | {'f1_macro','mcc','cohen_kappa'})}"
    )

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
    """Construct a cross-validation splitter from configuration.

    Parameters
    ----------
    cfg : dict
        Cross-validation configuration with the structure::

            {
                "name": "StratifiedKFold",
                "args": {
                    "n_splits": 5,
                    "shuffle": False,
                },
            }

        ``name`` must correspond to an entry in ``_CV_REGISTRY``.
        ``args`` are forwarded to the splitter constructor after validation.

    Returns
    -------
    sklearn.model_selection.BaseCrossValidator
        Instantiated cross-validation splitter.

    Raises
    ------
    TypeError
        If ``cfg`` is not a dictionary or contains constructor arguments
        unsupported by the selected splitter.
    ValueError
        If the requested splitter is not registered.
    KeyError
        If ``name`` is missing from the configuration.
    """
    if not isinstance(cfg, dict):
        raise TypeError(
            f"`cfg` must be a dictionary, got {type(cfg).__name__}."
        )

    name = cfg["name"]
    args = dict(cfg.get("args", {}))

    if name not in _CV_REGISTRY:
        raise ValueError(
            f"Unknown CV splitter {name!r}. "
            f"Available splitters: {sorted(_CV_REGISTRY)}"
        )

    cls = _CV_REGISTRY[name]

    # Validate constructor arguments before instantiation.
    sig = inspect.signature(cls.__init__)
    parameters = sig.parameters

    accepts_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )

    if not accepts_kwargs:
        valid_args = {
            key
            for key, param in parameters.items()
            if key != "self"
            and param.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }

        extra = set(args) - valid_args

        if extra:
            raise TypeError(
                f"{name} got unexpected args: {sorted(extra)}"
            )

    return cls(**args)


def _scorer_needs_proba(scoring: str | object) -> bool:
    # Strings that sklearn handles natively and need proba/decision_function
    NEEDS = {
        "roc_auc", "roc_auc_ovr", "roc_auc_ovo",
        "roc_auc_ovr_weighted", "roc_auc_ovo_weighted",
        "neg_log_loss"
    }
    if isinstance(scoring, str):
        return scoring in NEEDS
    # If you pass a callable make_scorer, assume no proba unless you know otherwise
    return False

def estimator_from_config(cfg, *, random_state=None, labels=None, scoring=None):

    name = cfg["name"]
    name = _ALIASES.get(name, name)
    if name not in _ESTIMATOR_REGISTRY:
        known = ", ".join(sorted(_ESTIMATOR_REGISTRY))
        raise ValueError(f"Unknown estimator '{name}'. Choose one of: {known}")

    cls  = _ESTIMATOR_REGISTRY[name]
    args = dict(cfg.get("args", {}))

    # fail fast on unknown kwargs (matches your current style)
    sig = inspect.signature(cls.__init__)
    extra = set(args) - set(sig.parameters)
    if extra:
        raise TypeError(f"{name} got unexpected args: {sorted(extra)}")

    # Inject random_state if supported
    if random_state is not None and "random_state" in sig.parameters and "random_state" not in args:
        args["random_state"] = random_state

    # Label-aware conveniences (no data leakage)
    n_classes = None
    if labels is not None:
        labs = np.asarray(labels)
        n_classes = int(np.unique(labs).size)

        # 2a) class_weight='auto' → compute weights from label frequencies
        if args.get("class_weight", None) == "auto":
            # inverse frequency weights
            _, counts = np.unique(labs, return_counts=True)
            weights = {cls_id: (len(labs) / (n_classes * cnt)) for cls_id, cnt in zip(np.unique(labs), counts)}
            args["class_weight"] = weights

        # 2b) LDA safe default in p >> n
        if name == "LinearDiscriminantAnalysis":
            if "solver" not in args and "shrinkage" not in args:
                args["solver"] = "lsqr"
                args["shrinkage"] = "auto"

    # Build the estimator
    est = cls(**args)

    # Probability plumbing based on scoring
    cal_cfg = cfg.get("calibrate", {}) or {}
    needs_proba = _scorer_needs_proba(scoring if scoring is not None else cfg.get("scoring", "balanced_accuracy"))
    
    # If scorer needs proba and estimator has none:
    #  - for SVC, prefer turning on probability=True (if not set)
    #  - else, fall back to CalibratedClassifierCV if user enabled it
    if needs_proba and not hasattr(est, "predict_proba"):
        if name == "SVC":
            # Sklearn SVC needs probability=True at construction time
            # rebuild with probability=True if user didn't specify
            if "probability" not in args or args.get("probability") is False:
                args["probability"] = True
                est = cls(**args)
        elif cal_cfg.get("enable", False):
            method = cal_cfg.get("method", "sigmoid")
            cv = cal_cfg.get("cv", 3)
            est = CalibratedClassifierCV(
                estimator=est,
                method=method,
                cv=cv
            )

    # If user explicitly asked for calibration, keep your existing behavior
    elif cal_cfg.get("enable", False) and not hasattr(est, "predict_proba"):
        method = cal_cfg.get("method", "sigmoid")
        cv     = cal_cfg.get("cv", 3)
        est    = CalibratedClassifierCV(estimator=est, method=method, cv=cv)

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

    if not cfg:
        return None
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

def _splitter_accepts_groups(cv):
    """Return whether ``cv.split`` accepts a groups argument."""
    signature = inspect.signature(cv.split)
    return "groups" in signature.parameters
