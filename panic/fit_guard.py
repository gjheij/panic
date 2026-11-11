"""Execution safeguards for searchlight SVC fits; never score failed convergence."""
from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC


class SearchlightFitError(RuntimeError):
    fatal_searchlight_fit = True

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


@dataclass
class CenterFitFailure:
    center: tuple
    diagnostics: dict


def fit_guard_settings(parallel):
    value = parallel.get('svc_fit_guard', {})
    if not isinstance(value, dict) or set(value)-{'enabled','max_iter','on_failure'}:
        raise ValueError('parallel.svc_fit_guard supports enabled, max_iter and on_failure')
    result = dict(enabled=True, max_iter=100000, on_failure='mark_center')
    result.update(value)
    if type(result['enabled']) is not bool:
        raise ValueError('svc_fit_guard.enabled must be boolean')
    if type(result['max_iter']) is not int or result['max_iter'] < 1:
        raise ValueError('svc_fit_guard.max_iter must be a positive integer')
    if result['on_failure'] not in ('mark_center','raise'):
        raise ValueError('svc_fit_guard.on_failure must be mark_center or raise')
    return result


def guarded_fit(model, X, y, policy, **fit_kwargs):
    """Bound previously unlimited SVC solvers; reject warnings before scoring."""
    cap = policy['max_iter']
    params = model.get_params(deep=True)
    solvers = [('', model)] if isinstance(model, SVC) else []
    solvers.extend((name, value) for name, value in params.items() if isinstance(value, SVC))
    changes = {}
    for name, solver in solvers:
        if solver.max_iter == -1:
            changes[(name+'__' if name else '')+'max_iter'] = cap
    # Search grids could otherwise override the cap after cloning candidates.
    grids = getattr(model, 'param_grid', None)
    if grids is not None:
        for grid in ([grids] if isinstance(grids, dict) else grids):
            if any(key.endswith('max_iter') for key in grid):
                raise SearchlightFitError('Searchlight fit guard cannot bound a grid that tunes max_iter',
                                         {'reason_code':'unsupported_fit_guard_config'})
    if changes:
        model.set_params(**changes)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', ConvergenceWarning)
        fitted = model.fit(X, y, **fit_kwargs)
    failed = [str(item.message) for item in captured if issubclass(item.category, ConvergenceWarning)]
    final = getattr(fitted, 'best_estimator_', fitted)
    estimators = [final, *final.get_params(deep=True).values()]
    failed_status = any(getattr(value, 'fit_status_', 0) != 0 for value in estimators if isinstance(value, SVC))
    if failed or failed_status:
        array = np.asarray(X)
        finite = array[np.isfinite(array)]
        diagnostics = dict(reason_code='solver_nonconvergence', max_iter_guard=cap,
            bounded_parameters=changes, shape=list(array.shape), n_nonfinite=int(array.size-finite.size),
            solvers=[dict(path=name or 'estimator', kernel=str(solver.kernel), C=float(solver.C),
                          tol=float(solver.tol), max_iter=int(solver.max_iter)) for name,solver in solvers],
            minimum=float(finite.min()) if finite.size else None,
            maximum=float(finite.max()) if finite.size else None,
            class_counts={str(label):int((np.asarray(y)==label).sum()) for label in np.unique(y)},
            warnings=failed, fit_status_failure=bool(failed_status))
        raise SearchlightFitError('Searchlight estimator did not converge; center will not receive a score', diagnostics)
    for item in captured:
        if not issubclass(item.category, ConvergenceWarning):
            warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
    return fitted
