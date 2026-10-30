"""Exercise production CV functions without requiring image-processing packages."""
import ast
import inspect
import logging
from pathlib import Path
import types
import unittest
from typing import Any, Optional

import numpy as np
from sklearn import model_selection
from sklearn.metrics import get_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils.validation import has_fit_parameter
from sklearn.exceptions import ConvergenceWarning
import warnings

ROOT = Path(__file__).resolve().parents[1]

def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names), names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)

CV_NAMES = ['StratifiedKFold', 'RepeatedStratifiedKFold', 'KFold',
            'GroupKFold', 'StratifiedGroupKFold', 'LeaveOneGroupOut', 'LeavePGroupsOut']
factory_ns = dict(inspect=inspect, _CV_REGISTRY={name: getattr(model_selection, name) for name in CV_NAMES},
                  **{name: getattr(model_selection, name) for name in CV_NAMES})
load_functions(ROOT / 'panic/factory.py',
               ['cv_from_config', '_splitter_accepts_groups', '_splitter_requires_groups'], factory_ns)
factory = types.SimpleNamespace(cv_from_config=factory_ns['cv_from_config'],
    _splitter_accepts_groups=factory_ns['_splitter_accepts_groups'],
    _splitter_requires_groups=factory_ns['_splitter_requires_groups'], scorer_from_config=get_scorer)
ns = dict(np=np, factory=factory, logger=logging.getLogger(__name__), Any=Any, Optional=Optional,
    FoldScoreFunction=object, errors=types.SimpleNamespace(NoFeaturesSelectedError=ValueError),
    BaseSearchCV=BaseSearchCV, has_fit_parameter=has_fit_parameter,
    warnings=warnings, ConvergenceWarning=ConvergenceWarning)
load_functions(ROOT / 'panic/pipeline.py', ['_validate_stratified_counts', '_stratified_outer_cv',
    '_permute_within_groups', 'create_outer_folds', 'cv_mean_function_score', 'sklearn_pipeline_score'], ns)

class StratifiedCVTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([0]*16 + [1]*6)
        self.X = np.random.default_rng(9).normal(size=(22, 4))
        self.cfg = {'outer_cv': {'mode': 'sklearn', 'name': 'StratifiedKFold',
            'args': {'n_splits': 3, 'shuffle': True, 'random_state': 42}}}

    def folds(self, y=None, cfg=None):
        return ns['create_outer_folds'](cfg or self.cfg, self.y if y is None else y)

    def test_optional_groups_are_not_required_for_stratified_splitter(self):
        cv = model_selection.StratifiedKFold(3)
        self.assertTrue(factory._splitter_accepts_groups(cv))
        self.assertFalse(factory._splitter_requires_groups(cv))
        # Reproduce the searchlight path after permutation groups are cleared.
        y = np.array([0]*8 + [1]*6)
        folds = ns['create_outer_folds'](self.cfg, y, groups=None)
        self.assertEqual(len(folds), 3)
        for train, test in folds:
            self.assertEqual(set(y[train]), {0,1})
            self.assertEqual(set(y[test]), {0,1})

    def test_group_splitters_still_require_real_groups(self):
        for name, args in [('GroupKFold', {'n_splits':2}),
                           ('StratifiedGroupKFold', {'n_splits':2}),
                           ('LeaveOneGroupOut', {}), ('LeavePGroupsOut', {'n_groups':1})]:
            cfg = {'outer_cv': {'mode':'sklearn', 'name':name, 'args':args}}
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'requires `groups`'):
                    self.folds(cfg=cfg)
                groups = np.tile(np.arange(2),11)
                folds = ns['create_outer_folds'](cfg, self.y, groups=groups)
                self.assertTrue(folds)
                for train, test in folds:
                    self.assertFalse(set(groups[train]) & set(groups[test]))

    def test_required_custom_groups_parameter_is_detected(self):
        class Required:
            def split(self, X, y, groups):
                pass
        class OptionalGroups:
            def split(self, X, y, groups=None):
                pass
        self.assertTrue(factory._splitter_requires_groups(Required()))
        self.assertFalse(factory._splitter_requires_groups(OptionalGroups()))

    def test_barcelona_folds_have_two_minority_test_trials(self):
        folds = self.folds()
        np.testing.assert_array_equal(np.sort(np.concatenate([te for _, te in folds])), np.arange(22))
        for tr, te in folds:
            self.assertEqual(int(self.y[te].sum()), 2)
            self.assertEqual(int(self.y[tr].sum()), 4)
            self.assertFalse(set(tr) & set(te))

    def test_rebuilds_from_each_permuted_vector_and_is_repeatable(self):
        def run(seed):
            seen = []
            def score(X, y, tr, te, **kw):
                seen.append((y.copy(), tr.copy(), te.copy()))
                self.assertEqual(int(y[te].sum()), 2)
                return float(X[te, 0].mean())
            result = ns['cv_mean_function_score'](self.X, self.y, self.folds(), score,
                cv_config=self.cfg, permute=True, rng=np.random.default_rng(seed))
            return result, seen
        for seed in range(100):
            result, seen = run(seed)
            expected = self.folds(seen[0][0])
            for (y, tr, te), (etr, ete) in zip(seen, expected):
                np.testing.assert_array_equal(y, seen[0][0])
                np.testing.assert_array_equal(tr, etr)
                np.testing.assert_array_equal(te, ete)
            self.assertEqual(result, run(seed)[0])
        self.assertTrue(any(not np.array_equal(te, oldte) for (_, _, te), (_, oldte) in zip(seen, self.folds())))

    def test_group_permutations_preserve_counts_within_real_blocks(self):
        groups = np.tile(np.arange(2), 11)
        def score(X, y, tr, te, **kw):
            for group in [0, 1]:
                self.assertEqual(int(y[groups == group].sum()), int(self.y[groups == group].sum()))
            return .5
        ns['cv_mean_function_score'](self.X, self.y, self.folds(), score, cv_config=self.cfg,
            groups=groups, permute=True, rng=np.random.default_rng(6))

    def test_unsupported_class_counts_fail_before_scoring(self):
        for y in (np.array([0]*10+[1]*2), np.zeros(12)):
            with self.assertRaisesRegex(ValueError, 'at least 3 trials'):
                self.folds(y)

    def test_randomized_splitters_require_fixed_integer_seed(self):
        self.cfg['outer_cv']['args'].pop('random_state')
        with self.assertRaisesRegex(ValueError, 'integer random_state'):
            self.folds()

    def test_repeated_stratified_folds(self):
        self.cfg['outer_cv']['name'] = 'RepeatedStratifiedKFold'
        self.cfg['outer_cv']['args'].pop('shuffle')
        self.cfg['outer_cv']['args']['n_repeats'] = 2
        self.assertEqual(len(self.folds()), 6)

    def test_fixed_design_is_not_rebuilt(self):
        cfg = {'outer_cv': {'mode': 'nth_trial', 'args': {'fold_interval': 6}}}
        folds = self.folds(cfg=cfg)
        seen = []
        def score(X, y, tr, te, **kw):
            seen.append(te)
            return .5
        ns['cv_mean_function_score'](self.X, self.y, folds, score, cv_config=cfg,
            permute=True, rng=np.random.default_rng(3))
        for got, (_, expected) in zip(seen, folds):
            np.testing.assert_array_equal(got, expected)

    def test_real_svc_observed_and_permuted_scores(self):
        ns['pipeline_from_config'] = lambda cfg, **kw: make_pipeline(StandardScaler(), SVC(kernel='linear'))
        cfg = {**self.cfg, 'scoring': 'balanced_accuracy'}
        for permute in [False, True]:
            value = ns['cv_mean_function_score'](self.X, self.y, self.folds(),
                ns['sklearn_pipeline_score'], cfg=cfg, permute=permute, rng=np.random.default_rng(4))
            self.assertTrue(0 <= value <= 1)

    def test_inner_cv_rejects_five_folds_on_four_minority_training_trials(self):
        def pipeline(cfg, **kw):
            return model_selection.GridSearchCV(SVC(kernel='linear'), {'C': [1]},
                cv=model_selection.StratifiedKFold(5), scoring='balanced_accuracy')
        ns['pipeline_from_config'] = pipeline
        tr, te = self.folds()[0]
        with self.assertRaisesRegex(ValueError, 'inner_cv.*at least 5 trials'):
            ns['sklearn_pipeline_score'](self.X, self.y, tr, te, cfg={'scoring':'balanced_accuracy'})

    def test_all_builtin_cv_plugins_pass_configuration_and_group_policy(self):
        tree = ast.parse((ROOT / 'panic/plugins/decoding.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == 'cv_mean_function_score']
        self.assertEqual(len(calls), 4)
        for call in calls:
            self.assertTrue({'cv_config', 'permute_within_groups'} <= {k.arg for k in call.keywords})

if __name__ == '__main__':
    unittest.main()
