"""Production selector/factory/pipeline tests without image-stack imports."""
import ast
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import Any, Optional
import unittest
import warnings

import numpy as np
from joblib import Parallel, delayed, dump, load
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import has_fit_parameter
from sklearn.experimental import enable_halving_search_cv  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
# Import the actual leaf module without PANIC's image-stack __init__. This path
# also lets loky workers import the selector normally during serialization tests.
sys.path.insert(0, str(ROOT/'panic'))
selector_module = importlib.import_module('feature_selection')
Selector = selector_module.SelectPercentileMinFeatures

factory_tree = ast.parse((ROOT/'panic/factory.py').read_text())
factory_tree.body = [node for node in factory_tree.body if not (
    isinstance(node, ast.ImportFrom) and node.module == 'panic.feature_selection')]
factory_ns = {'SelectPercentileMinFeatures':Selector}
exec(compile(factory_tree, str(ROOT/'panic/factory.py'), 'exec'), factory_ns)
factory = types.SimpleNamespace(**factory_ns)

class NoFeaturesSelectedError(ValueError):
    pass

pipeline_ns = dict(np=np, os=os, json=json, logger=logging.getLogger(__name__), factory=factory,
    errors=types.SimpleNamespace(NoFeaturesSelectedError=NoFeaturesSelectedError),
    Pipeline=Pipeline, VarianceThreshold=VarianceThreshold, BaseEstimator=BaseEstimator,
    TransformerMixin=TransformerMixin, BaseSearchCV=BaseSearchCV, has_fit_parameter=has_fit_parameter,
    warnings=warnings, ConvergenceWarning=ConvergenceWarning, Any=Any, Optional=Optional,
    FoldScoreFunction=object)
names = {'FailIfNoFeatures','pipeline_from_config','sklearn_pipeline_score','cv_mean_function_score',
         'create_outer_folds','_validate_stratified_counts','_stratified_outer_cv','_permute_within_groups'}
tree=ast.parse((ROOT/'panic/pipeline.py').read_text())
nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
assert len(nodes)==len(names)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/'panic/pipeline.py'),'exec'),pipeline_ns)


def ranked_scores(X,y):
    return np.arange(X.shape[1],dtype=float), np.ones(X.shape[1])


def nan_scores(X,y):
    return np.full(X.shape[1],np.nan), np.ones(X.shape[1])


def tied_scores(X,y):
    return np.ones(X.shape[1]), np.ones(X.shape[1])


def fit_one(selector,X,y):
    return clone(selector).fit_transform(X,y).shape[1]


class MinFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.y=np.array([0]*8+[1]*6)
        self.X=np.random.default_rng(24).normal(size=(14,1))

    def test_one_voxel_keeps_one_at_every_positive_grid_percentile(self):
        for percentile in [1,10,20,40,100]:
            with self.subTest(percentile=percentile):
                fitted=Selector(percentile=percentile).fit(self.X,self.y)
                self.assertEqual(fitted.transform(self.X).shape,(14,1))

    def test_larger_masks_match_sklearn_when_floor_is_not_needed(self):
        X=np.random.default_rng(44).normal(size=(30,100))
        y=np.tile([0,1],15)
        for percentile in [10,20,40,100]:
            normal=SelectPercentile(percentile=percentile).fit(X,y)
            guarded=Selector(percentile=percentile).fit(X,y)
            np.testing.assert_array_equal(guarded.get_support(),normal.get_support())

    def test_fallback_selects_highest_training_score_and_breaks_ties_stably(self):
        X=np.tile(np.arange(4),(14,1))
        np.testing.assert_array_equal(Selector(ranked_scores,percentile=1).fit(X,self.y).get_support(),[False,False,False,True])
        np.testing.assert_array_equal(Selector(tied_scores,percentile=1).fit(X,self.y).get_support(),[True,False,False,False])
        self.assertEqual(Selector(ranked_scores,percentile=1,min_features=2).fit(X,self.y).get_support().sum(),2)

    def test_zero_percentile_and_zero_minimum_preserve_explicit_empty_selection(self):
        self.assertEqual(Selector(percentile=0).fit(self.X,self.y).get_support().sum(),0)
        self.assertEqual(Selector(percentile=10,min_features=0).fit(self.X,self.y).get_support().sum(),0)

    def test_all_nan_scores_do_not_invent_a_ranked_feature(self):
        self.assertEqual(Selector(nan_scores,percentile=10).fit(self.X,self.y).get_support().sum(),0)

    def test_invalid_minimum_and_empty_inputs_raise(self):
        for value in [-1,True,1.5,None]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                Selector(min_features=value).fit(self.X,self.y)
        with self.assertRaises(ValueError):
            Selector().fit(np.empty((14,0)),self.y)

    def test_factory_existing_name_defaults_to_floor_and_supports_opt_out(self):
        selector=factory.selector_from_config({'name':'SelectPercentile','args':{'percentile':10,'score_func':'f_classif'}})
        self.assertIsInstance(selector,Selector)
        self.assertEqual(selector.min_features,1)
        legacy=factory.selector_from_config({'name':'SelectPercentile','args':{'percentile':10,'min_features':0}})
        self.assertEqual(legacy.fit(self.X,self.y).get_support().sum(),0)

    def test_selector_is_cloneable_serializable_and_runs_in_loky_workers(self):
        selector=Selector(percentile=10)
        self.assertEqual(clone(selector).get_params(),selector.get_params())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'selector.joblib'
            dump(selector.fit(self.X,self.y),path)
            self.assertEqual(load(path).transform(self.X).shape[1],1)
        self.assertEqual(Parallel(n_jobs=2)(delayed(fit_one)(selector,self.X,self.y) for _ in range(2)),[1,1])

    def config(self):
        return {'outer_cv':{'mode':'sklearn','name':'StratifiedKFold','args':{'n_splits':3,'shuffle':True,'random_state':42}},
                'inner_cv':{'name':'StratifiedKFold','args':{'n_splits':3,'shuffle':True,'random_state':42}},
                'estimator':{'name':'SVC','args':{'kernel':'linear','class_weight':'balanced'}},
                'feature_selection':{'name':'SelectPercentile','args':{'percentile':10,'score_func':'f_classif'}},
                'gridsearch':{'name':'GridSearchCV','args':{'param_grid':{'select__percentile':[10,20,40,100],'clf__C':[.1,1]},'error_score':'raise','n_jobs':1}},
                'scoring':'balanced_accuracy'}

    def test_actual_factory_and_nested_pipeline_fit_single_voxel_with_full_grid(self):
        cfg=self.config()
        folds=pipeline_ns['create_outer_folds'](cfg,self.y)
        for permute in [False,True]:
            score=pipeline_ns['cv_mean_function_score'](self.X,self.y,folds,pipeline_ns['sklearn_pipeline_score'],
                cfg=cfg,permute=permute,rng=np.random.default_rng(8))
            self.assertTrue(np.isfinite(score))
            self.assertTrue(0<=score<=1)
        pipe=pipeline_ns['pipeline_from_config'](cfg,labels=self.y)
        pipe.fit(self.X,self.y)
        self.assertEqual(pipe.best_estimator_.named_steps['select'].get_support().sum(),1)
        self.assertTrue(np.all(np.isfinite(pipe.cv_results_['mean_test_score'])))

    def test_upstream_variance_filter_still_rejects_all_constant_features(self):
        cfg=self.config()
        pipe=pipeline_ns['pipeline_from_config'](cfg,labels=self.y)
        with self.assertRaisesRegex(ValueError,'No feature'):
            pipe.fit(np.ones((14,1)),self.y)

if __name__=='__main__':
    unittest.main()
