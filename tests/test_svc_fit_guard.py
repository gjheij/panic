"""Real sklearn solver checks and synthetic searchlight/checkpoint recovery."""
import ast
import copy
import csv
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

ROOT=Path(__file__).resolve().parents[1]
PACKAGE='panic_guard_test'
pkg=types.ModuleType(PACKAGE);pkg.__path__=[str(ROOT/'panic')];sys.modules[PACKAGE]=pkg
fg=importlib.import_module(PACKAGE+'.fit_guard')
cp=importlib.import_module(PACKAGE+'.checkpoint')
sp=importlib.import_module(PACKAGE+'.searchlight_progress')


def toy_plugin(X,labels,*,permute=False,**kwargs):
    if permute and X.shape[1]==3 and X[0,0]==0:
        raise fg.SearchlightFitError('synthetic nonconvergence',{'reason_code':'solver_nonconvergence'})
    return .5 if permute else .75


def always_fail_plugin(*args,**kwargs):
    raise fg.SearchlightFitError('failure')


def parallel_guard_probe(seed):
    import fit_guard
    rng=np.random.default_rng(seed);X=rng.normal(size=(30,5));y=np.array([0,1]*15)
    try:fit_guard.guarded_fit(SVC(kernel='linear',max_iter=1),X,y,fit_guard.fit_guard_settings({}))
    except fit_guard.SearchlightFitError as exc:return fit_guard.CenterFitFailure((seed,0,0),exc.diagnostics)
    raise AssertionError('Expected iteration exhaustion')


class FakeImage:
    def __init__(self,data):
        self.data=np.asarray(data);self.shape=self.data.shape;self.affine=np.eye(4)
        self.header=types.SimpleNamespace(get_zooms=lambda:(1.,1.,1.))
    def to_filename(self,path):
        with open(path,'wb') as stream:np.save(stream,self.data)
    def get_fdata(self,dtype=None):return self.data.astype(dtype or float)


class FakeMonitor:
    def __init__(self,*args,**kwargs):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def mark_completed(self,*args):pass


class SolverGuardTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(4)
        self.X=rng.normal(size=(40,8));self.y=np.array([0,1]*20)
        self.policy=fg.fit_guard_settings({})

    def test_default_is_bounded_and_marks_whole_center(self):
        self.assertEqual(self.policy,dict(enabled=True,max_iter=100000,on_failure='mark_center'))
        for value in [None,{'max_iter':0},{'max_iter':True},{'on_failure':'ignore'},{'extra':1}]:
            with self.subTest(value=value),self.assertRaises(ValueError):fg.fit_guard_settings({'svc_fit_guard':value})

    def test_converged_svc_score_and_other_parameters_are_preserved(self):
        model=make_pipeline(StandardScaler(),SVC(kernel='linear',C=.1))
        original=copy.deepcopy(model).fit(self.X,self.y)
        fg.guarded_fit(model,self.X,self.y,self.policy)
        self.assertEqual(model[-1].max_iter,100000)
        np.testing.assert_array_equal(original.predict(self.X),model.predict(self.X))
        self.assertEqual(model[-1].C,.1)

    def test_existing_finite_limit_is_preserved_and_nonconvergence_is_not_scored(self):
        model=SVC(kernel='linear',max_iter=1)
        with self.assertRaises(fg.SearchlightFitError) as caught:fg.guarded_fit(model,self.X,self.y,self.policy)
        self.assertEqual(model.max_iter,1)
        self.assertEqual(caught.exception.diagnostics['reason_code'],'solver_nonconvergence')
        self.assertTrue(caught.exception.diagnostics['warnings'])
        self.assertTrue(caught.exception.fatal_searchlight_fit)

    def test_status_failure_is_rejected_even_without_warning(self):
        model=SVC()
        def fit(X,y,**kwargs):model.fit_status_=1;return model
        with patch.object(model,'fit',side_effect=fit):
            with self.assertRaises(fg.SearchlightFitError):fg.guarded_fit(model,self.X,self.y,self.policy)

    def test_roi_pipeline_path_does_not_use_guard(self):
        import test_stratified_cv as cv
        cfg={'scoring':'balanced_accuracy'}
        clf=make_pipeline(StandardScaler(),SVC(kernel='linear',C=.1))
        with patch.dict(cv.ns,{'pipeline_from_config':lambda *a,**kw:clf}):
            value=cv.ns['sklearn_pipeline_score'](self.X,self.y,np.arange(30),np.arange(30,40),cfg=cfg)
        self.assertTrue(np.isfinite(value));self.assertEqual(clf[-1].max_iter,-1)

    def test_searchlight_pipeline_guard_is_used_before_score(self):
        import test_stratified_cv as cv
        clf=make_pipeline(StandardScaler(),SVC(kernel='linear',max_iter=1))
        cfg={'scoring':'balanced_accuracy','_searchlight_fit_guard':self.policy}
        with patch.dict(sys.modules,{'panic.fit_guard':fg}),patch.dict(cv.ns,{'pipeline_from_config':lambda *a,**kw:clf}):
            with self.assertRaises(fg.SearchlightFitError):cv.ns['sklearn_pipeline_score'](self.X,self.y,np.arange(30),np.arange(30,40),cfg=cfg)

    def test_failure_record_survives_real_loky_worker_transport(self):
        with patch.object(sys,'path',[str(ROOT/'panic'),*sys.path]):
            values=joblib.Parallel(n_jobs=2)(joblib.delayed(parallel_guard_probe)(i) for i in [0,1])
        self.assertEqual([tuple(value.center) for value in values],[(0,0,0),(1,0,0)])
        self.assertTrue(all(value.diagnostics['reason_code']=='solver_nonconvergence' for value in values))

    def test_search_grid_cannot_override_runtime_limit(self):
        from sklearn.model_selection import GridSearchCV
        model=GridSearchCV(SVC(),{'max_iter':[-1]})
        with patch.object(model,'fit') as fit:
            with self.assertRaises(fg.SearchlightFitError) as caught:fg.guarded_fit(model,self.X,self.y,self.policy)
        fit.assert_not_called()
        self.assertEqual(caught.exception.diagnostics['reason_code'],'unsupported_fit_guard_config')


class CheckpointGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT);self.addCleanup(self.tmp.cleanup)
        self.base=str(Path(self.tmp.name)/'searchlight');self.path=self.base+'_checkpoint.joblib'
        self.seeds=np.array([10,20,30],dtype=np.uint32)
        self.saved=((0,0,0),.8,.5,.3,.01,2,250,0,0,None)
        joblib.dump(dict(version=1,fingerprint='same',seeds=self.seeds,results=[self.saved,None,None]),self.path)

    def checkpoint(self):return cp.SearchlightCheckpoint(self.path,'same',np.array([99,99,99]),3)
    def failure(self):return fg.CenterFitFailure((1,0,0),{'reason_code':'solver_nonconvergence','phase':'permutation','permutation':26,'perm_seed':42,'n_features':3,'error':'cap reached'})

    def test_legacy_migration_retains_original_results_seeds_and_fingerprint(self):
        obj=self.checkpoint();obj.record_guard(fg.fit_guard_settings({}));obj.save()
        after=joblib.load(self.path)
        self.assertEqual(after['version'],2);self.assertEqual(after['fingerprint'],'same')
        self.assertEqual(after['results'][0],self.saved);np.testing.assert_array_equal(after['seeds'],self.seeds)
        self.assertTrue(after['fit_guard_history'][0]['legacy_fit_status_unverified'])

    def test_incompatible_inputs_still_rejected(self):
        with self.assertRaisesRegex(ValueError,'Incompatible'):cp.SearchlightCheckpoint(self.path,'different',self.seeds,3)
        with self.assertRaisesRegex(ValueError,'Incompatible'):cp.SearchlightCheckpoint(self.path,'same',self.seeds[:2],2)

    def test_whole_failed_center_is_nan_with_zero_permutations(self):
        obj=self.checkpoint();progress=sp.CenterProgress(obj.results,obj,fg.fit_guard_settings({}),self.base)
        progress.record(1,self.failure());progress.record(2,((2,0,0),.7,.5,.2,.02,2,250,0,0,None));progress.flush()
        rows=list(progress.output_rows())
        self.assertEqual(rows[0],self.saved);self.assertTrue(all(np.isnan(value) for value in rows[1][1:5]));self.assertEqual(rows[1][6],0)
        self.assertIsNone(obj.results[1]);self.assertEqual(progress.completed,3);self.assertEqual(progress.pending,[])
        self.assertTrue(Path(self.base+'_fit_failures.csv').is_file())
        with open(self.base+'_fit_failures.csv',newline='') as stream:row=next(csv.DictReader(stream))
        self.assertEqual(row['permutation'],'26');self.assertEqual(row['center_index'],'1')

    def test_strict_failure_saves_diagnostics_and_completed_centers_before_raising(self):
        obj=self.checkpoint();policy=fg.fit_guard_settings({'svc_fit_guard':{'on_failure':'raise'}})
        progress=sp.CenterProgress(obj.results,obj,policy,self.base)
        with self.assertRaises(fg.SearchlightFitError):progress.record(1,self.failure())
        after=joblib.load(self.path)
        self.assertEqual(after['results'][0],self.saved);self.assertIsNone(after['results'][1])
        self.assertIn('1',after['failures']);self.assertIn(1,progress.pending)

    def test_failed_checkpoint_can_be_explicitly_retried_with_strict_policy(self):
        obj=self.checkpoint();progress=sp.CenterProgress(obj.results,obj,fg.fit_guard_settings({}),self.base);progress.record(1,self.failure())
        loaded=self.checkpoint()
        strict=sp.CenterProgress(loaded.results,loaded,fg.fit_guard_settings({'svc_fit_guard':{'on_failure':'raise'}}),self.base)
        self.assertIn(1,strict.pending)
        strict.record(1,((1,0,0),.6,.5,.1,.1,3,250,0,0,None));strict.flush()
        self.assertNotIn('1',loaded.failures);self.assertNotIn(1,strict.pending)

    def test_time_based_save_preserves_tail_before_count_interval(self):
        obj=self.checkpoint();progress=sp.CenterProgress(obj.results,obj,fg.fit_guard_settings({}),self.base,update_interval=1000,save_seconds=60)
        progress.last_save-=61
        progress.record(2,((2,0,0),.7,.5,.2,.02,2,250,0,0,None))
        self.assertIsNotNone(joblib.load(self.path)['results'][2])

    def test_failed_atomic_write_preserves_previous_checkpoint(self):
        obj=self.checkpoint();before=Path(self.path).read_bytes()
        with patch.object(cp.os,'replace',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):obj.save()
        self.assertEqual(Path(self.path).read_bytes(),before)


class SearchlightIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT);self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name)
        self.X=np.arange(16,dtype=np.float32).reshape(4,4);self.y=np.array([0,1,0,1]);self.ref=FakeImage(np.ones((4,1,1)))
        self.folds=[(np.array([0,1]),np.array([2,3]))]
        self.cfg={'parallel':{'n_jobs':1,'update_interval':1000},'searchlight':{'radius_mm':1,'checkpoint':True,'basepath':'sl'},'n_permutations':2,'analysis':{'type':'scikit_decoding'}}
        self.maps={}
        def save(base,**kwargs):self.maps.update(kwargs);return {}
        mf=types.SimpleNamespace(X=self.X,labels=self.y,fov_mask=self.ref,roi_linidx=np.arange(4),roi_metrics={})
        self.ns=dict(np=np,os=os,opj=os.path.join,json=json,csv=csv,tempfile=tempfile,time=importlib.import_module('time'),uuid=importlib.import_module('uuid'),
            dump=joblib.dump,joblib_hash=joblib.hash,SearchlightCheckpoint=cp.SearchlightCheckpoint,CenterProgress=sp.CenterProgress,
            CenterFitFailure=fg.CenterFitFailure,SearchlightFitError=fg.SearchlightFitError,fit_guard_settings=fg.fit_guard_settings,
            data=types.SimpleNamespace(MaskAndFilterBetas=lambda *a,**kw:mf),logger=logging.getLogger(__name__),
            core=types.SimpleNamespace(get_analysis_plugin=lambda *a,**kw:(toy_plugin,{}),unpack_plugin_result=lambda value:(value,{})),
            create_outer_folds=lambda *a,**kw:self.folds,cv_permutation_policy=lambda cfg:'test',
            install_faulthandler=lambda:None,worker_task_started=lambda:1,worker_heartbeat=lambda *a,**kw:None,
            worker_mark_completed=lambda *a,**kw:None,maybe_log_worker_resources=lambda *a,**kw:None,
            load_feature_matrix=lambda path,cols=None:joblib.load(path,mmap_mode='r')[:,cols],
            SearchlightMonitor=FakeMonitor,tqdm=lambda iterator,**kw:iterator,tqdm_disabled=lambda:True,
            image=types.SimpleNamespace(new_img_like=lambda ref,data,**kw:FakeImage(data)),save_searchlight_maps=save)
        tree=ast.parse((ROOT/'panic/searchlight.py').read_text());names={'_cols_for_center','_neighbors_ball_mm','_call_searchlight_plugin','_one_center','_run_searchlight_center','permutation_searchlight'}
        nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
        self.assertEqual(len(nodes),len(names));exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/'panic/searchlight.py'),'exec'),self.ns)

    def run_searchlight(self):
        return self.ns['permutation_searchlight'](None,None,[],{},self.cfg,tmpdir=str(self.folder),save_dir=str(self.folder/'maps'))

    def test_failed_permutation_invalidates_whole_center_and_exports_qc(self):
        result=self.run_searchlight()
        self.assertTrue(np.isnan(self.maps['observed_map'][1,0,0]));self.assertTrue(np.isnan(self.maps['pvalue_map'][1,0,0]))
        self.assertEqual(self.maps['nperms_run'][1,0,0],0)
        self.assertTrue(np.isfinite(self.maps['observed_map'][0,0,0]))
        meta=json.loads((self.folder/'maps/sl/searchlight_desc-metadata.json').read_text())
        self.assertEqual(meta['status'],'partial');self.assertEqual(meta['n_centers_fit_failed'],1)
        self.assertTrue(Path(result['fit_failed']).exists());self.assertTrue(Path(result['fit_failures']).exists())

    def test_original_fingerprint_and_legacy_completed_center_are_preserved(self):
        centers=np.column_stack(np.where(np.ones((4,1,1),dtype=bool)))
        offs=self.ns['_neighbors_ball_mm']((1.,1.,1.),1)
        identity={key:value for key,value in self.cfg.items() if key!='parallel'}
        identity['searchlight']={key:value for key,value in self.cfg['searchlight'].items() if key not in ('checkpoint','basepath')}
        fingerprint=joblib.hash((self.X,self.y,None,centers,np.eye(4),offs,self.folds,identity,0,toy_plugin,{},{}))
        path=self.folder/'maps/sl/searchlight_checkpoint.joblib';path.parent.mkdir(parents=True)
        seeds=np.array([11,22,33,44],dtype=np.uint32)
        old=((0,0,0),.99,.5,.49,.1,2,2,0,0,None)
        joblib.dump({'version':1,'fingerprint':fingerprint,'seeds':seeds,'results':[old,None,None,None]},path)
        self.run_searchlight();after=joblib.load(path)
        self.assertEqual(after['fingerprint'],fingerprint);self.assertEqual(after['results'][0],old)
        np.testing.assert_array_equal(after['seeds'],seeds)
        self.assertEqual(float(self.maps['observed_map'][0,0,0]),float(np.float32(.99)))
        self.assertTrue(after['fit_guard_history'][0]['legacy_fit_status_unverified'])

    def test_strict_policy_propagates_after_saving_checkpoint(self):
        self.cfg['parallel']['svc_fit_guard']={'on_failure':'raise'}
        with self.assertRaises(fg.SearchlightFitError):self.run_searchlight()
        state=joblib.load(self.folder/'maps/sl/searchlight_checkpoint.joblib')
        self.assertIsNotNone(state['results'][0]);self.assertIn('1',state['failures'])
        self.assertFalse((self.folder/'maps/sl/searchlight_desc-metadata.json').exists())

    def test_decode_does_not_swallow_fatal_fit_failure(self):
        import test_roi_skips as skips
        decoder=skips.Decoder();decoder.searchlight=True;decoder.dec_settings_searchlight={}
        failure=fg.SearchlightFitError('fatal solver error')
        with patch.dict(skips.namespace,{'permutation_searchlight':lambda *a,**kw:(_ for _ in ()).throw(failure)}):
            with self.assertRaises(fg.SearchlightFitError):decoder.decode_single_mask(None,None,[],{})

    def test_real_map_assembly_excludes_failed_center_from_fdr(self):
        calls=[]
        def correction(values,**kwargs):calls.append(np.asarray(values));return np.asarray(values)<.05,np.asarray(values)
        self.ns['fdrcorrection']=correction
        tree=ast.parse((ROOT/'panic/searchlight.py').read_text())
        node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='save_searchlight_maps')
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'panic/searchlight.py'),'exec'),self.ns)
        result=self.run_searchlight()
        self.assertEqual(len(calls),1);self.assertEqual(len(calls[0]),3)
        self.assertTrue(np.isfinite(calls[0]).all())
        self.assertTrue(np.isnan(np.load(result['observed'])[1,0,0]))
        self.assertTrue(np.isnan(np.load(result['pvalue'])[1,0,0]))

    def test_all_failed_centers_refuse_final_maps(self):
        self.ns['core'].get_analysis_plugin=lambda *args,**kwargs:(always_fail_plugin,{})
        with self.assertRaisesRegex(fg.SearchlightFitError,'No valid searchlight center'):self.run_searchlight()
        self.assertFalse((self.folder/'maps/sl/searchlight_desc-metadata.json').exists())


if __name__=='__main__':unittest.main()
