"""Fatal storage errors, persistent checkpoints, temp routing and CLI cleanup."""
import ast
import contextlib
import errno
import importlib
import importlib.util
import io
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PACKAGE='panic_storage_test'
pkg=types.ModuleType(PACKAGE);pkg.__path__=[str(ROOT/'panic')];sys.modules[PACKAGE]=pkg
storage=importlib.import_module(PACKAGE+'.storage')
checkpoint=importlib.import_module(PACKAGE+'.checkpoint')
progress=importlib.import_module(PACKAGE+'.searchlight_progress')
monitor=importlib.import_module(PACKAGE+'.monitor')


def disk_full():return OSError(errno.ENOSPC,'No space left on device')


def memmap_location(values):
    return str(values.filename)


def functions(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_temp_policy_and_single_resolution(self):
        with patch.dict(os.environ,{'TMPDIR':str(self.root/'env')}),patch.object(storage.tempfile,'gettempdir',return_value=str(self.root/'python')):
            self.assertEqual(storage.resolve_tmp_dir(self.root/'explicit'),str(self.root/'explicit'))
            self.assertEqual(storage.resolve_tmp_dir(),str(self.root/'env'))
            with patch.dict(os.environ,{'TMPDIR':''}):
                selected=storage.resolve_tmp_dir()
                self.assertEqual(selected,str(self.root/'python'))
            with patch.object(storage.tempfile,'TemporaryFile',side_effect=AssertionError('resolved twice')):
                self.assertIs(storage.resolve_tmp_dir(selected),selected)

    def test_temp_unusable_does_not_fall_back(self):
        with patch.object(storage.tempfile,'TemporaryFile',side_effect=disk_full()),self.assertRaises(OSError) as caught:
            storage.resolve_tmp_dir(self.root/'selected')
        self.assertEqual(caught.exception.errno,errno.ENOSPC)
        self.assertEqual(caught.exception.filename,str(self.root/'selected'))
        with self.assertRaises(ValueError):storage.resolve_tmp_dir('')

    def test_joblib_automatic_memmaps_use_selected_parent(self):
        selected=storage.resolve_tmp_dir(self.root/'selected')
        with joblib.parallel_config(temp_folder=selected):
            paths=joblib.Parallel(n_jobs=2,max_nbytes=1024)(
                joblib.delayed(memmap_location)(np.ones(10000)) for _ in range(2))
        self.assertTrue(all(Path(selected) in Path(path).parents for path in paths))

    def test_default_config_and_parallel_routing(self):
        import yaml
        self.assertIsNone(yaml.safe_load((ROOT/'panic/config.yml').read_text())['general_settings']['tmp_dir'])
        tree=ast.parse((ROOT/'panic/decode.py').read_text())
        roi=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_decoding_with_permutation')
        calls=[n for n in ast.walk(roi) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Parallel']
        self.assertTrue(calls)
        self.assertTrue(all(any(k.arg=='temp_folder' and isinstance(k.value,ast.Name) and k.value.id=='tmpd' for k in c.keywords) for c in calls))
        sl=(ROOT/'panic/searchlight.py').read_text()
        self.assertIn('temp_folder=tmpd',sl)
        self.assertNotIn('~/.joblib_cache',sl)

    def test_checkpoint_failure_stages_preserve_original(self):
        path=self.root/'persistent/checkpoint.joblib'
        obj=checkpoint.SearchlightCheckpoint(str(path),'identity',np.arange(2),2)
        obj.save();before=path.read_bytes()
        for target,stage in [('dump','serialization'),('fsync','fsync'),('replace','replacement')]:
            owner=checkpoint.joblib if target=='dump' else checkpoint.os
            with self.subTest(stage=stage),patch.object(owner,target,side_effect=disk_full()):
                with self.assertLogs(checkpoint.logger,level='INFO') as logs,self.assertRaises(OSError) as caught:
                    obj.save()
                self.assertEqual(caught.exception.filename,str(path))
                self.assertIn(stage,str(caught.exception))
                self.assertIn('.checkpoint-',str(caught.exception))
                self.assertEqual(caught.exception.errno,errno.ENOSPC)
                self.assertIsInstance(caught.exception.__cause__,OSError)
                self.assertIn(str(path),logs.output[0])
            self.assertEqual(path.read_bytes(),before)
            self.assertFalse(list(path.parent.glob('.checkpoint-*')))

    def test_checkpoint_flush_close_and_creation(self):
        from test_csv_atomic import FailingStream
        path=self.root/'checkpoint.joblib'
        obj=checkpoint.SearchlightCheckpoint(str(path),'id',np.arange(1),1);obj.save()
        before=path.read_bytes();fdopen=checkpoint.os.fdopen
        for stage,expected in [('flush','flush'),('close','close temporary file')]:
            with patch.object(checkpoint.os,'fdopen',side_effect=lambda *a,**kw:FailingStream(fdopen(*a,**kw),stage)):
                with self.assertRaises(OSError) as caught:obj.save()
            self.assertIn(expected,str(caught.exception));self.assertEqual(path.read_bytes(),before)
        with patch.object(checkpoint.tempfile,'mkstemp',side_effect=disk_full()),self.assertRaises(OSError) as caught:
            obj.save()
        self.assertEqual(caught.exception.filename,str(path))

    def test_successful_save_cannot_hide_cleanup_storage_error(self):
        path=self.root/'checkpoint.joblib'
        obj=checkpoint.SearchlightCheckpoint(str(path),'id',[0],1)
        with patch.object(checkpoint.os,'unlink',side_effect=disk_full()),self.assertRaises(OSError) as caught:
            obj.save()
        self.assertIn('cleanup failed',str(caught.exception))
        self.assertEqual(caught.exception.filename,str(path))
        self.assertTrue(path.exists())  # Published data is valid, but execution failed.

    def test_many_failed_centers_are_batched_and_final_tail_saved(self):
        cp=types.SimpleNamespace(results=[None]*5,failures={},save=Mock())
        policy={'enabled':True,'on_failure':'mark_center'}
        runner=progress.CenterProgress(cp.results,cp,policy,str(self.root/'sl'),update_interval=3)
        failure=progress.CenterFitFailure((0,0,0),{'n_features':1})
        runner.record(0,failure);runner.record(1,failure);cp.save.assert_not_called()
        runner.record(2,failure);self.assertEqual(cp.save.call_count,1)
        runner.record(3,failure);self.assertEqual(cp.save.call_count,1)
        runner.record(4,failure);self.assertEqual(cp.save.call_count,2)

    def test_strict_failure_saved_immediately(self):
        cp=types.SimpleNamespace(results=[None]*5,failures={},save=Mock())
        runner=progress.CenterProgress(cp.results,cp,{'enabled':True,'on_failure':'raise'},str(self.root/'sl'))
        with self.assertRaises(progress.SearchlightFitError):
            runner.record(0,progress.CenterFitFailure((0,0,0),{}))
        cp.save.assert_called_once()

    def test_monitor_write_errors_are_fatal(self):
        with patch.object(monitor,'_atomic_json',side_effect=disk_full()),self.assertRaises(OSError):
            monitor.worker_heartbeat(str(self.root),0,'START')
        m=monitor.SearchlightMonitor(1,runtime_dir=str(self.root/'runtime'),snapshot_dir=str(self.root/'snap'),logger=logging.getLogger('monitor-test'))
        with patch.object(monitor.os,'fsync',side_effect=disk_full()),self.assertRaises(OSError):m.dump_snapshot()
        m._storage_error=disk_full()
        with self.assertRaises(OSError):m.mark_completed(0)

    def test_watchdog_storage_error_reaches_parent(self):
        m=monitor.SearchlightMonitor(1,runtime_dir=str(self.root/'runtime'),snapshot_dir=str(self.root/'snap'),logger=logging.getLogger('monitor-test'))
        with patch.object(m._stop,'wait',return_value=False),patch.object(m,'dump_snapshot',side_effect=disk_full()):m._watchdog()
        with self.assertRaises(OSError):m.close()

    def test_initial_searchlight_checkpoint_is_logged_before_failure(self):
        import test_svc_fit_guard as fixtures
        case=fixtures.SearchlightIntegrationTests();case.setUp()
        try:
            with patch.object(fixtures.cp.SearchlightCheckpoint,'save',side_effect=disk_full()), \
                 self.assertLogs(case.ns['logger'],level='INFO') as logs,self.assertRaises(OSError):
                case.run_searchlight()
            self.assertTrue(any('checkpoint destination:' in line for line in logs.output))
            self.assertFalse(any('Storing searchlight information' in line for line in logs.output))
        finally:case.doCleanups()


class CLITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.base=self.root/'base.yml';self.base.write_text('base')
        self.cfg={'general_settings':{'save_dir':str(self.root/'outputs'),'source':'halfpipe','method':'LSA'},
                  'decoding_settings':{'analysis':{'name':'contrast'}}}
        self.ns={'Path':Path,'os':os,'opj':os.path.join,'sys':sys,'logging':logging,
                 'Dict':dict,'Any':object,'tempfile':tempfile,'List':list,
                 'load_yaml':lambda p:self.cfg,'dump_yaml':lambda cfg,p:Path(p).write_text('temporary'),
                 'apply_assignments':lambda cfg,v:(cfg,[]),'init_logging':lambda **kw:None}
        functions(ROOT/'panic/cli.py',{'_write_temp_config','action_run','main'},self.ns)
        self.args=types.SimpleNamespace(config=str(self.base),subject=['sub-1'],set=['x=1'],
            searchlight=False,debug=False,save_imgs=False)

    def invoke(self,error=None,overrides=True):
        self.args.set=['x=1'] if overrides else []
        files=[]
        class Decoder:
            def __init__(_,subj,config,**kw):files.append(Path(config))
            def _fit(_):
                if error is not None:raise error
        self.ns['ClassifySubject']=Decoder
        self.ns['build_parser']=lambda:types.SimpleNamespace(parse_args=lambda argv:types.SimpleNamespace(func=lambda args:self.ns['action_run'](self.args)))
        with contextlib.redirect_stdout(io.StringIO()):
            if error is None:self.ns['main']([])
            else:
                with self.assertRaises(type(error)) as caught:self.ns['main']([])
                self.assertIs(caught.exception,error)
        return files

    def test_own_config_removed_after_success_and_failure(self):
        for error in (None,disk_full()):
            files=self.invoke(error)
            self.assertEqual(len(files),1);self.assertFalse(files[0].exists())
            self.assertEqual(self.base.read_text(),'base')

    def test_base_config_never_removed(self):
        files=self.invoke(overrides=False)
        self.assertEqual(files,[self.base]);self.assertTrue(self.base.exists())

    def test_dump_failure_cleans_created_config(self):
        created=self.root/'panic_cfg_owned.yml'
        fd=os.open(created,os.O_CREAT|os.O_WRONLY,0o600)
        with patch.object(tempfile,'mkstemp',return_value=(fd,str(created))), \
             patch.dict(self.ns,{'dump_yaml':Mock(side_effect=disk_full())}),self.assertRaises(OSError):
            self.ns['_write_temp_config']({})
        self.assertFalse(created.exists())

    def test_config_cleanup_failure_keeps_primary_exception(self):
        primary=disk_full()
        with patch.object(Path,'unlink',side_effect=PermissionError('cleanup denied')):
            files=self.invoke(error=primary)
        for path in files:path.unlink()


class FatalDecoderTests(unittest.TestCase):
    def test_cli_subprocess_storage_failures_exit_nonzero_without_complete(self):
        # Production main -> action_run -> _fit -> decode_single_mask, with
        # synthetic extraction so this test does not require an imaging stack.
        harness = r'''
import ast, logging, os, sys, tempfile, types
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import test_roi_skips as fixture
import test_storage_safety as safety
logging.basicConfig(level=logging.INFO)
case=fixture.ROISkipTests();case.setUp();d=case.decoder
root=case.root
d.do_standardization=False;d._init_betas=lambda:None
tree=ast.parse((safety.ROOT/'panic/decode.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ClassifySubject')
fit=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_fit')
fixture.namespace['update_kwargs']=lambda kwargs,key,value:dict(kwargs,**{key:value})
exec(compile(ast.Module(body=[fit],type_ignores=[]),'production_fit','exec'),fixture.namespace)
fixture.Decoder._fit=fixture.namespace['_fit']
def storage_write(*a,**kw):
    if sys.argv[1]=='roi':
        fixture.store.write_dataframe_csv(pd.DataFrame({'x':[1]}),root/'trial.csv',index=False)
    else:
        cp=safety.checkpoint.SearchlightCheckpoint(str(root/'persistent/checkpoint.joblib'),'id',[0],1)
        cp.save()
d.extract_betas_from_rois=storage_write
if sys.argv[1]=='searchlight':
    d.searchlight=True;d.dec_settings_searchlight=d.dec_settings
    fixture.namespace['permutation_searchlight']=storage_write
d.decode_masks=lambda **kw:d.decode_single_mask(None,None,trial_list=d.trial_list,label_mapper=d.cfg['label_dict'])
args=types.SimpleNamespace(config=str(root/'base.yml'),subject=['sub-017'],set=None,searchlight=d.searchlight,debug=False,save_imgs=False)
cfg={'general_settings':{'save_dir':str(root),'source':'halfpipe','method':'LSA'},'decoding_settings':{'analysis':{'name':'contrast'}}}
ns={'Path':Path,'os':os,'opj':os.path.join,'sys':sys,'logging':logging,'List':list,
    'load_yaml':lambda p:cfg,'init_logging':lambda **kw:None,'ClassifySubject':lambda *a,**kw:d,
    'build_parser':lambda:types.SimpleNamespace(parse_args=lambda argv:types.SimpleNamespace(func=lambda a:ns['action_run'](args)))}
safety.functions(safety.ROOT/'panic/cli.py',{'action_run','main'},ns)
with patch.object(os,'fsync',side_effect=safety.disk_full()):
    ns['main']([])
'''
        env=dict(os.environ,PYTHONPATH=str(ROOT/'tests'))
        for mode in ('roi','searchlight'):
            with self.subTest(mode=mode):
                result=subprocess.run([sys.executable,'-c',harness,mode],env=env,capture_output=True,text=True,timeout=30)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('No space left on device',result.stderr)
                self.assertIn('Filesystem operation failed; aborting decoding',result.stderr)
                self.assertNotIn('Decoding sub-017 complete',result.stdout+result.stderr)

    def test_roi_extraction_analysis_and_searchlight_io_errors_propagate(self):
        import test_roi_skips as fixtures
        case=fixtures.ROISkipTests();case.setUp()
        try:
            error=disk_full();d=case.decoder
            with patch.object(d,'extract_betas_from_rois',side_effect=error,create=True):
                with self.assertRaises(OSError):d.decode_single_mask(None,None)
            extract=types.SimpleNamespace(X=np.ones((4,1)),labels=np.array([0,1,0,1]),fov_mask=None,roi_linidx=np.array([0]))
            d.extract_betas_from_rois=lambda *a,**kw:extract
            with patch.dict(fixtures.namespace,{'run_decoding_with_permutation':Mock(side_effect=error)}):
                with self.assertRaises(OSError):d.decode_single_mask(None,None,trial_list=d.trial_list,label_mapper=d.cfg['label_dict'])
            d.searchlight=True;d.dec_settings_searchlight=d.dec_settings
            with patch.dict(fixtures.namespace,{'permutation_searchlight':Mock(side_effect=error)}):
                with self.assertRaises(OSError):d.decode_single_mask(None,None,trial_list=d.trial_list,label_mapper=d.cfg['label_dict'])
        finally:case.doCleanups()


if __name__=='__main__':unittest.main()
