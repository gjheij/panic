"""Real NIfTI beta alignment and grouped selection tests without lazyfmri."""
import ast
from collections import Counter
import importlib.util
import json
import logging
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('panic_manifest_under_test', ROOT/'panic/betas.py')
b=importlib.util.module_from_spec(spec)
lazy=types.ModuleType('lazyfmri');lazy.utils=types.SimpleNamespace()
with patch.dict(sys.modules,{'lazyfmri':lazy}):spec.loader.exec_module(b)


def identity(path):
    path=path.resolve();stat=path.stat()
    return dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)


class Manifest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.files=[]
        rows=[]
        for index,(run,label) in enumerate([('01','US'),('01','CS'),('02','US')],start=1):
            path=self.root/f'model-typed_beta-{index}.nii.gz'
            nib.save(nib.Nifti1Image(np.full((2,2,2),index,dtype=np.float32),np.eye(4)),path)
            self.files.append(str(path))
            rows.append(dict(beta_index=index,beta_file=path.name,subject='01',session='',run=run,
                run_index=0 if run=='01' else 1,trial_id=f'trial-{index}',trial_type=label,onset=index,
                duration=0,model='typed'))
        self.table=pd.DataFrame(rows)
        self.manifest=self.root/'model-typed_trial_manifest.tsv'
        self.write()

    def write(self):
        self.table.to_csv(self.manifest,sep='\t',index=False)
        provenance=dict(status='complete',tr=2.,run_order=[
            dict(entities={'run':'01'},n_volumes=7),dict(entities={'run':'02'},n_volumes=9)],
            outputs=[identity(Path(path)) for path in self.files]+[identity(self.manifest)])
        (self.root/'fit_provenance.json').write_text(json.dumps(provenance))

    def load(self, files=None):
        return b.BetaLoaderMixin._load_glmsingle_manifest(files or self.files,'lsa',{'CS':0,'US':1})

    def test_explicit_order_run_labels_and_time_are_preserved(self):
        images,trials,groups,frame=self.load(self.files[::-1])
        self.assertEqual([img.get_fdata()[0,0,0] for img in images],[1,2,3])
        self.assertEqual(trials,['US','CS','US'])
        self.assertEqual(groups,['ses-_run-01','ses-_run-01','ses-_run-02'])
        self.assertEqual(frame.run.tolist(),['01','01','02'])
        self.assertEqual(frame.onset.tolist(),[1.,2.,17.])
        self.assertEqual(frame.label.tolist(),[1,0,1])

    def test_duplicate_trial_is_rejected(self):
        self.table.loc[1,'trial_id']='trial-1';self.write()
        with self.assertRaisesRegex(ValueError,'duplicate'):self.load()

    def test_missing_selected_beta_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'exactly match'):self.load(self.files[:2])

    def test_modified_output_is_rejected(self):
        Path(self.files[0]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'completion provenance'):self.load()

    def test_incomplete_fit_is_rejected(self):
        (self.root/'fit_provenance.json').write_text('{"status":"failed"}')
        with self.assertRaisesRegex(ValueError,'successful completion'):self.load()

    def test_legacy_inputs_return_no_implicit_groups(self):
        self.manifest.unlink()
        self.assertIsNone(self.load())

    def test_public_loader_source_override_keeps_images_and_metadata_aligned(self):
        class Loader(b.BetaLoaderMixin):
            @classmethod
            def sanitize_img(cls, image, **kwargs):
                return image, True
        image,trials,standardize,groups,frame=Loader.load_and_merge_betas(
            subject=self.root.name, beta_dir=str(self.root.parent), source='glmsingle',
            model='lsa',label_mapper={'CS':0,'US':1},standardize=None)
        self.assertEqual(image.shape,(2,2,2,3))
        self.assertEqual(image.get_fdata()[0,0,0].tolist(),[1,2,3])
        self.assertEqual(trials,frame.trial_type.tolist())
        self.assertEqual(groups,frame.run_id.tolist())
        self.assertTrue(standardize)

    def test_public_loader_rejects_global_scaling(self):
        with self.assertRaisesRegex(ValueError, 'standardize: null'):
            b.BetaLoaderMixin.load_and_merge_betas(subject=self.root.name,
                beta_dir=str(self.root.parent),source='glmsingle',model='lsa',standardize='zscore')

    def test_glmsingle_file_filters_include_stream_directory(self):
        found=b.BetaLoaderMixin._find_beta_files(str(self.root),src='glmsingle',model='lsa',filters=self.root.name,derivative=False)
        self.assertEqual(found, sorted(self.files))

    def test_multiple_streams_require_selection(self):
        other=self.root/'other';other.mkdir()
        path=other/'model-typed_beta-1.nii.gz';path.write_bytes(b'dummy')
        with self.assertRaisesRegex(ValueError,'one GLMsingle analysis stream'):
            self.load(self.files+[str(path)])


class Selection(unittest.TestCase):
    def test_selected_indices_preserve_group_alignment(self):
        tree=ast.parse((ROOT/'panic/data.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MaskAndFilterBetas')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='filter_trials')
        ns=dict(np=np,Counter=Counter,logger=logging.getLogger(__name__))
        exec(compile(ast.Module(body=[method],type_ignores=[]),'filter_trials','exec'),ns)
        obj=types.SimpleNamespace(trial_list=['CS','US','US','CS'],label_mapper={'US':1},
                                  betas_in_mask=np.arange(8).reshape(4,2))
        X,trials,labels=ns['filter_trials'](obj)
        self.assertEqual(obj.trial_indices.tolist(),[1,2])
        self.assertEqual(np.asarray(['01','01','02','02'])[obj.trial_indices].tolist(),['01','02'])
        self.assertEqual(X.tolist(),[[2,3],[4,5]])


    def test_roi_passes_selected_groups_to_folds_and_decoder(self):
        tree=ast.parse((ROOT/'panic/decode.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ClassifySubject')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='decode_single_mask')
        calls=[]
        def folds(cfg,y,groups):
            calls.append(groups.tolist());return [(np.array([0]),np.array([1]))]
        def decode(X,y,folds,cfg,**kwargs):
            calls.append(kwargs['groups'].tolist());return {'observed':.5}
        ns=dict(np=np,logger=logging.getLogger(__name__),create_outer_folds=folds,
                run_decoding_with_permutation=decode,EmptyMaskError=type('EmptyMaskError',(Exception,),{}),
                PanicAnalysisError=type('PanicAnalysisError',(Exception,),{}))
        exec(compile(ast.Module(body=[method],type_ignores=[]),'decode_single_mask','exec'),ns)
        extracted=types.SimpleNamespace(labels=np.array([0,1]),X=np.ones((2,2)),
                                        trial_indices=np.array([1,2]),fov_mask=None,roi_linidx=None)
        obj=types.SimpleNamespace(searchlight=False,extract_betas_from_rois=lambda *args,**kwargs:extracted,
                                 dec_settings={'permute_within_groups':False},trial_order_file=None)
        result,_=ns['decode_single_mask'](obj,None,None,trial_list=['CS','US','US','CS'],
            label_mapper={'US':1},groups=['01','01','02','02'])
        self.assertEqual(calls,[['01','02'],['01','02']])
        self.assertEqual(result,{'observed':.5})

if __name__=='__main__':unittest.main()
