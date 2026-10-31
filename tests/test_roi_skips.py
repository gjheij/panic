"""Explicit empty ROI rows, without importing PANIC's full imaging stack."""
import ast
import importlib.util
import logging
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('roi_skip_errors',ROOT/'panic/errors.py')
errors=importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)
methods={'_empty_mask_result','_failed_result','_extend_results_dict','decode_single_mask','decode_masks'}
tree=ast.parse((ROOT/'panic/decode.py').read_text())
original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ClassifySubject')
klass=ast.ClassDef(name='Decoder',bases=[],keywords=[],decorator_list=[],
    body=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in methods])
store_spec=importlib.util.spec_from_file_location('roi_skip_store',ROOT/'panic/result_store.py')
store=importlib.util.module_from_spec(store_spec)
store_spec.loader.exec_module(store)
namespace=dict(merge_result_tables=store.merge_result_tables,np=np,pd=pd,os=os,opj=os.path.join,shutil=shutil,
    logger=logging.getLogger(__name__),PanicAnalysisError=errors.PanicAnalysisError,
    EmptyMaskError=errors.EmptyMaskError,NoFeaturesSelectedError=errors.NoFeaturesSelectedError,
    mask_erosion_request=lambda *a:('voxels',0),cv_permutation_policy=lambda cfg:'fixed',
    dump_yaml=lambda value,path:Path(path).write_text(yaml.safe_dump(value)),
    create_outer_folds=lambda cfg,labels,**kw:[(np.arange(2),np.arange(2,4))])
exec(compile(ast.fix_missing_locations(ast.Module(body=[klass],type_ignores=[])),str(ROOT/'panic/decode.py'),'exec'),namespace)
Decoder=namespace['Decoder']

class ROISkipTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.decoder=Decoder()
        d=self.decoder
        d.cfg={'general_settings':{'source':'halfpipe','method':'LSA','tmp_dir':str(self.root),'overwrite':True},
               'decoding_settings':{'analysis':{'type':'scikit_decoding'},'n_permutations':1},
               'label_dict':{'CSm':0,'CSpu':1}}
        d.gen_settings=d.cfg['general_settings'];d.dec_settings=d.cfg['decoding_settings']
        d.subject='sub-017';d.analysis_name='primary';d.analysis_type='scikit_decoding';d.analysis_id='identity'
        d.searchlight=False;d.save_imgs=False;d.save_dir=str(self.root);d.groups=None
        d.betas=None;d.trial_list=['CSm','CSpu','CSm','CSpu'];d.trial_order_file=None
        d.config_file=str(self.root/'config.yml');Path(d.config_file).write_text('test: true\n')
        d.generate_filename=lambda **kw:str(self.root/('sub-017_desc-'+kw['desc']+'.'+kw['ext']))
        d.define_mask_inputs=lambda:({'ahta':[320]},True)
        d.prepare_rois=lambda *a,**kw:(None,{'left':['ahta','left'],'right':['ahta','right']})

    def empty(self,reason='empty_mask_after_resampling'):
        return errors.EmptyMaskError('No voxels remain',reason_code=reason,
            roi_metrics={'n_voxels':0,'n_voxels_native':234,'n_voxels_resampled':0,'valid_voxel_fraction':0})

    def test_known_empty_masks_skip_but_unknown_mask_errors_fail(self):
        for reason in ['empty_native_mask','empty_mask_after_resampling','empty_mask_after_erosion','empty_mask_after_support_filtering']:
            result=self.decoder._empty_mask_result(self.empty(reason))
            self.assertEqual(result['status'],'skipped')
            self.assertEqual(result['n_features'],0)
            for key in ['observed','mean_permuted','delta','p']:
                self.assertTrue(np.isnan(result[key]))
        self.assertEqual(self.decoder._empty_mask_result(errors.EmptyMaskError('unknown'))['status'],'failed')

    def test_extraction_failure_produces_row_instead_of_disappearing(self):
        def extraction(betas,mask,**kw):
            raise self.empty()
        self.decoder.extract_betas_from_rois=extraction
        self.decoder.decode_masks()
        rows=pd.read_csv(self.root/'sub-017_desc-results.csv')
        self.assertEqual(set(rows['hemi']),{'left','right'})
        self.assertEqual(set(rows['status']),{'skipped'})
        self.assertEqual(set(rows['n_features']),{0})
        self.assertTrue(rows['observed_acc'].isna().all())
        self.assertEqual(set(rows['n_voxels_native']),{234})
        self.assertEqual(set(rows['reason_code']),{'empty_mask_after_resampling'})
        self.assertTrue(rows['n_samples'].isna().all())

    def test_empty_left_and_successful_right_both_survive_csv_export(self):
        extract=types.SimpleNamespace(X=np.ones((4,1)),labels=np.array([0,1,0,1]),
            roi_metrics={'n_voxels':1},fov_mask=None,roi_linidx=np.array([0]))
        def extraction(betas,mask,**kw):
            if mask=='left':raise self.empty()
            return extract
        self.decoder.extract_betas_from_rois=extraction
        def decode(*a,**kw):
            result=self.decoder._failed_result()
            result.update(observed=.75,mean_permuted=.5,delta=.25,p=.5,
                          permuted=np.array([.5]),status='ok',error_type=None,error=None,n_run=1)
            return result
        namespace['run_decoding_with_permutation']=decode
        self.decoder.decode_masks()
        rows=pd.read_csv(self.root/'sub-017_desc-results.csv').set_index('hemi')
        self.assertEqual(rows.loc['left','status'],'skipped')
        self.assertEqual(rows.loc['right','status'],'ok')
        self.assertEqual(rows.loc['right','observed_acc'],.75)
        null=pd.read_csv(self.root/'sub-017_desc-null_distribution.csv')
        self.assertEqual(set(null['hemi']),{'right'})

    def test_zero_resampled_voxels_produce_qc_without_division_by_zero(self):
        tree=ast.parse((ROOT/'panic/data.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MaskAndFilterBetas')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_compute_roi_metrics')
        method.decorator_list=[]
        ns={'np':np}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(ROOT/'panic/data.py'),'exec'),ns)
        def image(values):
            return types.SimpleNamespace(affine=np.eye(4),get_fdata=lambda:values)
        zero=np.zeros((2,2,2))
        for native in [zero,np.ones_like(zero)]:
            qc=ns['_compute_roi_metrics'](image(native),image(zero),valid_mask=zero)
            self.assertEqual(qc['n_voxels'],0)
            self.assertEqual(qc['n_voxels_resampled'],0)
            self.assertEqual(qc['valid_voxel_fraction'],0)

    def test_other_extraction_errors_remain_failed_rows(self):
        def extraction(*a,**kw):raise ValueError('bad input')
        self.decoder.extract_betas_from_rois=extraction
        self.decoder.decode_masks()
        rows=pd.read_csv(self.root/'sub-017_desc-results.csv')
        self.assertEqual(set(rows['status']),{'failed'})
        self.assertEqual(set(rows['error_type']),{'ValueError'})

try:
    import nibabel as nib
    import nilearn
except ImportError:
    nib=None

@unittest.skipUnless(nib is not None,'nibabel/nilearn unavailable')
class MaskQCIntegrationTests(unittest.TestCase):
    def test_empty_mask_errors_carry_reason_and_qc(self):
        from panic.data import MaskAndFilterBetas
        from panic.errors import EmptyMaskError
        for case in ['native','resampling','erosion','support']:
            with self.subTest(case=case):
                B=np.random.default_rng(6).normal(size=(5,5,5,4))
                M=np.ones((5,5,5),dtype=np.uint8)
                affine=np.eye(4)
                settings={}
                if case=='native':M[:]=0
                if case=='resampling':affine[:3,3]=1000
                if case=='erosion':settings={'erode_masks':['ahta'],'erosion_voxels':3}
                if case=='support':B[:]=1
                with self.assertRaises(EmptyMaskError) as caught:
                    MaskAndFilterBetas(nib.Nifti1Image(B,np.eye(4)),nib.Nifti1Image(M,affine),
                        trial_list=['CSm','CSpu','CSm','CSpu'],label_mapper={'CSm':0,'CSpu':1},
                        mask_settings=settings,roi_name='ahta')
                expected={'native':'empty_native_mask','resampling':'empty_mask_after_resampling',
                          'erosion':'empty_mask_after_erosion','support':'empty_mask_after_support_filtering'}[case]
                self.assertEqual(caught.exception.reason_code,expected)
                self.assertEqual(caught.exception.roi_metrics['n_voxels'],0)

if __name__=='__main__':unittest.main()
