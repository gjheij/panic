"""Run matching and spatial QC use the production implementation, no decoding fits."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
try:
    import nibabel as nib
except ImportError:
    nib=None

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('panic_tsnr_qc',ROOT/'panic/tsnr.py')
qc=importlib.util.module_from_spec(spec);spec.loader.exec_module(qc)


class Image:
    def __init__(self,data,affine=None):
        self.data=np.asarray(data);self.affine=np.eye(4) if affine is None else np.asarray(affine)
        self.shape=self.data.shape;self.ndim=self.data.ndim;self.extra={}
    def get_fdata(self):return self.data.astype(float)


def beta(run='1',session=None,task='HRA',condition='CSm'):
    return '/data/sub-017/'+('ses-'+session+'/' if session else '')+'func/task-'+task+'/sub-017_'+('ses-'+session+'_' if session else '')+'task-'+task+('_run-'+run if run else '')+'_space-T1w_feature-LSS_condition-'+condition+'_stat-effect_statmap.nii.gz'


def tsnr(run='1',session=None,task='HRA'):
    return beta(run,session,task).split('_feature-')[0]+'_stat-tsnr_boldmap.nii.gz'


class TsnrTests(unittest.TestCase):
    def test_multiple_conditions_share_one_run_map(self):
        units=qc.matching_tsnr_maps([beta(condition='CSm'),beta(condition='CSpu')],[tsnr()])
        self.assertEqual(len(units),1);self.assertEqual(units[0]['files'],[tsnr()])

    def test_sessions_tasks_and_runs_are_not_mixed(self):
        maps=[tsnr('1','1'),tsnr('2','1'),tsnr('1','2'),tsnr('1','1','OTHER')]
        units=qc.matching_tsnr_maps([beta('1','1'),beta('2','1')],maps)
        self.assertEqual([unit['files'] for unit in units],[[tsnr('1','1')],[tsnr('2','1')]])

    def test_session_directory_fallback_and_numeric_run_padding(self):
        path=tsnr('01','1').replace('sub-017_ses-1_task','sub-017_task')
        units=qc.matching_tsnr_maps([beta('1','1')],[path])
        self.assertEqual(units[0]['files'],[path])

    def test_no_run_dataset_has_no_invented_run_numbers(self):
        units=qc.matching_tsnr_maps([beta(None),beta(None,condition='CSpu')],[tsnr(None),tsnr('1')])
        self.assertEqual(units[0]['files'],[tsnr(None)])

    def test_ambiguous_duplicates_are_reported_not_selected_arbitrarily(self):
        paths=[tsnr().replace('_stat-','_desc-A_stat-'),tsnr().replace('_stat-','_desc-B_stat-')]
        reader=qc.TsnrQC([beta()],paths,loader=lambda p:self.fail('ambiguous map loaded'))
        metrics,rows=reader.extract(Image(np.ones((2,2,2))))
        self.assertTrue(np.isnan(metrics['tsnr_mean']));self.assertEqual(rows[0]['tsnr_status'],'ambiguous')

    def test_only_final_roi_voxels_are_sampled(self):
        data=np.ones((3,3,3))*10;data[1,1,1]=100
        mask=np.zeros((3,3,3));mask[1,1,1]=1
        reader=qc.TsnrQC([beta()],[tsnr()],loader=lambda p:Image(data))
        metrics,rows=reader.extract(Image(mask))
        self.assertEqual(metrics['tsnr_mean'],100);self.assertEqual(rows[0]['tsnr_n_voxels'],1)

    def test_invalid_zero_and_negative_values_are_excluded(self):
        data=np.array([10,0,np.nan,np.inf,-1,30]).reshape(6,1,1)
        metrics,rows=qc.TsnrQC([beta()],[tsnr()],loader=lambda p:Image(data)).extract(Image(np.ones_like(data)))
        self.assertEqual(metrics['tsnr_mean'],20);self.assertEqual(rows[0]['tsnr_median'],20)
        self.assertEqual(rows[0]['tsnr_n_voxels'],2);self.assertAlmostEqual(rows[0]['tsnr_coverage'],1/3)

    def test_runs_are_weighted_equally_despite_different_coverage(self):
        maps={tsnr('1'):Image(np.ones((4,1,1))*10),tsnr('2'):Image(np.array([100,0,0,0]).reshape(4,1,1))}
        reader=qc.TsnrQC([beta('1'),beta('2')],maps,loader=maps.__getitem__)
        metrics,rows=reader.extract(Image(np.ones((4,1,1))))
        self.assertEqual(metrics['tsnr_mean'],55);self.assertEqual(metrics['tsnr_n_maps'],2)
        self.assertEqual(metrics['tsnr_status'],'partial')

    def test_linear_resampling_excludes_padded_outside_fov(self):
        reader=qc.TsnrQC([beta()],[tsnr()],loader=lambda p:Image(np.ones((2,2,2))*20,np.diag([2,2,2,1])))
        metrics,rows=reader.extract(Image(np.ones((5,5,5))))
        self.assertEqual(metrics['tsnr_mean'],20);self.assertEqual(rows[0]['tsnr_n_voxels'],27)

    def test_flipped_affine_preserves_world_coordinate_sampling(self):
        data=np.arange(27).reshape(3,3,3)+1
        affine=np.eye(4);affine[0,0]=-1;affine[0,3]=2
        mask=np.zeros((3,3,3));mask[0,1,1]=1
        metrics,_=qc.TsnrQC([beta()],[tsnr()],loader=lambda p:Image(data,affine)).extract(Image(mask))
        self.assertEqual(metrics['tsnr_mean'],data[2,1,1])

    def test_missing_corrupt_and_empty_rois_do_not_raise(self):
        mask=Image(np.ones((2,2,2)))
        metrics,rows=qc.TsnrQC([beta()],[]).extract(mask)
        self.assertEqual(metrics['tsnr_status'],'missing');self.assertTrue(np.isnan(metrics['tsnr_mean']))
        def corrupt(p):raise ValueError('broken image')
        metrics,rows=qc.TsnrQC([beta()],[tsnr()],loader=corrupt).extract(mask)
        self.assertEqual(rows[0]['tsnr_status'],'invalid')
        metrics,rows=qc.TsnrQC([beta()],[tsnr()],loader=corrupt).extract(Image(np.zeros((2,2,2))))
        self.assertEqual(metrics['tsnr_status'],'empty_roi')

    def test_map_is_resampled_once_for_all_rois_on_same_grid(self):
        calls=[]
        def load(p):calls.append(p);return Image(np.ones((3,3,3))*42)
        reader=qc.TsnrQC([beta()],[tsnr()],loader=load)
        reader.extract(Image(np.ones((3,3,3))));reader.extract(Image(np.eye(3)[:,:,None]*np.ones((1,1,3))))
        self.assertEqual(calls,[tsnr()])

    def test_constructor_attaches_qc_after_final_mask_extraction(self):
        tree=ast.parse((ROOT/'panic/data.py').read_text())
        klass=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='MaskAndFilterBetas')
        init=next(node for node in klass.body if isinstance(node,ast.FunctionDef) and node.name=='__init__')
        namespace={'EmptyMaskError':type('EmptyMaskError',(ValueError,),{})}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Masker',bases=[],keywords=[],decorator_list=[],body=[init])],type_ignores=[])),'data.py','exec'),namespace)
        masker=namespace['Masker']
        final=np.zeros((3,3,3));final[1,1,1]=1
        def extract(self,**kwargs):
            self.fov_mask=Image(final);self.roi_metrics={'n_voxels':1};return np.ones((4,1)),np.array([13])
        masker._masker=extract;masker.filter_trials=lambda self:(self.betas_in_mask,[],[])
        reader=qc.TsnrQC([beta()],[tsnr()],loader=lambda p:Image(np.ones((3,3,3))*50))
        result=masker(None,None,tsnr_qc=reader)
        self.assertEqual(result.roi_metrics['tsnr_mean'],50)
        self.assertEqual(result.tsnr_rows[0]['tsnr_n_voxels'],1)

    def test_per_run_table_updates_preserve_other_roi_rows(self):
        spec=importlib.util.spec_from_file_location('tsnr_store',ROOT/'panic/result_store.py')
        store=importlib.util.module_from_spec(spec);spec.loader.exec_module(store)
        import csv
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'qc.csv'
            def row(roi,run,value):return dict(subject='sub-017',analysis='contrast',method='LSA',source='halfpipe',roi=roi,hemi='left',run=run,tsnr_mean=value)
            store.write_qc_rows(path,[row('bl','1',10),row('lat','1',30)])
            store.write_qc_rows(path,[row('bl','1',20),row('bl','2',40)])
            with path.open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),3)
            self.assertEqual([r['tsnr_mean'] for r in rows if r['roi']=='lat'],['30'])

    def test_loaded_beta_paths_survive_concatenation_and_sanitization(self):
        import logging,os,types,pandas as pd
        from typing import Mapping
        tree=ast.parse((ROOT/'panic/betas.py').read_text())
        klass=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='BetaLoaderMixin')
        methods=[node for node in klass.body if isinstance(node,ast.FunctionDef) and node.name in ('load_and_merge_betas','_load_statmap_betas')]
        namespace={'Mapping':Mapping,'os':os,'opj':os.path.join,'np':np,
            'logger':logging.getLogger(__name__),'utils':types.SimpleNamespace(get_file_from_substring=lambda needles,files:[p for p in files if all(n in p for n in needles)])}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Loader',bases=[],keywords=[],decorator_list=[],body=methods)],type_ignores=[])),'betas.py','exec'),namespace)
        loader=namespace['Loader'];paths=[beta(condition='CSm'),beta(condition='CSpu')]
        loader._validate_beta_dir=classmethod(lambda cls,*args:None)
        loader._validate_source=classmethod(lambda cls,*args:None)
        loader._find_beta_files=classmethod(lambda cls,*args,**kw:paths)
        loader._load_nifti=classmethod(lambda cls,path:Image(np.ones((2,2,2))))
        loader._as_list=classmethod(lambda cls,value,**kw:value)
        loader._parse_run_id=classmethod(lambda cls,*args:1)
        loader._read_json_sidecar=classmethod(lambda cls,path:None)
        loader._build_events_df=classmethod(lambda cls,*args:pd.DataFrame())
        loader._concat_betas=classmethod(lambda cls,images:Image(np.ones((2,2,2,2))))
        loader._validate_beta_metadata=classmethod(lambda cls,*args,**kw:None)
        loader.sanitize_img=classmethod(lambda cls,img,**kw:(img,True))
        images,*_=loader.load_and_merge_betas('sub-017',beta_dir='/data/halfpipe',label_mapper={'CSm':0,'CSpu':1},model='LSS')
        self.assertEqual(images.extra['panic_beta_files'],paths)

    @unittest.skipUnless(nib is not None,'nibabel unavailable')
    def test_real_nifti_loading_and_resampling(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/Path(tsnr()).name
            nib.save(nib.Nifti1Image(np.ones((2,2,2),dtype=np.float32)*20,np.diag([2,2,2,1])),path)
            mask=nib.Nifti1Image(np.ones((5,5,5),dtype=np.uint8),np.eye(4))
            metrics,rows=qc.TsnrQC([beta()],[path]).extract(mask)
            self.assertEqual(metrics['tsnr_mean'],20)
            self.assertEqual(rows[0]['tsnr_n_voxels'],27)

    def test_singleton_fourth_dimension_is_a_single_map(self):
        reader=qc.TsnrQC([beta()],[tsnr()],loader=lambda p:Image(np.ones((2,2,2,1))*25))
        metrics,_=reader.extract(Image(np.ones((2,2,2))))
        self.assertEqual(metrics['tsnr_mean'],25)


if __name__=='__main__':unittest.main()
