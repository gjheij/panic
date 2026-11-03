"""Registered HALFpipe QC uses explicit provenance, never a filename-only guess."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from test_tsnr import qc, Image, nib


class RegisteredTsnrTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.beta = self.root/'sub-sc001_task-FearAcquisitionRepLear_feature-LSS_condition-CSm_stat-effect_statmap.nii.gz'
        self.map = self.root/'sub-sc001_task-FearAcquisitionRepLear_desc-betagrid_stat-tsnr_boldmap.nii.gz'
        self.original = self.root/'sub-sc001_task-FearAcquisitionRepLear_stat-tsnr_boldmap.nii.gz'
        self.metadata = {'SpatialReference': 'T1w', 'ReferenceImage': str(self.beta).replace('LSS', 'LSA'),
                         'ReferenceGrid': {'Shape': [3,3,3], 'Affine': np.eye(4).tolist()},
                         'Sources': [str(self.original)], 'AppliedTransform': 'mni-to-t1w.h5',
                         'SourceMetadata': {'Setting': {'Space': 'standard'}},
                         'Setting': {'Space': 'standard'}}  # Older registration outputs retain source setting.
        self.write(self.beta, {'Setting': {'Space': 'native'}})
        self.write(self.original, {'Setting': {'Space': 'standard'}})
        self.write(self.map, self.metadata)
        self.images = {str(self.beta): Image(np.ones((3,3,3,4))), str(self.map): Image(np.ones((3,3,3))*42)}

    def write(self, path, metadata):
        qc.tsnr_sidecar(path).write_text(json.dumps(metadata))

    def select(self, maps=None):
        return qc.matching_tsnr_maps([self.beta], maps or [self.original, self.map])[0]

    def extract(self):
        return qc.TsnrQC([self.beta], [self.original,self.map], loader=self.images.__getitem__).extract(Image(np.ones((3,3,3))))

    def test_unlabelled_registered_map_supersedes_original_template_map(self):
        self.assertEqual(self.select()['files'], [str(self.map)])
        metrics, rows = self.extract()
        self.assertEqual(metrics['tsnr_mean'], 42)
        self.assertEqual(rows[0]['tsnr_status'], 'ok')

    def test_explicit_t1w_filename_also_supported(self):
        named = self.map.with_name(self.map.name.replace('_desc-', '_space-T1w_desc-'))
        self.write(named, self.metadata)
        self.assertEqual(self.select([named])['files'], [str(named)])

    def test_desc_alone_is_not_evidence_of_space_or_registration(self):
        self.write(self.map, {})
        self.assertEqual(self.select()['files'], [])

    def test_unknown_beta_space_is_not_guessed_from_reference(self):
        self.write(self.beta, {})
        self.assertIn('Beta coordinate space', self.select()['error'])
        self.assertEqual(self.extract()[1][0]['tsnr_status'], 'unresolved')

    def test_wrong_reference_acquisition_rejected(self):
        self.metadata['ReferenceImage'] = str(self.beta).replace('task-FearAcquisitionRepLear', 'task-OTHER')
        self.write(self.map, self.metadata)
        self.assertFalse(self.select()['files'])

    def test_malformed_json_rejected(self):
        qc.tsnr_sidecar(self.map).write_text('{broken')
        self.assertFalse(self.select()['files'])

    def test_spatial_entity_and_metadata_conflict_rejected(self):
        named = self.map.with_name(self.map.name.replace('_desc-', '_space-MNI152NLin2009cAsym_desc-'))
        self.write(named, self.metadata)
        self.assertFalse(self.select([named])['files'])

    def test_identical_grid_in_wrong_space_is_never_sampled(self):
        self.metadata['SpatialReference'] = 'MNI152NLin2009cAsym'
        self.write(self.map, self.metadata)
        self.assertFalse(self.select()['files'])
        self.assertTrue(np.isnan(self.extract()[0]['tsnr_mean']))

    def test_map_header_must_match_declared_reference_grid(self):
        self.images[str(self.map)] = Image(np.ones((3,3,3)), np.diag([2,2,2,1]))
        metrics, rows = self.extract()
        self.assertEqual(rows[0]['tsnr_status'], 'invalid')
        self.assertIn('ReferenceGrid', rows[0]['tsnr_error'])
        self.assertTrue(np.isnan(metrics['tsnr_mean']))

    def test_all_selected_beta_headers_must_match_reference_grid(self):
        self.images[str(self.beta)] = Image(np.ones((4,3,3,4)))
        self.assertIn('selected beta grid', self.extract()[1][0]['tsnr_error'])

    def test_two_registered_maps_remain_ambiguous(self):
        other = self.root/'other'/self.map.name
        other.parent.mkdir()
        self.write(other, self.metadata)
        records = self.select([self.map,other])
        self.assertEqual(len(records['files']),2)

    @unittest.skipUnless(nib is not None, 'nibabel unavailable')
    def test_real_final_mask_after_erosion_and_beta_support(self):
        # Execute the production class with leaf imports to avoid unrelated package metadata.
        import ast
        import importlib.util
        import logging
        from collections import Counter
        try:
            from nilearn import image, maskers
        except ImportError:
            self.skipTest('nilearn unavailable')
        root = Path(qc.__file__).parent
        def leaf(name):
            spec = importlib.util.spec_from_file_location('tsnr_test_'+name,root/(name+'.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        masking, errors, resampling = leaf('masking'),leaf('errors'),leaf('resampling')
        klass = next(node for node in ast.parse((root/'data.py').read_text()).body
                     if isinstance(node,ast.ClassDef) and node.name=='MaskAndFilterBetas')
        namespace = dict(np=np,nib=nib,image=image,maskers=maskers,Path=Path,Counter=Counter,
                         logger=logging.getLogger(__name__),EmptyMaskError=errors.EmptyMaskError,
                         beta_support_mask=masking.beta_support_mask,erode_binary_mask=masking.erode_binary_mask,
                         mask_erosion_request=masking.mask_erosion_request,
                         resample_to_voxel_size=resampling.resample_to_voxel_size)
        exec(compile(ast.Module(body=[klass],type_ignores=[]),'data.py','exec'),namespace)
        values = np.broadcast_to(np.arange(4,dtype=np.float32),(7,7,7,4)).copy()
        values[3,3,3,:] = 0
        betas = nib.Nifti1Image(values,np.eye(4))
        nib.save(betas,self.beta)
        self.metadata['ReferenceGrid']['Shape'] = [7,7,7]
        self.write(self.map,self.metadata)
        tsnr = np.full((7,7,7),1000.,dtype=np.float32)
        tsnr[2:5,2:5,2:5] = 42
        tsnr[3,3,3] = 999
        nib.save(nib.Nifti1Image(tsnr,np.eye(4)),self.map)
        mask = np.zeros((7,7,7),dtype=np.uint8)
        mask[1:6,1:6,1:6] = 1
        result = namespace['MaskAndFilterBetas'](betas,nib.Nifti1Image(mask,np.eye(4)),
            trial_list=['CSm','CSpu','CSm','CSpu'],label_mapper={'CSm':0,'CSpu':1},
            mask_settings={'erode_masks':['wm'],'erosion_voxels':1},roi_name='wm',
            tsnr_qc=qc.TsnrQC([self.beta],[self.map]))
        self.assertEqual(result.X.shape,(4,26))
        self.assertEqual(result.roi_metrics['tsnr_mean'],42)
        self.assertEqual(result.tsnr_rows[0]['tsnr_n_voxels'],26)

    @unittest.skipUnless(nib is not None, 'nibabel unavailable')
    def test_registered_nifti_and_json_on_disk(self):
        for path, img in self.images.items():
            nib.save(nib.Nifti1Image(img.get_fdata(), img.affine), path)
        metrics, rows = qc.TsnrQC([self.beta], [self.map,self.original]).extract(nib.Nifti1Image(np.ones((3,3,3)), np.eye(4)))
        self.assertEqual(metrics['tsnr_mean'],42)
        self.assertEqual(rows[0]['tsnr_n_voxels'],27)


if __name__ == '__main__':
    unittest.main()
