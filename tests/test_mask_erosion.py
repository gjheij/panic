"""Named-mask erosion numerical tests and optional image integration tests."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

source = Path(__file__).resolve().parents[1] / "panic" / "masking.py"
spec = importlib.util.spec_from_file_location("panic_erosion_helpers", source)
masking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(masking)


class ErosionTests(unittest.TestCase):
    def test_voxel_erosion_removes_six_neighbour_layers(self):
        mask = np.ones((7, 7, 7), dtype=bool)
        for amount, expected in ((0, 343), (1, 125), (2, 27), (3, 1), (4, 0)):
            with self.subTest(amount=amount):
                result = masking.erode_binary_mask(mask, np.eye(4), erosion_voxels=amount)
                self.assertEqual(int(result.sum()), expected)
                self.assertTrue(np.all(~result | mask))
        self.assertEqual(int(mask.sum()), 343)

    def test_mm_erosion_uses_anisotropic_spacing(self):
        mask = np.ones((5, 5, 5), dtype=bool)
        affine = np.diag([1., 3., 4., 1.])
        result = masking.erode_binary_mask(mask, affine, erosion_mm=1.)
        self.assertEqual(int(result.sum()), 3 * 5 * 5)
        self.assertFalse(result[0].any())
        self.assertTrue(result[1].all())

    def test_small_mm_radius_does_not_force_voxel_layer(self):
        mask = np.ones((5, 5, 5), dtype=bool)
        result = masking.erode_binary_mask(mask, np.diag([3., 3., 3., 1.]), erosion_mm=2.)
        np.testing.assert_array_equal(result, mask)

    def test_rotated_and_reflected_mm_grid_is_supported(self):
        mask = np.ones((5, 5, 5), dtype=bool)
        affine = np.array([[0, -2., 0, 10], [2., 0, 0, 20], [0, 0, -2., 30], [0, 0, 0, 1]])
        result = masking.erode_binary_mask(mask, affine, erosion_mm=2.)
        self.assertEqual(int(result.sum()), 27)

    def test_mm_rejects_shear_but_voxel_erosion_supports_it(self):
        affine = np.eye(4)
        affine[0, 1] = .3
        mask = np.ones((5, 5, 5))
        with self.assertRaisesRegex(ValueError, 'orthogonal'):
            masking.erode_binary_mask(mask, affine, erosion_mm=1.)
        self.assertEqual(int(masking.erode_binary_mask(mask, affine, erosion_voxels=1).sum()), 27)

    def test_names_select_only_requested_roi_and_normalize_order(self):
        settings = masking.normalize_mask_settings({'erode_masks': ['wm', 'wm'], 'erosion_voxels': 1}, roi_names=['wm', 'lat'])
        self.assertEqual(settings['erode_masks'], ['wm'])
        self.assertEqual(masking.mask_erosion_request(settings, 'wm'), ('voxels', 1))
        self.assertEqual(masking.mask_erosion_request(settings, 'lat'), ('voxels', 0))
        self.assertEqual(masking.mask_erosion_request({}, 'wm')[1], 0)
        with self.assertRaisesRegex(ValueError, 'not found'):
            masking.normalize_mask_settings(settings, roi_names=['lat'])

    def test_settings_reject_ambiguous_or_invalid_values(self):
        bad = [dict(erosion_voxels=1, erosion_mm=2), dict(erosion_voxels=-1),
               dict(erosion_voxels=True), dict(erosion_voxels=1.5), dict(erosion_mm=-1),
               dict(erosion_mm=float('nan')), dict(erosion_mm=True), dict(erode_masks='wm'),
               dict(erode_masks=[None]), dict(erosion_voxel=1)]
        for settings in bad:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                masking.normalize_mask_settings(settings)

    def test_empty_input_and_nonfinite_voxels(self):
        self.assertFalse(masking.erode_binary_mask(np.zeros((5, 5, 5)), np.eye(4), erosion_voxels=1).any())
        mask = np.ones((5, 5, 5))
        mask[2, 2, 2] = np.nan
        result = masking.erode_binary_mask(mask, np.eye(4), erosion_voxels=1)
        self.assertFalse(result[2, 2, 2])
        for offset in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            self.assertFalse(result[tuple(2 + value for value in offset)])

    def test_cache_reuse_checks_effective_policy_for_each_roi(self):
        settings = {'erode_masks': ['wm'], 'erosion_voxels': 1}
        self.assertFalse(masking.erosion_policy_matches({}, settings, 'wm'))
        self.assertTrue(masking.erosion_policy_matches({}, settings, 'lat'))
        cached = {'mask_settings': settings, 'mask_roi_name': 'wm'}
        self.assertTrue(masking.erosion_policy_matches(cached, settings, 'wm'))
        self.assertFalse(masking.erosion_policy_matches(cached, {}, 'wm'))
        self.assertFalse(masking.erosion_policy_matches(cached, {'erode_masks': ['wm'], 'erosion_mm': 1}, 'wm'))


IMAGING_AVAILABLE = all(importlib.util.find_spec(name) for name in ('nibabel', 'nilearn'))


@unittest.skipUnless(IMAGING_AVAILABLE, 'Install PANIC imaging dependencies for integration tests')
class ImageTests(unittest.TestCase):
    def setUp(self):
        import nibabel as nib
        from panic.data import MaskAndFilterBetas
        self.nib = nib
        self.extract = MaskAndFilterBetas
        self.labels = dict(trial_list=['CSm', 'CSpu', 'CSm', 'CSpu'], label_mapper={'CSm': 0, 'CSpu': 1})
        self.settings = {'erode_masks': ['white_matter_forebrain'], 'erosion_voxels': 1}

    def test_selective_erosion_before_support_and_processed_mask_exports(self):
        values = np.broadcast_to(np.arange(4, dtype=np.float32), (7, 7, 7, 4)).copy()
        values[3, 3, 3] = 0
        affine = np.diag([3., 3., 3., 1.])
        betas = self.nib.Nifti1Image(values, affine)
        mask = np.zeros((7, 7, 7), dtype=np.uint8)
        mask[1:6, 1:6, 1:6] = 1
        img = self.nib.Nifti1Image(mask, affine)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'sub-017_roi-white_matter_forebrain_hemi-left_desc-valid_mask.nii.gz'
            extracted = self.extract(betas, img, roi_name='white_matter_forebrain', mask_settings=self.settings,
                                     output_file=str(output), **self.labels)
            self.assertEqual(extracted.X.shape, (4, 26))
            self.assertEqual(extracted.roi_metrics['n_voxels_before_erosion'], 125)
            self.assertEqual(extracted.roi_metrics['n_voxels_after_erosion'], 27)
            eroded = output.with_name(output.name.replace('desc-valid_mask', 'desc-eroded_mask'))
            self.assertEqual(int(self.nib.load(str(eroded)).get_fdata().sum()), 27)
            self.assertEqual(int(self.nib.load(str(output)).get_fdata().sum()), 26)
        untouched = self.extract(betas, img, roi_name='lat', mask_settings=self.settings, **self.labels)
        self.assertEqual(untouched.X.shape, (4, 124))
        self.assertFalse(untouched.roi_metrics['erosion_applied'])

    def test_voxel_iterations_apply_after_mask_resampling(self):
        values = np.broadcast_to(np.arange(4, dtype=np.float32), (5, 5, 5, 4)).copy()
        betas = self.nib.Nifti1Image(values, np.diag([2., 2., 2., 1.]))
        native = self.nib.Nifti1Image(np.ones((9, 9, 9), dtype=np.uint8), np.eye(4))
        settings = dict(self.settings, erosion_voxels=2)
        extracted = self.extract(betas, native, mask_settings=settings, roi_name='white_matter_forebrain', **self.labels)
        self.assertEqual(extracted.roi_metrics['n_voxels_before_erosion'], 125)
        self.assertEqual(extracted.X.shape, (4, 1))

    def test_empty_mask_fails_and_automatic_fov_cannot_be_eroded(self):
        from panic.errors import EmptyMaskError
        values = np.broadcast_to(np.arange(4, dtype=np.float32), (3, 3, 3, 4)).copy()
        betas = self.nib.Nifti1Image(values, np.eye(4))
        mask = self.nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), np.eye(4))
        with self.assertRaises(EmptyMaskError):
            self.extract(betas, mask, roi_name='white_matter_forebrain',
                         mask_settings=dict(self.settings, erosion_voxels=2), **self.labels)
        with self.assertRaisesRegex(ValueError, 'label-derived'):
            self.extract(betas, None, roi_name='white_matter_forebrain', mask_settings=self.settings, **self.labels)

    def test_erosion_settings_change_analysis_identity(self):
        from panic.utils import make_analysis_id
        args = dict(analysis_name='test', analysis_type='scikit_decoding', source='halfpipe', method='LSA', standardize='zscore')
        original = make_analysis_id(**args)
        self.assertEqual(original, make_analysis_id(**args, mask_settings={}))
        self.assertNotEqual(original, make_analysis_id(**args, mask_settings=self.settings))


if __name__ == '__main__':
    unittest.main()
