"""Synthetic FOV-mask regression tests; imaging tests need PANIC dependencies."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

# Load the numerical helper without PANIC's package-wide imaging imports.
source = Path(__file__).resolve().parents[1] / "panic" / "masking.py"
spec = importlib.util.spec_from_file_location("panic_masking_support", source)
masking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(masking)


class SupportTests(unittest.TestCase):
    def test_finite_varying_support_excludes_padding_and_constants(self):
        values = np.zeros((2, 2, 2, 4), dtype=np.float32)
        values[0, 0, 0] = [0, 1, 2, 3]
        values[0, 0, 1] = [-1, 0, 1, 0]
        values[0, 1, 0] = [5, 5, 5, 5]
        support = masking.beta_support_mask(values)
        self.assertEqual(set(map(tuple, np.argwhere(support))), {(0, 0, 0), (0, 0, 1)})

    def test_any_nonfinite_trial_excludes_voxel(self):
        values = np.array([[[[0, 1, np.nan], [0, 1, np.inf], [0, 1, -np.inf], [0, 1, 2]]]])
        np.testing.assert_array_equal(masking.beta_support_mask(values), [[[False, False, False, True]]])

    def test_empty_support_is_representable_for_extractor_to_reject(self):
        self.assertFalse(masking.beta_support_mask(np.zeros((2, 2, 2, 4))).any())

    def test_configured_threshold_and_float32_floor(self):
        values = np.array([[[[0, 1, 0, 1], [0, 1e-6, 0, 1e-6]]]])
        np.testing.assert_array_equal(masking.beta_support_mask(values), [[[True, False]]])
        self.assertFalse(masking.beta_support_mask(values, variance_threshold=0.3).any())

    def test_invalid_threshold_rejected(self):
        for threshold in (-1, float('nan'), float('inf')):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                masking.beta_support_mask(np.zeros((2, 2, 2, 4)), threshold)

    def test_non_4d_input_rejected(self):
        with self.assertRaisesRegex(ValueError, '4D'):
            masking.beta_support_mask(np.zeros((2, 2, 2)))


IMAGING_AVAILABLE = all(importlib.util.find_spec(name) for name in ('nibabel', 'nilearn'))


@unittest.skipUnless(IMAGING_AVAILABLE, 'Install PANIC imaging dependencies for image integration tests')
class ImageIntegrationTests(unittest.TestCase):
    def setUp(self):
        import nibabel as nib
        from panic.data import MaskAndFilterBetas
        self.extract = MaskAndFilterBetas
        self.nib = nib
        values = np.zeros((3, 3, 3, 4), dtype=np.float32)
        values[1, 1, 1] = [0, 1, 2, 3]
        values[1, 1, 2] = [-1, 0, 1, 0]
        values[0, 0, 0] = [1, 2, np.nan, 4]
        self.values = values
        self.betas = nib.Nifti1Image(values, np.diag([2., 2., 2., 1.]))
        self.arguments = dict(trial_list=['CSm', 'CSpu', 'CSm', 'CSpu'], label_mapper={'CSm': 0, 'CSpu': 1})

    def test_no_mask_extracts_only_beta_support_on_original_grid(self):
        extracted = self.extract(self.betas, None, **self.arguments)
        self.assertEqual(extracted.X.shape, (4, 2))
        np.testing.assert_array_equal(extracted.fov_mask.affine, self.betas.affine)
        np.testing.assert_array_equal(extracted.fov_mask.get_fdata() > 0, masking.beta_support_mask(self.values))
        self.assertTrue(np.isfinite(extracted.X).all())

    def test_explicit_mask_still_restricts_support(self):
        mask = np.zeros((3, 3, 3), dtype=np.uint8)
        mask[1, 1, 1] = 1
        extracted = self.extract(self.betas, self.nib.Nifti1Image(mask, self.betas.affine), **self.arguments)
        self.assertEqual(extracted.X.shape, (4, 1))

    def test_no_valid_voxels_raises(self):
        from panic.errors import EmptyMaskError
        empty = self.nib.Nifti1Image(np.zeros((3, 3, 3, 4), dtype=np.float32), self.betas.affine)
        with self.assertRaises(EmptyMaskError):
            self.extract(empty, None, **self.arguments)

    def test_target_zooms_create_mask_on_resampled_grid(self):
        from nilearn import image
        extracted = self.extract(self.betas, None, zooms=[1., 1., 1.], **self.arguments)
        target = image.index_img(extracted.betas, 0)
        self.assertEqual(extracted.fov_mask.shape, target.shape)
        np.testing.assert_array_equal(extracted.fov_mask.affine, target.affine)

    def test_decoder_routes_null_only_for_searchlights(self):
        from panic.decode import ClassifySubject
        decoder = ClassifySubject.__new__(ClassifySubject)
        decoder.cfg = {'roi_dict': None}
        decoder.searchlight = True
        self.assertEqual(decoder.define_mask_inputs(), ({'fov': None}, False))
        self.assertEqual(decoder.prepare_rois(None), (None, {'uni': ['fov', None]}))
        decoder.searchlight = False
        with self.assertRaises(ValueError):
            decoder.define_mask_inputs()
        with self.assertRaises(ValueError):
            decoder.prepare_rois(None)


if __name__ == '__main__':
    unittest.main()
