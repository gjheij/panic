"""Explicit voxel-grid tests; real NIfTI tests require imaging dependencies."""
import importlib.util
import itertools
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

source = Path(__file__).resolve().parents[1] / "panic" / "resampling.py"
spec = importlib.util.spec_from_file_location("panic_resampling_geometry", source)
resampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resampling)


class GridTests(unittest.TestCase):
    def test_axis_aligned_affine_preserves_origin_and_reflections(self):
        affine = np.diag([-3., 2., 4., 1.])
        affine[:3, 3] = [80., -50., 30.]
        new, shape = resampling._voxel_size_grid((11, 12, 13), affine, [2, 2, 2])
        np.testing.assert_allclose(new[:3, :3], np.diag([-2., 2., 2.]))
        np.testing.assert_array_equal(new[:3, 3], affine[:3, 3])
        self.assertEqual(shape, (16, 12, 25))

    def test_rotated_and_sheared_affines_keep_all_axis_directions(self):
        angle = np.deg2rad(31.)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        for shear in (np.eye(3), np.array([[1., .2, 0], [0, 1, .1], [0, 0, 1]])):
            affine = np.eye(4)
            affine[:3, :3] = rotation @ shear @ np.diag([2., 3., -4.])
            affine[:3, 3] = [13, -21, 6]
            new, _ = resampling._voxel_size_grid((20, 30, 40), affine, (1., 2., 3.))
            current = np.linalg.norm(affine[:3, :3], axis=0)
            np.testing.assert_allclose(new[:3, :3] / [1., 2., 3.], affine[:3, :3] / current)
            np.testing.assert_allclose(np.linalg.norm(new[:3, :3], axis=0), [1., 2., 3.])
            np.testing.assert_array_equal(new[:3, 3], affine[:3, 3])

    def test_axis_permutations_with_zero_diagonal_do_not_collapse(self):
        affine = np.array([[0., -3., 0., 10.], [2., 0., 0., 20.], [0., 0., 4., 30.], [0., 0., 0., 1.]])
        new, _ = resampling._voxel_size_grid((10, 12, 14), affine, 2)
        self.assertNotEqual(np.linalg.det(new[:3, :3]), 0)
        np.testing.assert_allclose(new[:3, :3], [[0, -2, 0], [2, 0, 0], [0, 0, 2]])

    def test_all_source_center_corners_fit_in_target_grid(self):
        affine = np.array([[-3.111, .5, 0., 40.], [0., 3.111, .2, -80.], [0., 0., 3.54, 20.], [0., 0., 0., 1.]])
        shape = (50, 58, 42)
        for zooms in ((4., 4., 4.), (2., 2., 2.), (.9, 8., 3.)):
            new, target_shape = resampling._voxel_size_grid(shape, affine, zooms)
            mapping = np.linalg.inv(new) @ affine
            for corner in itertools.product(*[(0, n - 1) for n in shape]):
                target = (mapping @ np.r_[corner, 1])[:3]
                self.assertTrue(np.all(target >= -1e-6))
                self.assertTrue(np.all(target <= np.array(target_shape) - 1 + 1e-5))

    def test_identity_zoom_and_float32_roundoff_do_not_add_voxels(self):
        affine = np.diag([1.2, 2.5, 3.1, 1.]).astype(np.float32)
        new, shape = resampling._voxel_size_grid((31, 41, 51), affine, [1.2, 2.5, 3.1])
        self.assertEqual(shape, (31, 41, 51))
        np.testing.assert_allclose(new, affine, rtol=1e-6)

    def test_scalar_tuple_and_singleton_axes(self):
        _, shape = resampling._voxel_size_grid((1, 3, 5), np.eye(4), 100.)
        self.assertEqual(shape, (1, 2, 2))
        self.assertEqual(resampling._voxel_size_grid((4, 4, 4), np.eye(4), 2)[1],
                         resampling._voxel_size_grid((4, 4, 4), np.eye(4), (2, 2, 2))[1])

    def test_invalid_target_sizes_are_rejected(self):
        for value in (0, -2, [1, 2], [1, 2, 3, 4], [1, 0, 2], [1, np.nan, 2], [1, np.inf, 2], [[1, 2, 3]]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resampling._voxel_size_grid((2, 2, 2), np.eye(4), value)

    def test_invalid_affines_and_shapes_are_rejected(self):
        for affine in (np.zeros((4, 4)), np.eye(3), np.diag([np.nan, 1, 1, 1]), np.diag([0, 1, 1, 1])):
            with self.subTest(affine=affine), self.assertRaises(ValueError):
                resampling._voxel_size_grid((2, 2, 2), affine, 2)
        for shape in ((0, 2, 2), (2, 2), (2.5, 2, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                resampling._voxel_size_grid(shape, np.eye(4), 2)

    def test_wrapper_passes_explicit_grid_and_interpolation_to_nilearn(self):
        img = SimpleNamespace(shape=(11, 12, 13, 7), affine=np.diag([3., 2., 4., 1.]))
        nilearn = ModuleType('nilearn')
        image = ModuleType('nilearn.image')
        image.load_img = Mock(return_value=img)
        image.resample_img = Mock(return_value='result')
        with patch.dict(sys.modules, {'nilearn': nilearn, 'nilearn.image': image}):
            result = resampling.resample_to_voxel_size('input.nii.gz', (2, 2, 2), interpolation='linear')
        self.assertEqual(result, 'result')
        image.load_img.assert_called_once_with('input.nii.gz')
        kwargs = image.resample_img.call_args.kwargs
        self.assertEqual(kwargs['target_shape'], (16, 12, 25))
        self.assertEqual(kwargs['interpolation'], 'linear')
        self.assertTrue(kwargs['force_resample'])
        self.assertTrue(kwargs['copy_header'])
        self.assertEqual(kwargs['fill_value'], 0)


IMAGING_AVAILABLE = all(importlib.util.find_spec(name) for name in ('nibabel', 'nilearn'))


@unittest.skipUnless(IMAGING_AVAILABLE, 'Install nibabel and nilearn for image integration tests')
class ImageTests(unittest.TestCase):
    def test_4d_beta_resampling_keeps_trials_and_temporal_zoom(self):
        import nibabel as nib
        values = np.zeros((9, 11, 13, 5), dtype=np.float32)
        values[2:7, 2:9, 2:11] = np.arange(1, 6)
        img = nib.Nifti1Image(values, np.diag([2., 3., 4., 1.]))
        img.header.set_zooms((2., 3., 4., 2.4))
        result = resampling.resample_to_voxel_size(img, [3., 3., 3.], 'linear')
        self.assertEqual(result.shape, (7, 11, 17, 5))
        np.testing.assert_allclose(result.header.get_zooms(), [3., 3., 3., 2.4])
        np.testing.assert_allclose(result.affine[:3, 3], img.affine[:3, 3])

    def test_nearest_label_resampling_preserves_label_set(self):
        import nibabel as nib
        values = np.zeros((6, 7, 8), dtype=np.int16)
        values[1:4, 1:5, 1:6] = 216
        values[4:, 3:, 3:] = 377
        img = nib.Nifti1Image(values, np.diag([-2., 3., 4., 1.]))
        result = resampling.resample_to_voxel_size(img, (1, 2, 2), 'nearest')
        self.assertTrue(set(np.unique(result.get_fdata())).issubset({0, 216, 377}))

    def test_oblique_image_orientation_is_preserved(self):
        import nibabel as nib
        affine = np.array([[0., -3., 0., 50.], [2., 0., 0., -20.], [0., 0., 4., 10.], [0., 0., 0., 1.]])
        img = nib.Nifti1Image(np.ones((6, 7, 8), dtype=np.float32), affine)
        result = resampling.resample_to_voxel_size(img, 2, 'nearest')
        self.assertEqual(nib.aff2axcodes(result.affine), nib.aff2axcodes(img.affine))
        np.testing.assert_allclose(result.header.get_zooms(), [2, 2, 2])


if __name__ == '__main__':
    unittest.main()
