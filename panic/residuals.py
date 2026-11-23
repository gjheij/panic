"""Optional run-wise GLMsingle residual reader for ROI RSA.

Residuals and beta maps must share a voxel grid and affine. The caller must
supply the final beta feature indices, in their original NumPy C-order.
"""
from pathlib import Path
import numpy as np

DEFAULT_PATTERN = ("{beta_dir}/{subject}/task-aversivechoice_space-T1w_desc-masked/"
                   "residuals/model-typed/{subject}_task-aversivechoice_run-{run:02d}_"
                   "space-T1w_model-typed_desc-fullnuisance_residuals.nii.gz")


def residual_paths(settings, *, beta_dir, subject, run_ids):
    """Resolve run NIfTIs. Sorted group IDs map to BIDS run numbers 1..N.

    For nonconsecutive acquisition runs, supply ``run_numbers`` explicitly.
    """
    pattern = settings.get('path_pattern', DEFAULT_PATTERN)
    ids = list(np.unique(run_ids))
    numbers = settings.get('run_numbers', list(range(1, len(ids) + 1)))
    if len(numbers) != len(ids):
        raise ValueError('residual_settings.run_numbers must match the number of runs')
    paths = []
    for run in numbers:
        path = Path(pattern.format(beta_dir=beta_dir, subject=subject, run=int(run)))
        if not path.is_file():
            raise FileNotFoundError(f'GLMsingle residual file missing: {path}')
        paths.append(path)
    return paths


def valid_mask_path(path):
    """Return the validity mask corresponding to a GLMsingle residual file."""
    name = path.name
    for kind in ('fullnuisance', 'taskpoly'):
        needle = f'desc-{kind}_residuals.nii.gz'
        if needle in name:
            return path.with_name(name.replace(needle, 'desc-residualvalid_mask.nii.gz'))
    raise ValueError(f'Unrecognized GLMsingle residual filename: {path}')


def _check_grid(img, reference, name):
    if img.shape[:3] != reference.shape[:3] or not np.allclose(
        img.affine, reference.affine, atol=1e-4, rtol=0
    ):
        raise ValueError(f'{name}: image grid/affine differs from beta image')


def common_valid_mask(paths, reference_img):
    """Intersect run-wise validity masks in beta grid; require all masks."""
    import nibabel as nib
    valid = np.ones(reference_img.shape[:3], dtype=bool)
    for path in paths:
        mask_path = valid_mask_path(path)
        if not mask_path.is_file():
            raise FileNotFoundError(f'Residual validity mask missing: {mask_path}')
        img = nib.load(str(mask_path))
        _check_grid(img, reference_img, str(mask_path))
        valid &= np.asarray(img.dataobj) > 0
    return valid


def restrict_roi_mask(mask_img, valid, reference_img):
    """Resample ROI mask with nearest-neighbor interpolation and intersect validity."""
    import nibabel as nib
    from nilearn.image import resample_to_img
    aligned = resample_to_img(mask_img, reference_img, interpolation='nearest',
                              copy_header=True, force_resample=True)
    roi = (np.asarray(aligned.dataobj) > 0) & valid
    if not roi.any():
        raise ValueError('No ROI voxels survive the common residual-valid mask')
    return nib.Nifti1Image(roi.astype(np.uint8), reference_img.affine,
                           header=reference_img.header.copy())


def load_roi_residuals(paths, reference_img, feature_indices, *, center_per_run=True):
    """Extract (time, voxel) residuals in the exact beta feature order.

    Parameters
    ----------
    paths : sequence of Path
        Run-specific 4D residual NIfTIs in acquisition-run order.
    reference_img : nibabel image
        Beta image providing spatial grid and affine.
    feature_indices : 1D integer array
        C-order flattened voxel indices returned by PANIC ROI extraction.
    center_per_run : bool
        Subtract each run's per-voxel temporal mean before concatenation.

    Returns
    -------
    residuals : ndarray (sum_timepoints, n_features)
    residual_groups : ndarray (sum_timepoints,)
        Zero-based run indices corresponding to residual rows.
    """
    import nibabel as nib
    idx = np.asarray(feature_indices, dtype=int)
    n_space = int(np.prod(reference_img.shape[:3]))
    if idx.ndim != 1 or len(idx) == 0 or np.any(idx < 0) or np.any(idx >= n_space):
        raise ValueError('Invalid beta feature indices')
    matrices, groups = [], []
    for run_i, path in enumerate(paths):
        img = nib.load(str(path))
        _check_grid(img, reference_img, str(path))
        if len(img.shape) != 4:
            raise ValueError(f'Expected a 4D residual NIfTI: {path}')
        # Spatial C-order flattening matches np.flatnonzero(roi_mask.ravel()).
        arr = np.asarray(img.dataobj)
        r = arr.reshape((n_space, img.shape[3]), order='C')[idx, :].T.astype(np.float64)
        if not np.isfinite(r).all():
            raise ValueError(f'Nonfinite residuals in selected voxels: {path}')
        if center_per_run:
            r -= r.mean(axis=0, keepdims=True)
        matrices.append(r)
        groups.append(np.full(len(r), run_i, dtype=int))
    if not matrices:
        raise ValueError('No residual runs provided')
    return np.concatenate(matrices), np.concatenate(groups)
