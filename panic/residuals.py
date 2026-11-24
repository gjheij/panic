"""Optional run-wise GLMsingle residual reader for ROI RSA.

Residuals and beta maps must share a voxel grid and affine. The caller must
supply the final beta feature indices, in their original NumPy C-order.
"""
import logging
import os
import shutil
import tempfile
from time import perf_counter
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def resolve_cache_dir(settings, existing_dir=None):
    """Resolve temporary storage; explicit override takes precedence."""
    location = settings.get("cache_location", "local")
    if location not in {"local", "disk"}:
        raise ValueError("cache_location must be local or disk")
    root = settings.get("cache_dir")
    if not root:
        root = (os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR")
                or tempfile.gettempdir()) if location == "local" else existing_dir
    root = Path(root or tempfile.gettempdir()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


DEFAULT_PATTERN = ("{beta_dir}/{subject}/task-aversivechoice_space-T1w_desc-masked/"
                   "residuals/model-typed/{subject}_task-aversivechoice_run-{run:02d}_"
                   "space-T1w_model-typed_desc-fullnuisance_residuals.nii.gz")


def resolve_run_numbers(settings, run_ids):
    """Resolve BIDS acquisition runs from beta trial groups without renumbering.

    GLMsingle beta loader parses run-XX into integer groups. Explicit run_numbers
    is allowed only when it matches those actual IDs; silent remapping is unsafe.
    """
    ids = np.asarray(run_ids).ravel()
    if not ids.size:
        raise ValueError('Cannot resolve residual runs: beta groups are empty')
    try:
        numbers = [int(x) for x in np.unique(ids)]
    except (TypeError, ValueError) as exc:
        raise ValueError('Beta run groups must contain BIDS numeric acquisition run IDs') from exc
    if any(str(raw).strip() not in (str(int(raw)), f'{int(raw):02d}') for raw in ids):
        raise ValueError('Beta run groups must be numeric run IDs, not fold/session labels')
    if any(x < 1 for x in numbers):
        raise ValueError(f'Invalid BIDS acquisition run IDs: {numbers}')
    configured = settings.get('run_numbers', 'auto')
    if configured is not None and configured != 'auto':
        configured = [int(x) for x in configured]
        if len(set(configured)) != len(configured) or set(configured) != set(numbers):
            raise ValueError(
                f'residual_settings.run_numbers={configured} does not match '
                f'actual beta acquisition runs={numbers}; use run_numbers: auto'
            )
    return numbers


def residual_paths(settings, *, beta_dir, subject, run_ids):
    """Resolve residual paths in sorted *actual* beta acquisition-run order."""
    pattern = settings.get('path_pattern', DEFAULT_PATTERN)
    numbers = resolve_run_numbers(settings, run_ids)
    paths = []
    for run in numbers:
        path = Path(pattern.format(beta_dir=beta_dir, subject=subject, run=run))
        if not path.is_file():
            raise FileNotFoundError(
                f'GLMsingle residual missing for beta acquisition run-{run:02d}: {path}'
            )
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


def residual_mask_qc(mask_img, paths, reference_img):
    """Report native ROI overlap with each run's validity mask and their intersection."""
    import nibabel as nib
    from nilearn.image import resample_to_img
    aligned = resample_to_img(mask_img, reference_img, interpolation='nearest',
                              copy_header=True, force_resample=True)
    roi = np.asarray(aligned.dataobj) > 0
    intersection = roi.copy()
    per_run = []
    for path in paths:
        img = nib.load(str(valid_mask_path(path)))
        _check_grid(img, reference_img, str(path))
        valid = np.asarray(img.dataobj) > 0
        overlap = int(np.count_nonzero(roi & valid))
        per_run.append((str(path), int(np.count_nonzero(valid)), overlap))
        intersection &= valid
    return {'roi_voxels': int(np.count_nonzero(roi)),
            'intersection': int(np.count_nonzero(intersection)), 'per_run': per_run}


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


class ResidualRunCache:
    """Subject-scoped disk-backed residual cache; no full 4D arrays held in RAM.

    Cache files are temporary uncompressed NumPy arrays indexed (time, voxel).
    Construction streams NIfTI volumes in chunks, then ROI reads reuse the
    same decompressed data. Call ``close()`` after finishing the subject.
    """

    def __init__(self, paths, reference_img, *, cache_dir=None, chunk_size=8):
        self.paths = tuple(Path(p) for p in paths)
        self.reference_img = reference_img
        self.n_space = int(np.prod(reference_img.shape[:3]))
        started = perf_counter()
        self._tempdir = tempfile.TemporaryDirectory(prefix='panic-residuals-', dir=cache_dir)
        self._arrays = []
        self._lengths = []
        try:
            self._build(int(chunk_size))
            logger.info("RSA residual cache build: directory=%s elapsed=%.3fs",
                        self._tempdir.name, perf_counter() - started)
        except BaseException:
            self.close()
            raise

    def _build(self, chunk_size):
        import nibabel as nib
        if chunk_size < 1:
            raise ValueError('chunk_size must be >= 1')
        images = [nib.load(str(path)) for path in self.paths]
        for path, img in zip(self.paths, images):
            _check_grid(img, self.reference_img, str(path))
            if len(img.shape) != 4:
                raise ValueError(f"Expected a 4D residual NIfTI: {path}")
        required = sum(img.shape[3] * self.n_space * 4 + 4096 for img in images)
        free = shutil.disk_usage(self._tempdir.name).free
        logger.info("RSA residual cache capacity: required=%.3f GiB free=%.3f GiB",
                    required / 2**30, free / 2**30)
        if required > free:
            raise OSError(f"Insufficient residual cache space: need {required} bytes, available {free}")
        for run_i, (path, img) in enumerate(zip(self.paths, images)):
            started = perf_counter()
            _check_grid(img, self.reference_img, str(path))
            if len(img.shape) != 4:
                raise ValueError(f'Expected 4D residual NIfTI: {path}')
            n_time = img.shape[3]
            out_path = Path(self._tempdir.name) / f'run-{run_i:02d}.npy'
            out = np.lib.format.open_memmap(
                out_path, mode='w+', dtype=np.float32,
                shape=(n_time, self.n_space),
            )
            for start in range(0, n_time, chunk_size):
                stop = min(start + chunk_size, n_time)
                block = np.asarray(img.dataobj[..., start:stop], dtype=np.float32)
                out[start:stop, :] = block.reshape(
                    (self.n_space, stop - start), order='C'
                ).T
            out.flush()
            del out
            self._arrays.append(np.load(out_path, mmap_mode='r'))
            self._lengths.append(n_time)
            logger.info("RSA residual cache run: file=%s shape=%s elapsed=%.3fs",
                        path.name, (n_time, self.n_space), perf_counter() - started)

    def extract(self, feature_indices, *, center_per_run=True):
        idx = np.asarray(feature_indices, dtype=int)
        if idx.ndim != 1 or idx.size == 0 or np.any(idx < 0) or np.any(idx >= self.n_space):
            raise ValueError('Invalid beta feature indices')
        started = perf_counter()
        matrices, groups = [], []
        for run_i, array in enumerate(self._arrays):
            read_started = perf_counter()
            r = np.asarray(array[:, idx], dtype=np.float64)
            logger.debug("RSA residual cache read: file=%s shape=%s elapsed=%.3fs",
                         self.paths[run_i].name, r.shape, perf_counter() - read_started)
            if not np.isfinite(r).all():
                raise ValueError(f'Nonfinite residuals in selected voxels: {self.paths[run_i]}')
            if center_per_run:
                r -= r.mean(axis=0, keepdims=True)
            matrices.append(r)
            groups.append(np.full(r.shape[0], run_i, dtype=int))
        result = np.concatenate(matrices), np.concatenate(groups)
        logger.info("RSA residual cache extract: shape=%s elapsed=%.3fs",
                    result[0].shape, perf_counter() - started)
        return result

    def close(self):
        self._arrays.clear()
        self._tempdir.cleanup()
        logger.debug("RSA residual cache removed: %s", self._tempdir.name)
