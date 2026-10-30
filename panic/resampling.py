"""Voxel-size resampling that preserves input affine directions and origin."""
import numpy as np


def _voxel_size_grid(shape, affine, target_vox_size):
    """Return a grid covering the original voxel-center bounding box."""
    target = np.asarray(target_vox_size, dtype=float)
    if target.ndim == 0:
        target = np.repeat(target, 3)
    if target.shape != (3,) or not np.isfinite(target).all() or np.any(target <= 0):
        raise ValueError("target_vox_size must be a positive finite scalar or three voxel sizes")
    shape = np.asarray(shape[:3], dtype=float)
    if shape.shape != (3,) or not np.isfinite(shape).all() or np.any(shape < 1) or np.any(shape != np.floor(shape)):
        raise ValueError("Image must have three positive spatial dimensions")
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Image affine must be a finite 4x4 matrix")
    if not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError("Image affine must have homogeneous last row [0, 0, 0, 1]")
    basis = affine[:3, :3]
    current = np.linalg.norm(basis, axis=0)
    if np.any(current <= 0) or np.linalg.matrix_rank(basis) < 3:
        raise ValueError("Image affine must have three independent spatial axes")
    new_affine = affine.copy()
    # Scaling columns keeps rotations, axis permutations, reflections, shear,
    # and the voxel-zero world coordinate. Diagonal signs cannot represent these.
    new_affine[:3, :3] = basis * (target / current)
    extent = (shape - 1) * current / target
    # Avoid spurious padding from float32 affine round-off at integer extents.
    nearest = np.rint(extent)
    extent = np.where(np.isclose(extent, nearest, rtol=1e-7, atol=1e-7), nearest, extent)
    target_shape = tuple(int(value) for value in np.ceil(extent) + 1)
    return new_affine, target_shape


def resample_to_voxel_size(img, target_vox_size, interpolation="continuous"):
    """Resample a 3D/4D Niimg to explicit spatial voxel sizes in affine units.

    Parameters
    ----------
    img : Niimg-like
        A NIfTI image or filename. NIfTI affines are normally in millimetres.
    target_vox_size : float or sequence of three floats
        Positive target spatial voxel sizes. A scalar requests isotropic voxels.
    interpolation : {'continuous', 'linear', 'nearest'}, default='continuous'
        Nilearn interpolation: continuous is cubic, linear is trilinear, and
        nearest preserves categorical label values.

    Notes
    -----
    Preserves the full axis directions and original affine translation. The
    explicit target shape covers the original voxel-center bounds, allowing a
    boundary layer of zero padding instead of cropping centers. The time/trial
    axis of a 4D image is not resampled. Use nearest-neighbor interpolation for
    anatomical label images. Masks used by PANIC are subsequently aligned to
    the resampled beta grid, rather than resampled on an independent grid.
    """
    if interpolation not in ("continuous", "linear", "nearest"):
        raise ValueError("interpolation must be continuous, linear, or nearest")
    from nilearn.image import load_img, resample_img

    img = load_img(img)
    if len(img.shape) not in (3, 4):
        raise ValueError("Voxel-size resampling requires a 3D or 4D image")
    target_affine, target_shape = _voxel_size_grid(img.shape, img.affine, target_vox_size)
    return resample_img(
        img,
        target_affine=target_affine,
        target_shape=target_shape,
        interpolation=interpolation,
        fill_value=0,
        force_resample=True,
        copy_header=True,
    )
