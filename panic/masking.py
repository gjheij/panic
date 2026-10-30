# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Numerical support mask shared by ROI and full-FOV beta extraction."""
import numpy as np


def beta_support_mask(betas, variance_threshold=0.0):
    """Keep voxels finite in every trial and varying across the trial axis.

    All-zero padding, constant voxels, and voxels containing NaN/Inf are excluded.
    This defines usable beta support; it does not perform anatomical brain masking.
    """
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim != 4:
        raise ValueError("Beta support requires a 4D image with trials on axis 3")
    threshold = float(variance_threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("variance_threshold must be finite and nonnegative")
    threshold = max(threshold, np.finfo(np.float32).eps)
    with np.errstate(invalid="ignore", over="ignore"):
        varying = betas.var(axis=3) > threshold
    return np.isfinite(betas).all(axis=3) & varying


def normalize_mask_settings(settings=None, roi_names=None):
    """Validate named-mask erosion settings; voxel and mm modes are exclusive."""
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise ValueError("mask_settings must be a mapping")
    unknown = set(settings) - {"erode_masks", "erosion_mm", "erosion_voxels"}
    if unknown:
        raise ValueError("Unknown mask_settings keys: " + ", ".join(sorted(unknown)))
    if "erosion_mm" in settings and "erosion_voxels" in settings:
        raise ValueError("Specify only one of erosion_voxels or erosion_mm")
    names = settings.get("erode_masks", [])
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("erode_masks must be a list of nonempty ROI names")
    if roi_names is not None:
        missing = set(names) - set(roi_names)
        if missing:
            raise ValueError("erode_masks names not found in roi_dict: " + ", ".join(sorted(missing)))
    result = {"erode_masks": sorted(set(names))}
    if "erosion_voxels" in settings:
        value = settings["erosion_voxels"]
        if type(value) is not int or value < 0:
            raise ValueError("erosion_voxels must be a nonnegative integer")
        result["erosion_voxels"] = value
    else:
        value = settings.get("erosion_mm", 0.0)
        if isinstance(value, bool):
            raise ValueError("erosion_mm must be finite and nonnegative")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("erosion_mm must be finite and nonnegative") from None
        if not np.isfinite(value) or value < 0:
            raise ValueError("erosion_mm must be finite and nonnegative")
        result["erosion_mm"] = value
    return result


def mask_erosion_request(settings=None, roi_name=None):
    settings = normalize_mask_settings(settings)
    selected = roi_name in settings.get("erode_masks", [])
    if "erosion_voxels" in settings:
        return ("voxels", settings["erosion_voxels"]) if selected else ("voxels", 0)
    return ("mm", settings.get("erosion_mm", 0.0)) if selected else ("mm", 0.0)


def erode_binary_mask(mask, affine, *, erosion_voxels=None, erosion_mm=None):
    """Erode a 3D binary mask on its current grid, treating exterior as background.

    Voxel mode uses iterations of six-neighbour binary erosion. Millimetre mode
    uses Euclidean distance to background voxel centers on an orthogonal grid.
    No minimum layer is forced: a small mm radius can legitimately do nothing.
    """
    if erosion_voxels is not None and erosion_mm is not None:
        raise ValueError("Specify only one erosion mode")
    request = ({"erosion_voxels": erosion_voxels} if erosion_voxels is not None
               else {"erosion_mm": 0.0 if erosion_mm is None else erosion_mm})
    validated = normalize_mask_settings(request)
    mask = np.asarray(mask)
    if mask.ndim != 3:
        raise ValueError("Erosion requires a 3D binary mask")
    mask = np.isfinite(mask) & (mask > 0.5)
    amount = validated.get("erosion_voxels", validated.get("erosion_mm", 0))
    if amount == 0 or not mask.any():
        return mask.copy()
    from scipy import ndimage

    if erosion_voxels is not None:
        return ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1),
                                     iterations=amount, border_value=0)
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Erosion requires a finite 4x4 affine")
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    if np.any(spacing <= 0):
        raise ValueError("Erosion requires positive voxel sizes")
    directions = basis / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), rtol=1e-5, atol=1e-5):
        raise ValueError("Millimetre erosion requires orthogonal voxel axes; use voxel erosion for a sheared grid")
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    distances = ndimage.distance_transform_edt(padded, sampling=spacing)[1:-1, 1:-1, 1:-1]
    return mask & (distances > amount)


def erosion_policy_matches(metadata, settings, roi_name):
    """Whether cached maps used the same effective erosion for this ROI."""
    try:
        previous = mask_erosion_request(metadata.get("mask_settings"), metadata.get("mask_roi_name", roi_name))
        current = mask_erosion_request(settings, roi_name)
    except (ValueError, TypeError):
        return False
    return previous == current or (previous[1] == 0 and current[1] == 0)
