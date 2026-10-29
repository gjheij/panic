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
