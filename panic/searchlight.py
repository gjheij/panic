# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import json
import os
import tempfile
import uuid

import numpy as np
from joblib import Parallel, delayed, dump
from nilearn import image
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

from panic import data
from panic.logger import get_logger
from panic.monitor import (
    SearchlightMonitor,
    install_faulthandler,
    maybe_log_worker_resources,
    worker_heartbeat,
    worker_mark_completed,
    worker_task_started,
)
from panic.pipeline import create_outer_folds
from panic.plugins import core
from panic.utils import tqdm_disabled, load_feature_matrix

logger = get_logger(__name__)
opj = os.path.join


def _cols_for_center(center_ijk, offsets, col_index_vol, vol_shape):
    cx, cy, cz = map(int, center_ijk)

    cols = []
    for dx, dy, dz in offsets:
        x, y, z = cx + dx, cy + dy, cz + dz

        if (
            0 <= x < vol_shape[0]
            and 0 <= y < vol_shape[1]
            and 0 <= z < vol_shape[2]
        ):
            col = col_index_vol[x, y, z]
            if col >= 0:
                cols.append(col)

    return cols


def _voxel_radius_in_voxels(mask_img, radius_mm):
    """
    Convert a spherical searchlight radius from millimeters to voxel units.

    This helper estimates the equivalent radius in voxel space by dividing
    the given physical radius (in mm) by the voxel size along the x-axis.
    It assumes approximately isotropic voxel dimensions and falls back to
    reasonable defaults if voxel size metadata is unavailable.

    Parameters
    ----------
    mask_img : nibabel.Nifti1Image
        A 3D NIfTI image (or compatible object) whose header contains voxel
        dimension information in ``pixdim`` or via ``header.get_zooms()``.
    radius_mm : float
        Searchlight radius expressed in millimeters.

    Returns
    -------
    int
        The radius converted to voxel units, rounded to the nearest integer.
        A minimum of 1 voxel is enforced.

    Notes
    -----
    - Attempts to extract voxel dimensions using
      ``mask_img.header.get_zooms()[:3]``.
    - If unavailable or malformed, falls back to using the array shape
      of the image data (via ``image.get_data(mask_img).shape[:3]``).
    - Uses only the first element of the voxel size (x-axis spacing)
      to compute an approximately isotropic voxel radius:
      ``radius_vox = round(radius_mm / voxel_size_x)``.
    - Returns at least 1 to avoid a zero-radius searchlight.
    - Assumes roughly isotropic voxel dimensions; if voxel sizes differ
      substantially across axes, the result is only an approximation.
    - Designed for compatibility with fMRI searchlight decoding pipelines.

    Examples
    --------
    >>> import nibabel as nib
    >>> mask_img = nib.load("roi_mask.nii.gz")
    >>> r_vox = _voxel_radius_in_voxels(mask_img, radius_mm=6.0)
    >>> print(f"Searchlight radius ≈ {r_vox} voxels")
    """
    hdr = mask_img.header
    zooms = np.array(image.get_data(mask_img).shape[:3], dtype=float)  # fallback if header missing pixdim
    try:
        zooms = np.array(mask_img.header.get_zooms()[:3], dtype=float)
    except Exception:
        pass
    vx = float(zooms[0])
    return max(1, int(round(radius_mm / vx)))


def _neighbors_ball_mm(zooms, r_mm):
    """
    Compute voxel offset coordinates forming a 3D spherical neighborhood
    of a given physical radius (in millimeters).

    This function enumerates all voxel offsets (Δx, Δy, Δz) that lie within
    a sphere of radius ``r_mm`` when accounting for anisotropic voxel
    dimensions. It returns an array of integer voxel offsets suitable for
    constructing searchlight neighborhoods in fMRI decoding analyses.

    Parameters
    ----------
    zooms : sequence of float
        Sequence of voxel dimensions (in mm) along the x, y, and z axes,
        typically obtained from a NIfTI image header via
        ``mask_img.header.get_zooms()[:3]``.
        Example: ``(3.5, 3.75, 5.0)``.
    r_mm : float
        Searchlight radius in millimeters.

    Returns
    -------
    numpy.ndarray
        Array of integer voxel offsets ``(dx, dy, dz)`` such that the
        Euclidean distance (in mm) from the origin does not exceed ``r_mm``.
        Shape is ``(n_neighbors, 3)``.

    Notes
    -----
    - The voxel-space bounding box is determined by dividing the
      millimeter radius by voxel sizes along each axis:
      ``rx = int(r_mm // vx)``, etc.
    - All integer offset combinations within
      ``[-rx, rx] × [-ry, ry] × [-rz, rz]`` are tested for inclusion using
      the distance criterion:
      ``(dx*vx)**2 + (dy*vy)**2 + (dz*vz)**2 <= r_mm**2``.
    - Returns offsets as a NumPy integer array.
    - The output represents **relative** voxel coordinates; to obtain
      absolute indices in an image, add these offsets to a voxel’s (i, j, k)
      coordinates.
    - Accounts for anisotropic voxel dimensions by scaling distances
      according to physical spacing.
    - Designed for use in searchlight mapping and neighborhood-based analyses.

    Examples
    --------
    >>> zooms = (3.5, 3.75, 5.0)
    >>> r_mm = 6.0
    >>> offsets = _neighbors_ball_mm(zooms, r_mm)
    >>> print(f"{len(offsets)} neighbors within {r_mm} mm radius")
    """
    logger.info(f"Compute voxel offset coordinates | zoom = {zooms} | radius = {r_mm}")
    vx, vy, vz = map(float, zooms[:3])     # e.g. (3.5, 3.75, 5.0)
    rx, ry, rz = int(r_mm // vx), int(r_mm // vy), int(r_mm // vz)
    offs = []
    r2 = r_mm * r_mm
    for dx in range(-rx, rx + 1):
        for dy in range(-ry, ry + 1):
            for dz in range(-rz, rz + 1):
                if (dx * vx) ** 2 + (dy * vy) ** 2 + (dz * vz) ** 2 <= r2:
                    offs.append((dx, dy, dz))
    return np.asarray(offs, dtype=int)


def _one_center(
        center_ijk,
        cols,
        X_mm_path,
        labels,
        folds,
        cfg,
        groups,
        n_perms,
        seed,
        *,
        plugin,
        plugin_kwargs,
        monitor_runtime_dir=None,
        monitor_ix=None,
        **kwargs,
    ):
    """
    Compute observed and permutation-based decoding accuracy for a single
    searchlight center voxel.

    This function performs decoding at one spatial location by extracting
    the neighborhood of voxels (as defined by ``offsets``), running
    cross-validated decoding on that subset of features, and computing both
    the observed and null (permutation) performance estimates. It returns
    a compact tuple summarizing key results for the given center voxel,
    suitable for aggregation across the searchlight volume.

    Parameters
    ----------
    center_ijk : tuple of int
        Integer voxel coordinates ``(i, j, k)`` of the current searchlight center.
    offsets : numpy.ndarray
        Array of voxel offset coordinates ``(dx, dy, dz)`` defining the
        spherical neighborhood (e.g., output of :func:`_neighbors_ball_mm`).
    col_index_vol : numpy.ndarray
        3D integer array mapping voxel coordinates to feature indices within
        the ROI feature space. Entries < 0 indicate non-ROI voxels.
    vol_shape : tuple of int
        Shape of the full 3D brain volume (e.g., from the mask image).
    X_mm_path : str
        Path to a ``joblib`` dump containing the full memory-mapped feature
        matrix of shape ``(n_samples, n_features_all_roi)``.
    labels : array_like
        Array of target labels for decoding, shape ``(n_samples,)``.
    folds : list of tuple
        List of outer cross-validation splits as tuples ``(train_idx, test_idx)``.
    cfg : dict
        Configuration dictionary for decoding, passed to :func:`_cv_mean_score`.
    groups : array_like
        Optional group labels of shape ``(n_samples,)`` for group-aware CV
        or within-group permutations.
    n_perms : int
        Number of label permutations to run for estimating the null distribution.
    seed : int
        Random seed controlling reproducibility of permutation sampling.
    **kwargs : dict, optional
        Additional keyword arguments forwarded to :func:`_cv_mean_score`.

    Returns
    -------
    tuple
        A 9-element tuple containing:
        
        1. ``center_ijk`` : tuple of int  
           Voxel coordinates.
        2. ``obs`` : float  
           Observed mean cross-validated score.
        3. ``null_mean`` : float  
           Mean score under permutation null.
        4. ``delta`` : float  
           Difference between observed and null mean.
        5. ``p`` : float  
           Empirical p-value computed as
           ``(sum(perm >= obs) + 1) / (len(perm) + 1)``.
        6. ``n_feat`` : int  
           Number of voxels included in the neighborhood.
        7. ``n_run`` : int  
           Number of permutation runs actually executed. In null-mean mode this equals ``n_perms``.
        8. ``stopped`` : int  
           Flag indicating if early stopping occurred (always 0 in null-mean searchlight mode).
        9. ``stop_code`` : int  
           Encoded reason for stopping (always 0 in null-mean searchlight mode).

    Notes
    -----
    - Identifies valid voxel neighbors within the searchlight radius
      using ``offsets`` and ``col_index_vol``.
    - Skips computation if fewer than 2 valid features are available.
    - Uses the relevant feature subset from the shared memmapped matrix without writing per-center temporary arrays.
    - Computes the observed decoding score via :func:`utils._cv_mean_score`.
    - Performs ``n_perms`` permutation runs with an independent RNG initialized from ``seed``.
    - Aggregates permutation scores and computes ``null_mean``, ``delta``,
      and empirical ``p``. The p-value is retained as a diagnostic; Bach-style searchlight inference should generally use the delta map at the group level.
    - The empirical p-value is bias-corrected using a +1 numerator and denominator adjustment when fixed permutations are available.
    - Uses :func:`numpy.random.default_rng` for reproducible random sampling.
    - Designed for internal use within parallelized searchlight loops.

    Examples
    --------
    >>> center = (32, 40, 20)
    >>> offsets = _neighbors_ball_mm((3.5, 3.5, 3.5), 6.0)
    >>> result = _one_center(
    ...     center_ijk=center,
    ...     offsets=offsets,
    ...     col_index_vol=col_index_vol,
    ...     vol_shape=(64, 64, 36),
    ...     roi_linidx=roi_idx,
    ...     X_mm_path="/tmp/X_roi.joblib",
    ...     labels=y,
    ...     folds=folds,
    ...     cfg=cfg,
    ...     groups=runs,
    ...     n_perms=100,
    ...     seed=1234
    ... )
    >>> print(result)
    ((32, 40, 20), 0.71, 0.50, 0.21, 0.01, 87, 100, 0, 0)
    """

    cx, cy, cz = map(int, center_ijk)

    if len(cols) < 2:
        return (cx, cy, cz), np.nan, np.nan, np.nan, np.nan, int(len(cols)), 0, 0, 0

    # read data
    worker_heartbeat(
        monitor_runtime_dir,
        monitor_ix if monitor_ix is not None else -1,
        "BEFORE_LOAD",
        ncols=int(len(cols)),
    )
    X_center = load_feature_matrix(X_mm_path, cols=cols)
    worker_heartbeat(
        monitor_runtime_dir,
        monitor_ix if monitor_ix is not None else -1,
        "AFTER_LOAD",
        ncols=int(len(cols)),
        shape=list(X_center.shape),
    )

    # read settings
    analysis_cfg = cfg.get("analysis", {})
    do_permutations = bool(analysis_cfg.get("permutations", True))
    higher_is_better = bool(analysis_cfg.get("higher_is_better", True))

    # observed
    output_kind = cfg.get("analysis", {}).get("output_kind", "scalar")

    result = plugin(
        X_center,
        labels,
        cfg=cfg,
        folds=folds,
        cols=None,
        permute=False,
        return_artifacts=(output_kind == "timeseries"),
        **plugin_kwargs,
        **kwargs,
    )

    obs, artifacts = core.unpack_plugin_result(result)
    worker_heartbeat(
        monitor_runtime_dir,
        monitor_ix if monitor_ix is not None else -1,
        "AFTER_OBS",
        ncols=int(len(cols)),
    )

    if output_kind == "timeseries":
        ts = artifacts.get("cs_us_similarity", None)
    else:
        ts = None

    null_sum = 0.0
    count_extreme = 0
    n_run = 0

    if do_permutations and n_perms > 0:
        center_rng = np.random.default_rng(int(seed))
        seeds = center_rng.integers(0, 2**32 - 1, size=n_perms, dtype=np.uint32)

        for perm_ix, s in enumerate(seeds):
            perm_seed = int(s)
            worker_heartbeat(
                monitor_runtime_dir,
                monitor_ix if monitor_ix is not None else -1,
                f"BEFORE_PERM_{perm_ix}",
                ncols=int(len(cols)),
                center_seed=int(seed),
                perm_ix=int(perm_ix),
                perm_seed=perm_seed,
                n_perms=int(n_perms),
            )

            v = float(
                plugin(
                    X_center,
                    labels,
                    cfg=cfg,
                    folds=folds,
                    cols=None,
                    rng=np.random.default_rng(perm_seed),
                    permute=True,
                    **plugin_kwargs,
                    **kwargs,
                )
            )

            worker_heartbeat(
                monitor_runtime_dir,
                monitor_ix if monitor_ix is not None else -1,
                f"AFTER_PERM_{perm_ix}",
                ncols=int(len(cols)),
                center_seed=int(seed),
                perm_ix=int(perm_ix),
                perm_seed=perm_seed,
                score=float(v),
                n_perms=int(n_perms),
            )

            null_sum += v

            if higher_is_better:
                count_extreme += int(v >= obs)
            else:
                count_extreme += int(v <= obs)

            n_run += 1

    null_mean = float(null_sum / n_run) if n_run else np.nan
    p = float((count_extreme + 1.0) / (n_run + 1.0)) if n_run else np.nan
    delta = float(obs - null_mean) if np.isfinite(null_mean) else np.nan

    stopped = 0
    stop_code = 0

    return (
        (cx, cy, cz),
        float(obs),
        float(null_mean),
        float(delta),
        float(p),
        int(len(cols)),
        int(n_run),
        stopped,
        stop_code,
        ts
    )


def _run_searchlight_center(
        ix,
        centers,
        offsets,
        col_index_vol,
        vol_shape,
        X_path,
        labels,
        folds,
        cfg,
        groups,
        n_perms,
        center_seeds,
        plugin,
        plugin_kwargs,
        locked_params,
        save_dir,
        kwargs,
        monitor_runtime_dir=None,
        monitor_resource_every=1000,
    ):
    """Run one searchlight center from a module-level, picklable worker.

    This wrapper is intentionally defined at module scope so it can be used by
    both Joblib's ``loky`` backend and the standard ``multiprocessing`` backend.
    The latter cannot pickle functions defined locally inside
    :func:`permutation_searchlight`.
    """
    install_faulthandler()
    task_count = worker_task_started()
    center_ijk = tuple(map(int, centers[ix]))
    worker_heartbeat(
        monitor_runtime_dir,
        ix,
        "START",
        center_ijk=list(center_ijk),
        center_seed=int(center_seeds[ix]),
        task_count=int(task_count),
    )

    try:
        cols = _cols_for_center(
            center_ijk,
            offsets,
            col_index_vol,
            vol_shape,
        )
        worker_heartbeat(
            monitor_runtime_dir,
            ix,
            "COLS_READY",
            center_ijk=list(center_ijk),
            ncols=int(len(cols)),
            task_count=int(task_count),
        )

        result = _one_center(
            center_ijk,
            cols,
            X_path,
            labels,
            folds,
            cfg,
            groups=groups,
            n_perms=n_perms,
            seed=int(center_seeds[ix]),
            plugin=plugin,
            plugin_kwargs=plugin_kwargs,
            locked=locked_params,
            searchlight=True,
            save_dir=save_dir,
            monitor_runtime_dir=monitor_runtime_dir,
            monitor_ix=ix,
            **kwargs,
        )

        worker_heartbeat(
            monitor_runtime_dir,
            ix,
            "DONE",
            center_ijk=list(center_ijk),
            ncols=int(len(cols)),
            task_count=int(task_count),
        )
        maybe_log_worker_resources(
            logger,
            task_count=task_count,
            every=int(monitor_resource_every),
            x_path=X_path,
        )
        worker_mark_completed(monitor_runtime_dir, ix)
        return int(ix), result

    except BaseException as exc:
        worker_heartbeat(
            monitor_runtime_dir,
            ix,
            "ERROR",
            center_ijk=list(center_ijk),
            task_count=int(task_count),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


def permutation_searchlight(
        betas_img,            # 4D betas (nifti)
        mask_img,             # binary ROI/brain mask (nifti)
        trial_list,           # list[str] or array[str] per volume in betas
        label_mapper,         # dict like {'CS-':0,'CS+':1}
        cfg,
        *,
        groups=None,          # run indices per trial (optional)
        seed=0,
        tmpdir=None,
        output_file=None,
        hemi_key=None,
        **kwargs
    ):
    """
    Run a permutation-based searchlight decoding analysis and write result maps.

    This function computes, for each voxel center inside the ROI, the observed
    cross-validated decoding score and a permutation-based null model, then
    assembles NIfTI maps of observed score, null mean, delta (observed − null),
    diagnostic empirical p-value, and number of features used. Work can be parallelized
    across centers.

    Parameters
    ----------
    betas_img : nibabel.Nifti1Image
        4D beta image with shape ``(X, Y, Z, n_samples)``.
    mask_img : nibabel.Nifti1Image
        Binary ROI (or brain) mask aligned to ``betas_img`` space
        (or resampled to it).
    trial_list : sequence of str
        Trial descriptors aligned with the 4th dimension of ``betas_img``.
        Used by :class:`data.MaskAndFilterBetas` to select and order samples.
    label_mapper : dict
        Mapping from trial labels (strings) to integer class labels,
        e.g. ``{'CS-': 0, 'CS+': 1}``.
    cfg : dict
        Decoding settings dictionary. Expected keys (under ``"searchlight"``)
        include:
        - ``radius_mm`` (float, default=6)
        - ``n_permutations`` (int, default=100)
        - ``n_jobs`` (int, default=1)
        - ``locked`` (dict of fixed estimator parameters)

        Top-level keys reused from the ROI path include:
        - ``outer_cv``
        - ``permute_within_groups``
        - ``estimator``
        - ``feature_selection``
        - ``variance_threshold``.
    groups : array_like, optional
        Optional group vector (e.g., run indices) of shape ``(n_samples,)``.
        Enables group-aware folds and within-group permutations.
    seed : int, default=0
        Global seed for center-wise RNG seeding.
    tmpdir : str, default="~/.joblib_cache"
        Directory for temporary memmaps and intermediate artifacts.
    output_file : str or None, optional
        Optional path forwarded to :class:`data.MaskAndFilterBetas` for writing
        intermediate outputs.
    **kwargs : dict, optional
        Additional keyword arguments forwarded to the per-center runner
        :func:`_one_center` and ultimately :func:`utils._cv_mean_score`
        (e.g., estimator-specific options).

    Returns
    -------
    dict of str to str
        Dictionary mapping map names to output file paths:

        - ``"observed"`` : NIfTI of observed cross-validated scores.
        - ``"null_mean"`` : NIfTI of permutation null mean scores.
        - ``"delta"`` : NIfTI of observed − null mean.
        - ``"pvalue"`` : NIfTI of diagnostic empirical p-values from the fixed null samples.
        - ``"nfeatures"`` : NIfTI of neighborhood sizes per center.

    Notes
    -----
    **Workflow**
        1. Build features and labels within ``ROI ∩ valid`` using
           :class:`data.MaskAndFilterBetas`.
        2. Create a voxel-to-column index volume and enumerate centers (ROI voxels).
        3. Generate outer folds via :func:`create_outer_folds` to mirror the ROI path.
        4. Memmap the full ROI feature matrix once; compute searchlight offsets with
           :func:`_neighbors_ball_mm`.
        5. For each center, call :func:`_one_center` to compute observed score and
           fixed-count permutation null (optionally in parallel with joblib).
        6. Assemble result volumes and write NIfTIs aligned to the resampled mask grid.
        7. Save a JSON sidecar with key metadata (radius, permutations, CV, etc.).

    **Parallelization**
        - Controlled by ``cfg["parallel"]["n_jobs"]``.
        - When ``> 1``, centers are processed with ``joblib.Parallel`` using the
          ``loky`` backend.
        - Center seeds are drawn from a global RNG initialized by ``seed``.

    **Output Files**
        - ``searchlight_observed.nii.gz``
        - ``searchlight_null_mean.nii.gz``
        - ``searchlight_delta.nii.gz``
        - ``searchlight_pvalue.nii.gz``
        - ``searchlight_nfeatures.nii.gz``
        - ``searchlight_desc-metadata.json`` (contains radius, permutations, CV config, etc.)

    Examples
    --------
    >>> out = permutation_searchlight(
    ...     betas_img=betas,
    ...     mask_img=mask,
    ...     trial_list=trials,
    ...     label_mapper={'CS-': 0, 'CS+': 1},
    ...     cfg=cfg['decoding_settings'],
    ...     groups=runs,
    ...     seed=123,
    ...     tmpdir="/tmp/sl-cache",
    ...     save_dir="results/sub-01"
    ... )
    >>> print(out["delta"])  # path to delta map

    See Also
    --------
    _one_center : Computes observed and permutation-based decoding accuracy for a single voxel.
    _neighbors_ball_mm : Computes voxel offsets defining a spherical neighborhood.
    utils._cv_mean_score : Runs cross-validated decoding for a given feature matrix.

    References
    ----------
    - Kriegeskorte, N., Goebel, R., & Bandettini, P. (2006).
      Information-based functional brain mapping.
      *Proceedings of the National Academy of Sciences*, 103(10), 3863–3868.
    """

    # config takes precedence over the presence of groups
    if not cfg.get("permute_within_groups", False):
        if groups is not None:
            logger.info("Groups (or runs) were detected, but permute_within_groups=False, so ignoring groups..")

        groups = None

    if groups is not None:
        logger.info(f"Groups = {groups}")

    # 0) Extract settings from cfg
    sl_cfg = cfg.get("searchlight", {})
    par_cfg = cfg.get("parallel", {})
    radius_mm = sl_cfg.get("radius_mm", 6)
    locked_params = sl_cfg.get("locked", None)
    alpha = sl_cfg.get("alpha", 0.05)

    # get plugin
    plugin, plugin_kwargs = core.get_analysis_plugin(
        cfg,
        label_dict=label_mapper,
    )

    logger.info(f"Running plugin: {plugin} with args: {plugin_kwargs}")

    analysis_cfg = cfg.get("analysis", {})
    output_kind = analysis_cfg.get("output_kind", "scalar")
    save_timeseries = output_kind == "timeseries"

    # 1) Extract X/labels inside ROI ∩ valid using your existing path
    mf = data.MaskAndFilterBetas(
        betas_img, mask_img,
        trial_list=trial_list,
        label_mapper=label_mapper,
        output_file=output_file,
        zooms=sl_cfg.get("target_zooms", None)
    )

    X = mf.X.astype("float32", copy=False)
    y = np.asarray(mf.labels)
    vol_shape = mf.fov_mask.shape[:3]
    col_index_vol = np.full(vol_shape, -1, np.int32)

    # No guessing about flattening order: just place columns where they belong
    coords = np.unravel_index(np.asarray(mf.roi_linidx, dtype=np.int64), vol_shape, order="C")
    col_index_vol[coords] = np.arange(len(mf.roi_linidx), dtype=np.int32)

    # Centers = every voxel that is actually a feature column
    centers = np.column_stack(np.where(col_index_vol >= 0))  # shape (N, 3)

    # 2) folds exactly like ROI path
    folds = create_outer_folds(cfg, y, groups=groups)

    # 3) memmap X once
    if tmpdir is None:
        tmpdir = os.environ.get("TMPDIR", "~/.joblib_cache")

    tmpdir = os.path.expanduser(tmpdir)
    os.makedirs(tmpdir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=tmpdir, prefix="panic_") as tmpd:
        X_path = dump(
            X,
            os.path.join(
                tmpd,
                f"Xsl_full_{uuid.uuid4().hex}.joblib",
            ),
            compress=0,
        )[0]

        # Don't mmap X in the parent just to get its shape
        logger.info(f"X={X.shape} (n_samples, n_features)")

        zooms = mf.fov_mask.header.get_zooms()
        offs  = _neighbors_ball_mm(zooms, radius_mm)

        # 5) run per-center
        rng = np.random.default_rng(seed)
        center_seeds = rng.integers(0, 2**32 - 1, size=len(centers), dtype=np.uint32)

        save_dir = tmpdir
        sl_base = sl_cfg.get("basepath", "searchlight")
        if "save_dir" in kwargs:
            save_dir = opj(kwargs.pop("save_dir"), sl_base)

        logger.info(f"Storing searchlight information in {save_dir}")
        n_perms = cfg.get("n_permutations", 1000)
        n_jobs = par_cfg.get("n_jobs", 1)

        sample_ix = np.linspace(
            0, len(centers) - 1,
            min(1000, len(centers)),
            dtype=int,
        )

        sample_sizes = np.asarray([
            len(_cols_for_center(centers[i], offs, col_index_vol, vol_shape))
            for i in sample_ix
        ])

        logger.info(
            "Searchlight neighbourhood sizes (sample=%d): mean=%.1f, median=%d, range=[%d,%d]",
            len(sample_sizes),
            float(sample_sizes.mean()),
            int(np.median(sample_sizes)),
            int(sample_sizes.min()),
            int(sample_sizes.max()),
        )

        logger.info(f"Centers={len(centers)} | r={radius_mm}mm | perms={n_perms} | jobs={n_jobs}")
        logger.info("Start searchlight analysis")

        monitor_root = par_cfg.get("monitor_runtime_dir")
        if not monitor_root:
            monitor_root = (
                os.environ.get("SLURM_TMPDIR")
                or os.environ.get("TMPDIR")
                or tmpd
            )
        monitor_runtime_dir = opj(
            os.path.expanduser(monitor_root),
            f"panic_searchlight_monitor_{os.getpid()}",
        )
        monitor_snapshot_dir = opj(save_dir, "searchlight_monitor")
        monitor_interval = float(par_cfg.get("monitor_interval", 15))
        monitor_resource_every = int(par_cfg.get("monitor_resource_every", 1000))

        worker_kwargs = {
            "centers": centers,
            "offsets": offs,
            "col_index_vol": col_index_vol,
            "vol_shape": vol_shape,
            "X_path": X_path,
            "labels": y,
            "folds": folds,
            "cfg": cfg,
            "groups": groups,
            "n_perms": n_perms,
            "center_seeds": center_seeds,
            "plugin": plugin,
            "plugin_kwargs": plugin_kwargs,
            "locked_params": locked_params,
            "save_dir": save_dir,
            "kwargs": kwargs,
            "monitor_runtime_dir": monitor_runtime_dir,
            "monitor_resource_every": monitor_resource_every,
        }

        update_interval = int(par_cfg.get("update_interval", 0))
        monitor_log_every = update_interval if update_interval > 0 else 5000
        out = [None] * len(centers)

        if n_jobs == 1:
            for i in tqdm(
                range(len(centers)),
                total=len(centers),
                disable=tqdm_disabled(),
            ):
                result_ix, result = _run_searchlight_center(
                    i,
                    **worker_kwargs,
                )
                out[result_ix] = result

        else:
            # Deliberately consume results directly rather than monkey-patching
            # joblib's BatchCompletionCallBack. This makes exact task identity
            # observable and removes the tqdm/joblib callback from the diagnostic
            # path. Results are restored to submission order via ``result_ix``.
            monitor = SearchlightMonitor(
                len(centers),
                runtime_dir=monitor_runtime_dir,
                snapshot_dir=monitor_snapshot_dir,
                logger=logger,
                interval=monitor_interval,
                log_every=monitor_log_every,
                stuck_thresholds=(30, 120, 600),
                signal_stuck_workers=True,
            )

            backend_name = par_cfg.get("backend", "loky")

            with monitor:
                parallel_kwargs = dict(
                    n_jobs=n_jobs,
                    backend=backend_name,
                    batch_size=par_cfg.get("batch_size", 16),
                    verbose=par_cfg.get("verbose", 0),
                    pre_dispatch=par_cfg.get("pre_dispatch", "2*n_jobs"),
                )

                # joblib's MultiprocessingBackend does not support generator
                # return modes. Its workers still journal exact completions, so
                # the watchdog remains informative while the parent blocks.
                if backend_name == "multiprocessing":
                    with Parallel(**parallel_kwargs) as parallel:
                        result_pairs = parallel(
                            delayed(_run_searchlight_center)(
                                i,
                                **worker_kwargs,
                            )
                            for i in range(len(centers))
                        )

                    for result_ix, result in result_pairs:
                        out[result_ix] = result
                        monitor.mark_completed(result_ix)

                else:
                    with Parallel(
                        return_as="generator_unordered",
                        **parallel_kwargs,
                    ) as parallel:
                        result_gen = parallel(
                            delayed(_run_searchlight_center)(
                                i,
                                **worker_kwargs,
                            )
                            for i in range(len(centers))
                        )

                        with tqdm(
                            total=len(centers),
                            disable=tqdm_disabled(),
                        ) as pbar:
                            for result_ix, result in result_gen:
                                out[result_ix] = result
                                monitor.mark_completed(result_ix)
                                pbar.update(1)

            missing = np.flatnonzero(
                np.fromiter((row is None for row in out), dtype=np.bool_, count=len(out))
            )
            if missing.size:
                raise RuntimeError(
                    "Parallel searchlight ended without results for center indices: "
                    f"{missing[:100].tolist()}"
                )

        # 6) assemble maps
        logger.info(f"Saving output maps (timeseries={save_timeseries})")
        obs_map         = np.full(vol_shape, np.nan, dtype=np.float32)
        null_map        = np.full(vol_shape, np.nan, dtype=np.float32)
        delta_map       = np.full(vol_shape, np.nan, dtype=np.float32)
        p_map           = np.full(vol_shape, np.nan, dtype=np.float32)
        nfeat_map       = np.zeros(vol_shape, dtype=np.int32)
        nperms_run_map  = np.zeros(vol_shape, dtype=np.int32)
        stopped_map     = np.zeros(vol_shape, dtype=np.uint8)
        stop_code_map   = np.zeros(vol_shape, dtype=np.uint8)

        # save CS-US similarity curve
        ts_map = None
        if save_timeseries:
            first_ts = next((row[-1] for row in out if row[-1] is not None), None)
            if first_ts is not None:
                n_time = len(first_ts)
                ts_map = np.full((*vol_shape, n_time), np.nan, dtype=np.float32)

        for row in out:
            (ix, iy, iz), obs, nullm, dlt, p, nf, nrun, stopped, stop_code, ts = row

            obs_map[ix, iy, iz] = obs
            null_map[ix, iy, iz] = nullm
            delta_map[ix, iy, iz] = dlt
            p_map[ix, iy, iz] = p
            nfeat_map[ix, iy, iz] = nf
            nperms_run_map[ix, iy, iz] = nrun
            stopped_map[ix, iy, iz] = stopped
            stop_code_map[ix, iy, iz] = stop_code

            if ts_map is not None and ts is not None:
                ts_map[ix, iy, iz, :] = ts

        # 7) save NIfTIs
        os.makedirs(save_dir, exist_ok=True)
        base = opj(save_dir or tmpdir, "searchlight")
        
        if hemi_key is not None and hemi_key != "uni":
            base += f"_hemi-{hemi_key}"

        ref = mf.fov_mask  # SAME grid/affine as vol_shape
        out_files = save_searchlight_maps(
            base,
            ref_img=ref,
            observed_map=obs_map,
            null_mean_map=null_map,
            delta_map=delta_map,
            pvalue_map=p_map,
            nfeatures_map=nfeat_map,
            nperms_run=nperms_run_map,
            stopped=stopped_map,
            stop_code=stop_code_map,
            fdr_alpha=alpha,
            n_perms=n_perms,
            mask_img=ref
        )

        if ts_map is not None:
            f = f"{base}_cs_us_similarity_timeseries.nii.gz"
            image.new_img_like(ref, ts_map, copy_header=True).to_filename(f)
            out_files["cs_us_similarity_timeseries"] = f

            logger.info(
                "y type=%s shape=%s cs_label=%r",
                type(y),
                getattr(y, "shape", None),
                plugin_kwargs.get("cs_label"),
            )

            cs_idx = np.where(np.asarray(y) == plugin_kwargs.get("cs_label"))[0]
            np.save(f"{base}_cs_trial_indices.npy", cs_idx)
            out_files["cs_trial_indices"] = f"{base}_cs_trial_indices.npy"

        # also drop a small JSON sidecar
        meta = {
            "radius_mm": float(radius_mm),
            "n_permutations": int(n_perms),
            "cv": cfg.get("outer_cv", {}),
            "permute_within_groups": bool(cfg.get("permute_within_groups", False)),
            "estimator": cfg.get("estimator", {}),
            "feature_selection": cfg.get("feature_selection", {}),
            "variance_threshold": float(cfg.get("variance_threshold", 1e-12)),
            "fdr_alpha": float(alpha),
            "null_mode": "fixed_count_mean",
            "n_centers": int(np.isfinite(p_map).sum()),
            "stopped_total": int(np.sum(stopped_map)),
            "stopped_pmin": int(np.sum(stop_code_map == 1)),
            "stopped_pmax": int(np.sum(stop_code_map == 2)),
            "median_nperms_run": float(np.median(nperms_run_map[nperms_run_map > 0])) if (nperms_run_map > 0).any() else 0.0,        
        }

        with open(f"{base}_desc-metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Done\n")
        return out_files


def save_searchlight_maps(
    base_path,
    ref_img,
    *,
    observed_map,
    null_mean_map,
    delta_map,
    pvalue_map,
    nfeatures_map,
    nperms_run,
    stopped,
    stop_code,
    mask_img=None,
    fdr_alpha=0.05,
    n_perms=None,
):
    """
    Save and summarize searchlight decoding results as NIfTI images.

    This function writes the core searchlight maps (``observed``,
    ``null_mean``, ``delta``, ``pvalue``, ``nfeatures``), permutation
    diagnostics, FDR-corrected p-values, visualization-friendly
    ``-log10(p)`` maps, and thresholded effect maps.

    All input arrays are assumed to already be defined on the spatial grid
    of ``ref_img``.

    If ``mask_img`` is provided, it must also already be on exactly the same
    voxel grid as ``ref_img``. No resampling is performed. Voxels outside
    this mask are written as zero in the ordinary output maps and are
    excluded from FDR correction.

    **Primary maps**
        - ``observed`` — Mean cross-validated decoding score with true
          labels at each searchlight center.

        - ``null_mean`` — Mean decoding score across permuted-label
          repetitions at each center.

        - ``delta`` — Difference ``observed - null_mean``.

        - ``pvalue`` — Uncorrected voxelwise permutation p-value.

        - ``nfeatures`` — Number of voxel features included in each
          searchlight sphere.

    **Diagnostic maps**
        - ``nperms_run`` — Actual number of permutations executed at each
          voxel.

        - ``nperms_run_frac`` — Fraction of the requested permutations
          executed, written when ``n_perms`` is provided.

        - ``stopped`` — Binary indicator of whether early stopping occurred.

        - ``stop_code`` — Early-stop reason:
          0 = not stopped,
          1 = ``p_min > alpha``,
          2 = ``p_max < alpha``.

    **Derived maps**
        - ``pvalue_fdr`` — Benjamini-Hochberg FDR-adjusted p-values,
          calculated only across valid voxels inside ``mask_img``. If no
          mask is supplied, all finite p-values are used.

        - ``neglogp`` — ``-log10(pvalue)`` within the valid mask.

        - ``neglogp_fdr`` — ``-log10(pvalue_fdr)`` within the valid mask.

        - ``sig_uncorrected`` — ``delta`` at voxels satisfying
          ``pvalue < fdr_alpha``. Non-significant voxels inside the mask are
          NaN; voxels outside the mask are zero.

        - ``sig_fdr`` — ``delta`` at voxels satisfying
          ``pvalue_fdr < fdr_alpha``. Non-significant voxels inside the mask
          are NaN; voxels outside the mask are zero.

    Parameters
    ----------
    base_path : str or os.PathLike
        Output prefix. For example, if
        ``base_path == '/tmp/sub-01_searchlight'``, the p-value map is
        written to ``/tmp/sub-01_searchlight_pvalue.nii.gz``.

    ref_img : nibabel.spatialimages.SpatialImage
        Reference image defining the shape, affine, and header for all
        output maps. Typically this is the final binary FOV mask in beta
        space.

    observed_map : ndarray, shape (X, Y, Z)
        Observed decoding score per searchlight center.

    null_mean_map : ndarray, shape (X, Y, Z)
        Mean permuted-label decoding score per center.

    delta_map : ndarray, shape (X, Y, Z)
        Difference ``observed_map - null_mean_map``.

    pvalue_map : ndarray, shape (X, Y, Z)
        Uncorrected voxelwise permutation p-values.

    nfeatures_map : ndarray, shape (X, Y, Z)
        Number of features included in each searchlight sphere.

    nperms_run : ndarray, shape (X, Y, Z)
        Number of permutations actually executed per voxel.

    stopped : ndarray, shape (X, Y, Z)
        Binary map indicating whether early stopping triggered.

    stop_code : ndarray, shape (X, Y, Z)
        Early-stopping reason code.

    mask_img : nibabel.spatialimages.SpatialImage, optional
        Binary mask defining valid searchlight/FOV voxels. It must already
        have the same shape and affine as ``ref_img``. No resampling is
        performed.

        Voxels outside this mask are excluded from statistical correction
        and written as zero in ordinary output maps.

        If ``None``, all voxels on the reference grid are considered part
        of the spatial mask; finite p-values determine validity for
        statistical operations.

    fdr_alpha : float, optional
        Significance level used for Benjamini-Hochberg FDR correction and
        thresholded significance maps. Default is 0.05.

    n_perms : int or None, optional
        Maximum/requested number of permutations. If provided,
        ``nperms_run_frac = nperms_run / n_perms`` is saved.

    Returns
    -------
    out_files : dict
        Mapping from map name to output filename.

        Always contains:
        ``observed``, ``null_mean``, ``delta``, ``pvalue``,
        ``nfeatures``, ``nperms_run``, ``stopped``, ``stop_code``.

        When ``n_perms`` is provided, also contains
        ``nperms_run_frac``.

        When at least one finite p-value exists inside the valid mask,
        additionally contains:
        ``pvalue_fdr``, ``neglogp``, ``neglogp_fdr``,
        ``sig_uncorrected``, and ``sig_fdr``.

    Notes
    -----
    - ``ref_img`` supplies spatial metadata only. It does not itself mask
      array values.

    - ``mask_img`` is therefore applied explicitly before maps are written.

    - No interpolation or resampling is performed in this function.
      Searchlight result arrays and ``mask_img`` are expected to already be
      on the same grid as ``ref_img``.

    - Outside-mask values are written as zero. This is done only for the
      saved maps; statistical calculations use the original p-value array
      together with the boolean validity mask, so outside-mask zeros can
      never be interpreted as significant p-values.

    - Empirical p-values generally have a minimum attainable value of
      ``1 / (n_permutations_run + 1)`` at each voxel.

    Examples
    --------
    >>> out = save_searchlight_maps(
    ...     base_path=os.path.join(output_dir, "searchlight"),
    ...     ref_img=mf.fov_mask,
    ...     observed_map=obs_map,
    ...     null_mean_map=null_map,
    ...     delta_map=delta_map,
    ...     pvalue_map=p_map,
    ...     nfeatures_map=nfeat_map,
    ...     nperms_run=nperms_run_map,
    ...     stopped=stopped_map,
    ...     stop_code=stop_code_map,
    ...     mask_img=mf.fov_mask,
    ...     n_perms=n_perms,
    ...     fdr_alpha=0.05,
    ... )
    """

    # ------------------------------------------------------------------
    # 0) Basic validation
    output_dir = os.path.dirname(os.fspath(base_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ref_shape = tuple(ref_img.shape[:3])

    arrays = {
        "observed_map": observed_map,
        "null_mean_map": null_mean_map,
        "delta_map": delta_map,
        "pvalue_map": pvalue_map,
        "nfeatures_map": nfeatures_map,
        "nperms_run": nperms_run,
        "stopped": stopped,
        "stop_code": stop_code,
    }

    for name, arr in arrays.items():
        arr = np.asarray(arr)

        if arr.shape != ref_shape:
            raise ValueError(
                f"{name} has shape {arr.shape}, but reference image "
                f"has spatial shape {ref_shape}."
            )

    if not (0.0 < fdr_alpha < 1.0):
        raise ValueError(
            f"fdr_alpha must lie between 0 and 1; got {fdr_alpha}."
        )

    if n_perms is not None:
        if not isinstance(n_perms, (int, np.integer)) or n_perms <= 0:
            raise ValueError(
                f"n_perms must be a positive integer; got {n_perms!r}."
            )

    # ------------------------------------------------------------------
    # 1) Build the spatial/FOV mask.
    #
    # mask_img is expected to ALREADY be on the ref_img grid.
    # Do not resample here.
    if mask_img is not None:
        if tuple(mask_img.shape[:3]) != ref_shape:
            raise ValueError(
                "mask_img and ref_img are not on the same grid: "
                f"mask shape={mask_img.shape[:3]}, "
                f"reference shape={ref_shape}."
            )

        if not np.allclose(
            mask_img.affine,
            ref_img.affine,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError(
                "mask_img and ref_img have different affines. "
                "The mask must already be in the searchlight/reference "
                "space; this function does not resample masks."
            )

        spatial_mask = (
            np.asarray(mask_img.get_fdata(dtype=np.float32)) > 0.5
        )

    else:
        spatial_mask = np.ones(ref_shape, dtype=bool)

    if not np.any(spatial_mask):
        raise ValueError("mask_img contains no non-zero voxels.")

    # Convert here so all subsequent calculations use predictable arrays.
    observed_map = np.asarray(observed_map, dtype=np.float32)
    null_mean_map = np.asarray(null_mean_map, dtype=np.float32)
    delta_map = np.asarray(delta_map, dtype=np.float32)
    pvalue_map = np.asarray(pvalue_map, dtype=np.float32)
    nfeatures_map = np.asarray(nfeatures_map)
    nperms_run = np.asarray(nperms_run)
    stopped = np.asarray(stopped)
    stop_code = np.asarray(stop_code)

    # Validate finite p-values where present.
    finite_p = np.isfinite(pvalue_map)
    bad_p = finite_p & (
        (pvalue_map < 0.0) |
        (pvalue_map > 1.0)
    )

    if np.any(bad_p):
        vals = pvalue_map[bad_p]
        raise ValueError(
            "pvalue_map contains finite values outside [0, 1]. "
            f"Observed range among invalid values: "
            f"{vals.min():.6g} to {vals.max():.6g}."
        )

    # ------------------------------------------------------------------
    # Helper: explicitly zero values outside the final FOV.
    #
    # This is what actually performs masking. new_img_like() by itself
    # only transfers spatial metadata.
    def masked_zero(data, dtype=np.float32):
        out = np.zeros(ref_shape, dtype=dtype)
        out[spatial_mask] = np.asarray(data)[spatial_mask]
        return out

    # ------------------------------------------------------------------
    # 2) Save primary and diagnostic maps.
    imgs = {
        "observed": image.new_img_like(
            ref_img,
            masked_zero(observed_map, np.float32),
            copy_header=True,
        ),
        "null_mean": image.new_img_like(
            ref_img,
            masked_zero(null_mean_map, np.float32),
            copy_header=True,
        ),
        "delta": image.new_img_like(
            ref_img,
            masked_zero(delta_map, np.float32),
            copy_header=True,
        ),
        "pvalue": image.new_img_like(
            ref_img,
            masked_zero(pvalue_map, np.float32),
            copy_header=True,
        ),
        "nfeatures": image.new_img_like(
            ref_img,
            masked_zero(nfeatures_map, np.float32),
            copy_header=True,
        ),
        "nperms_run": image.new_img_like(
            ref_img,
            masked_zero(nperms_run, np.float32),
            copy_header=True,
        ),
        "stopped": image.new_img_like(
            ref_img,
            masked_zero(stopped, np.uint8),
            copy_header=True,
        ),
        "stop_code": image.new_img_like(
            ref_img,
            masked_zero(stop_code, np.uint8),
            copy_header=True,
        ),
    }

    if n_perms is not None:
        nperms_run_frac = (
            nperms_run.astype(np.float32) /
            float(n_perms)
        )

        imgs["nperms_run_frac"] = image.new_img_like(
            ref_img,
            masked_zero(nperms_run_frac, np.float32),
            copy_header=True,
        )

    out_files = {}

    for name, img in imgs.items():
        filename = f"{base_path}_{name}.nii.gz"
        img.to_filename(filename)
        out_files[name] = filename

    # ------------------------------------------------------------------
    # 3) Define valid voxels for statistical operations.
    #
    # IMPORTANT:
    # We use the ORIGINAL p-value array here, not the zero-masked p-value
    # saved above. Thus outside-FOV zeros cannot enter FDR correction.
    mask_valid = spatial_mask & np.isfinite(pvalue_map)
    p_vec = pvalue_map[mask_valid]

    if p_vec.size == 0:
        return out_files

    # ------------------------------------------------------------------
    # 4) Benjamini-Hochberg FDR correction.
    reject_fdr, p_fdr_vec = fdrcorrection(
        p_vec,
        alpha=fdr_alpha,
    )

    # Internal representation:
    # NaN everywhere except valid statistical voxels.
    p_fdr = np.full(
        ref_shape,
        np.nan,
        dtype=np.float32,
    )
    p_fdr[mask_valid] = p_fdr_vec.astype(np.float32)

    # Saved p_FDR image:
    # zero outside the FOV, corrected values inside.
    p_fdr_out = np.zeros(
        ref_shape,
        dtype=np.float32,
    )
    p_fdr_out[mask_valid] = p_fdr[mask_valid]

    # ------------------------------------------------------------------
    # 5) -log10(p) maps.
    #
    # Construct these directly only at valid voxels. This prevents the
    # zero values used outside the FOV in the saved p-map from turning
    # into enormous -log10(p) values.
    eps = np.finfo(np.float32).tiny

    neglogp = np.zeros(
        ref_shape,
        dtype=np.float32,
    )
    neglogp[mask_valid] = -np.log10(
        np.clip(
            pvalue_map[mask_valid],
            eps,
            1.0,
        )
    )

    valid_fdr = spatial_mask & np.isfinite(p_fdr)

    neglogp_fdr = np.zeros(
        ref_shape,
        dtype=np.float32,
    )
    neglogp_fdr[valid_fdr] = -np.log10(
        np.clip(
            p_fdr[valid_fdr],
            eps,
            1.0,
        )
    )

    # ------------------------------------------------------------------
    # 6) Thresholded delta/effect maps.
    #
    # Outside FOV: 0
    # Inside FOV but non-significant: NaN
    # Significant: delta
    sig_uncorrected = np.zeros(
        ref_shape,
        dtype=np.float32,
    )
    sig_uncorrected[spatial_mask] = np.nan

    sig_unc_mask = (
        mask_valid &
        (pvalue_map < fdr_alpha)
    )
    sig_uncorrected[sig_unc_mask] = delta_map[sig_unc_mask]

    sig_fdr = np.zeros(
        ref_shape,
        dtype=np.float32,
    )
    sig_fdr[spatial_mask] = np.nan

    sig_fdr_mask = np.zeros(
        ref_shape,
        dtype=bool,
    )
    sig_fdr_mask[mask_valid] = reject_fdr

    sig_fdr[sig_fdr_mask] = delta_map[sig_fdr_mask]

    # ------------------------------------------------------------------
    # 7) Save derived maps.
    extras = {
        "pvalue_fdr": image.new_img_like(
            ref_img,
            p_fdr_out,
            copy_header=True,
        ),
        "neglogp": image.new_img_like(
            ref_img,
            neglogp,
            copy_header=True,
        ),
        "neglogp_fdr": image.new_img_like(
            ref_img,
            neglogp_fdr,
            copy_header=True,
        ),
        "sig_uncorrected": image.new_img_like(
            ref_img,
            sig_uncorrected,
            copy_header=True,
        ),
        "sig_fdr": image.new_img_like(
            ref_img,
            sig_fdr,
            copy_header=True,
        ),
    }

    for name, img in extras.items():
        filename = f"{base_path}_{name}.nii.gz"
        img.to_filename(filename)
        out_files[name] = filename

    return out_files

