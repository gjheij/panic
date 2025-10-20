# searchlight.py
# -*- coding: utf-8 -*-
import logging
import numpy as np
import nibabel as nib
from tqdm import tqdm
from nilearn import image
import os, math, json, tempfile, uuid
from joblib import Parallel, delayed, dump, load
from panic.logger import get_logger, tqdm_joblib
from panic import (
    data,
    factory,
    utils,
)
from statsmodels.stats.multitest import fdrcorrection

logger = get_logger(__name__, level=logging.INFO, use_tqdm=True)
opj = os.path.join


import numpy as np
from typing import Callable, Tuple, Optional

def _permute_with_early_stop(
    obs_score: float,
    seeds: np.ndarray,
    n_perms: int,
    alpha: Optional[float],
    score_fn: Callable[[np.random.Generator], float],
) -> Tuple[float, float, int]:
    """
    Perform permutation testing with optional *fail-fast* early stopping.

    This function estimates an empirical one-sided p-value for a decoding or
    cross-validation score by comparing the observed statistic against a
    permutation-derived null distribution. It optionally implements a
    **fail-fast early-stopping rule**, which halts permutations as soon as it
    becomes *statistically impossible* for the voxel/ROI to reach significance
    at level ``alpha``.

    Unlike symmetric early stopping (which can also stop for early significance),
    the fail-fast version only terminates permutations when the lower bound
    on the attainable p-value exceeds ``alpha``—that is, when even the most
    optimistic outcome would remain non-significant. This ensures conservative
    inference without the risk of prematurely “accepting” significance.

    Parameters
    ----------
    obs_score : float
        Observed score computed using true labels (e.g., cross-validated accuracy
        or correlation coefficient). Compared against permutation scores to estimate
        the empirical p-value.
    seeds : ndarray of int, shape (n_perms,)
        Integer random seeds used to initialize the RNG for each permutation.
        Provides reproducible sampling of null scores.
    n_perms : int
        Maximum number of permutations to execute.
    alpha : float or None, optional
        Early-stopping significance level (e.g., ``0.05``).  
        If provided, the algorithm stops early **only** when it becomes
        impossible for the final empirical p-value to fall below ``alpha``—
        specifically, when the lower bound
        ``p_min = (count_exceed + 1) / (n_run + 1)`` exceeds ``alpha`` after
        at least ``ceil(1/alpha) - 1`` permutations.  
        If ``None`` (default), all permutations are executed.
    score_fn : Callable[[numpy.random.Generator], float]
        A callable that accepts a NumPy ``Generator`` and returns a single
        permuted score (float). Typically wraps a decoding or cross-validation
        routine with shuffled labels.

    Returns
    -------
    null_mean : float
        Mean of all computed permutation scores (approximate null expectation).
    p_value : float
        Empirical one-sided p-value computed as
        ``p = (Σ(null ≥ obs) + 1) / (n_run + 1)``.
    n_run : int
        Number of permutations actually executed (≤ ``n_perms``).

    Notes
    -----
    - Uses a one-sided test: ``p = P(null ≥ obs_score)``.
    - Adds ``+1`` to numerator and denominator for finite-sample correction.
    - The smallest attainable p-value is ``1 / (n_run + 1)``.
    - Fail-fast early stopping only declares “cannot become significant”;
      it never accepts significance prematurely.
    - This rule typically saves computation for voxels with clearly null effects,
      while leaving truly significant voxels to complete all permutations.

    Example
    -------
    .. code-block:: python

        import numpy as np

        def score_fn(rng):
            # Randomized null distribution centered at 0.5
            return rng.normal(0.5, 0.05)

        obs_score = 0.72
        rng = np.random.default_rng(42)
        seeds = rng.integers(0, 2**32, size=1000)

        null_mean, p_val, n_run = run_permutations(
            obs_score=obs_score,
            seeds=seeds,
            n_perms=1000,
            alpha=0.05,
            score_fn=score_fn
        )

        print(null_mean, p_val, n_run)
        # Example output: 0.498, 0.002, 1000  (ran all perms; likely significant)

    Computational rule
    ------------------
    For each permutation ``j = 1 … n_perms``:

        1. Draw RNG = np.random.default_rng(int(seeds[j]))
        2. Compute ``perm_score = score_fn(RNG)``
        3. Update ``count_exceed += (perm_score >= obs_score)``
        4. Compute ``p_min = (count_exceed + 1) / (j + 1)``  
           If ``p_min > alpha`` and ``j >= ceil(1/alpha) - 1``, stop early.

    After termination, compute the final p-value as:

    .. math::

        p = \\frac{\\text{count}_\\text{exceed} + 1}{n_\\text{run} + 1}

    and report the mean null score as the empirical null expectation.
    """

    perm_vals = []
    count_exceed = 0
    stopped = False
    reason = None

    J0 = int(np.ceil(1.0/alpha) - 1) if alpha is not None else None
    for j, s in enumerate(seeds[:n_perms], 1):
        v = float(score_fn(np.random.default_rng(int(s))))
        perm_vals.append(v)
        if v >= obs_score:
            count_exceed += 1

        if alpha is not None and j >= J0:
            p_min = (count_exceed + 1) / (j + 1)
            if p_min > alpha:
                stopped = True
                reason = "p_min>alpha"   # fail-fast only
                break

    perm_vals = np.asarray(perm_vals, dtype=float)
    # Empirical p using the permutations actually run
    p_value = (np.sum(perm_vals >= obs_score) + 1.0) / (len(perm_vals) + 1.0)
    null_mean = float(np.mean(perm_vals)) if perm_vals.size else np.nan
    return null_mean, float(p_value), int(perm_vals.size), stopped, reason


def _voxel_radius_in_voxels(mask_img, radius_mm):
    """
    Convert a spherical searchlight radius from millimeters to voxel units.

    This helper estimates the equivalent radius in voxel space by dividing
    the given physical radius (in mm) by the voxel size along the x-axis.
    It assumes approximately isotropic voxel dimensions and falls back to
    reasonable defaults if voxel size metadata is unavailable.

    :param nibabel.Nifti1Image mask_img:
        A 3D NIfTI image (or compatible object) whose header contains voxel
        dimension information in ``pixdim`` or via ``header.get_zooms()``.
    :param float radius_mm:
        Searchlight radius expressed in millimeters.

    :returns:
        The radius converted to voxel units, rounded to the nearest integer.
        A minimum of 1 voxel is enforced.
    :rtype:
        int

    **Computation Details**
        - Attempts to extract voxel dimensions using
          ``mask_img.header.get_zooms()[:3]``.
        - If unavailable or malformed, falls back to using the array shape
          of the image data (via ``image.get_data(mask_img).shape[:3]``).
        - Uses only the first element of the voxel size (x-axis spacing)
          to compute an approximately isotropic voxel radius:
          ``radius_vox = round(radius_mm / voxel_size_x)``.
        - Returns at least 1 to avoid a zero-radius searchlight.

    **Example**
        .. code-block:: python

            import nibabel as nib

            mask_img = nib.load("roi_mask.nii.gz")
            r_vox = _voxel_radius_in_voxels(mask_img, radius_mm=6.0)
            print(f"Searchlight radius ≈ {r_vox} voxels")

    .. note::
       - This conversion assumes roughly isotropic voxel dimensions; if
         voxel sizes differ substantially across axes, the result is only
         an approximation.
       - Ensures a minimum radius of 1 voxel to maintain a valid neighborhood.
       - Designed for compatibility with fMRI searchlight decoding pipelines.
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

    :param sequence zooms:
        Sequence of voxel dimensions (in mm) along the x, y, and z axes,
        typically obtained from a NIfTI image header via
        ``mask_img.header.get_zooms()[:3]``.
        Example: ``(3.5, 3.75, 5.0)``.
    :param float r_mm:
        Searchlight radius in millimeters.

    :returns:
        Array of integer voxel offsets ``(dx, dy, dz)`` such that the
        Euclidean distance (in mm) from the origin does not exceed ``r_mm``.
        Shape is ``(n_neighbors, 3)``.
    :rtype:
        numpy.ndarray

    **Computation Details**
        - The voxel-space bounding box is determined by dividing the
          millimeter radius by voxel sizes along each axis:
          ``rx = int(r_mm // vx)``, etc.
        - All integer offset combinations within ``[-rx, rx] × [-ry, ry] × [-rz, rz]``
          are tested for inclusion using the distance criterion:
          ``(dx*vx)**2 + (dy*vy)**2 + (dz*vz)**2 <= r_mm**2``.
        - Returns offsets as a NumPy integer array.

    **Example**
        .. code-block:: python

            zooms = (3.5, 3.75, 5.0)
            r_mm = 6.0
            offsets = _neighbors_ball_mm(zooms, r_mm)
            print(f"{len(offsets)} neighbors within {r_mm} mm radius")

    .. note::
       - The output represents **relative** voxel coordinates; to obtain
         absolute indices in an image, add these offsets to a voxel’s (i, j, k)
         coordinates.
       - Accounts for anisotropic voxel dimensions by scaling distances
         according to physical spacing.
       - Designed for use in searchlight mapping and neighborhood-based
         analyses.
    """

    vx, vy, vz = map(float, zooms[:3])     # e.g. (3.5, 3.75, 5.0)
    rx, ry, rz = int(r_mm // vx), int(r_mm // vy), int(r_mm // vz)
    offs = []
    r2 = r_mm * r_mm
    for dx in range(-rx, rx+1):
        for dy in range(-ry, ry+1):
            for dz in range(-rz, rz+1):
                if (dx*vx)**2 + (dy*vy)**2 + (dz*vz)**2 <= r2:
                    offs.append((dx, dy, dz))
    return np.asarray(offs, dtype=int)


def _one_center(
    center_ijk,
    offsets,
    col_index_vol,
    vol_shape,
    roi_linidx,
    X_mm_path,
    labels,
    folds,
    cfg,
    groups,
    n_perms,
    seed,
    **kwargs
):
    """
    Compute observed and permutation-based decoding accuracy for a single
    searchlight center voxel.

    This function performs decoding at one spatial location by extracting
    the neighborhood of voxels (as defined by ``offsets``), running cross-validated
    decoding on that subset of features, and computing both the observed and
    null (permutation) performance estimates.

    It returns a compact tuple summarizing all key results for the given
    center voxel, suitable for aggregation across the searchlight volume.

    :param tuple[int, int, int] center_ijk:
        Integer voxel coordinates ``(i, j, k)`` of the current searchlight center.
    :param numpy.ndarray offsets:
        Array of voxel offset coordinates ``(dx, dy, dz)`` defining the
        spherical neighborhood (e.g., output of :func:`_neighbors_ball_mm`).
    :param numpy.ndarray col_index_vol:
        3D integer array mapping voxel coordinates to feature indices within
        the ROI feature space. Entries < 0 indicate non-ROI voxels.
    :param tuple[int, int, int] vol_shape:
        Shape of the full 3D brain volume (e.g., from the mask image).
    :param numpy.ndarray roi_linidx:
        Linear indices of voxels included in the ROI feature space.
        Used for mapping between image and feature coordinates.
    :param str X_mm_path:
        Path to a ``joblib`` dump containing the full memory-mapped feature
        matrix of shape ``(n_samples, n_features_all_roi)``.
    :param array_like labels:
        Array of target labels for decoding, shape ``(n_samples,)``.
    :param list folds:
        List of outer cross-validation splits as tuples ``(train_idx, test_idx)``.
    :param dict cfg:
        Configuration dictionary for decoding, passed to :func:`_cv_mean_score`.
    :param array_like groups:
        Optional group labels of shape ``(n_samples,)`` for group-aware CV or
        within-group permutations.
    :param int n_perms:
        Number of label permutations to run for estimating the null distribution.
    :param int seed:
        Random seed controlling the reproducibility of permutation sampling.
    :param kwargs:
        Additional keyword arguments forwarded to :func:`_cv_mean_score`.

    :returns:
        A 6-element tuple containing:

        1. ``center_ijk`` – voxel coordinates (tuple)
        2. ``obs`` – observed mean cross-validated score (float)
        3. ``null_mean`` – mean score under permutation null (float)
        4. ``delta`` – difference between observed and null mean (float)
        5. ``p`` – empirical p-value computed as
           ``(sum(perm >= obs) + 1) / (len(perm) + 1)``
        6. ``n_feat`` – number of voxels included in the neighborhood (int)

    :rtype:
        tuple[tuple[int, int, int], float, float, float, float, int]

    **Computation Steps**
        1. Identify valid voxel neighbors within the searchlight radius
           using ``offsets`` and ``col_index_vol``.
        2. Skip computation if fewer than 2 valid features are available.
        3. Extract the relevant feature subset from the memmapped matrix
           and save a lightweight temporary file for fast I/O.
        4. Compute the observed decoding score via :func:`utils._cv_mean_score`.
        5. Perform ``n_perms`` permutation runs, each using an independent RNG
           initialized from ``seed``.
        6. Aggregate permutation scores and compute ``null_mean``, ``delta``,
           and empirical ``p``.

    **Example**
        .. code-block:: python

            center = (32, 40, 20)
            offsets = _neighbors_ball_mm((3.5, 3.5, 3.5), 6.0)
            result = _one_center(
                center_ijk=center,
                offsets=offsets,
                col_index_vol=col_index_vol,
                vol_shape=(64, 64, 36),
                roi_linidx=roi_idx,
                X_mm_path="/tmp/X_roi.joblib",
                labels=y,
                folds=folds,
                cfg=cfg,
                groups=runs,
                n_perms=100,
                seed=1234
            )
            print(result)
            # ((32, 40, 20), 0.71, 0.50, 0.21, 0.01, 87)

    .. note::
        - For efficiency, the function dumps a small temporary
            ``joblib`` file containing only the subset of features in
            the current searchlight neighborhood.
        - A minimum of 2 valid voxels is required to compute decoding.
        - The empirical p-value is bias-corrected using a +1 numerator and
            denominator adjustment.
        - Uses :func:`numpy.random.default_rng` for reproducible random sampling.
        - Designed for internal use within parallelized searchlight loops.
    """

    X_mm = load(X_mm_path, mmap_mode="r")  # shape [n_samples, n_features_all_roi]
    # Build neighborhood feature indices (in ROI feature space) for this center
    cx, cy, cz = map(int, center_ijk)
    cols = []
    for dx, dy, dz in offsets:
        x, y, z = cx+dx, cy+dy, cz+dz
        if 0 <= x < vol_shape[0] and 0 <= y < vol_shape[1] and 0 <= z < vol_shape[2]:
            col = col_index_vol[x, y, z]
            if col >= 0:
                cols.append(col)

    if len(cols) < 2:
        return (cx, cy, cz), np.nan, np.nan, np.nan, np.nan, len(cols)

    # Build temporary mmap with only those columns for speed during CV
    # (slice view is fine; _cv_mean_score already memmaps by path, so dump a small array)
    tmp = X_mm[:, cols]

    # Save light-weight array uncompressed for fast mmap
    tmp_dir = os.path.dirname(X_mm_path)
    tmp_path = dump(
        tmp,
        opj(
            tmp_dir,
            f"Xsl_{uuid.uuid4().hex}.joblib"
        ),
        compress=0
    )[0]

    # observed
    obs = utils._cv_mean_score(
        tmp_path, labels, folds, cfg,
        groups=groups,
        permute=False,
        **kwargs
    )

    # permutations with early stop
    sl_cfg = cfg.get("searchlight", {}) or {}
    alpha = sl_cfg.get("alpha", None)
    early_stop_alpha = sl_cfg.get("early_stop_alpha", None)

    # derive permutation seeds from this center's seed
    center_rng = np.random.default_rng(int(seed))
    seeds = center_rng.integers(0, 2**32 - 1, size=n_perms, dtype=np.uint32)

    def _score_once(rng):
        # delegate to utils._cv_mean_score exactly as before
        return utils._cv_mean_score(
            tmp_path, labels, folds, cfg,
            rng=rng,
            permute=True,
            **kwargs
        )

    null_mean, p, n_run, stopped, reason = _permute_with_early_stop(
        obs_score=obs,
        seeds=seeds,
        n_perms=n_perms,
        alpha=early_stop_alpha,
        score_fn=_score_once,
    )

    delta = float(obs - null_mean) if np.isfinite(null_mean) else np.nan

    # encode reason as small int for a compact NIfTI if you like
    stop_code = 0
    if stopped:
        stop_code = 1 if reason == "p_min>alpha" else 2  # (1) cannot be sig, (2) already sig
            
    return (cx, cy, cz), float(obs), float(null_mean), float(delta), float(p), int(len(cols)), n_run, int(stopped), int(stop_code)

def permutation_searchlight(
    betas_img,            # 4D betas (nifti)
    mask_img,             # binary ROI/brain mask (nifti)
    trial_list,           # list[str] or array[str] per volume in betas
    label_mapper,         # dict like {'CS-':0,'CS+':1}
    cfg,
    *,
    groups=None,          # run indices per trial (optional)
    seed=0,
    tmpdir="~/.joblib_cache",
    output_file=None,
    **kwargs
):
    """
    Run a permutation-based searchlight decoding analysis and write result maps.

    This function computes, for each voxel center inside the ROI, the observed
    cross-validated decoding score and a permutation-based null model, then
    assembles NIfTI maps of observed score, null mean, delta (observed − null),
    empirical p-value, and number of features used. Work can be parallelized
    across centers.

    :param nibabel.Nifti1Image betas_img:
        4D beta image with shape ``(X, Y, Z, n_samples)``.
    :param nibabel.Nifti1Image mask_img:
        Binary ROI (or brain) mask aligned to ``betas_img`` space (or resampled to it).
    :param sequence trial_list:
        Trial descriptors aligned with the 4th dimension of ``betas_img``; used
        by :class:`data.MaskAndFilterBetas` to select and order samples.
    :param dict label_mapper:
        Mapping from trial labels (strings) to integer class labels, e.g.
        ``{'CS-': 0, 'CS+': 1}``.
    :param dict cfg:
        Decoding settings dictionary. Expected keys (under ``"searchlight"``) include:
        ``"radius_mm"`` (float, default 6), ``"n_permutations"`` (int, default 100),
        ``"n_jobs"`` (int, default 1), and ``"locked"`` (dict of fixed estimator params).
        Top-level keys reused from ROI path include
        ``"outer_cv"``, ``"permute_within_groups"``, ``"estimator"``,
        ``"feature_selection"``, and ``"variance_threshold"``.
    :param array_like groups:
        Optional group vector (e.g., run indices) of shape ``(n_samples,)``.
        Enables group-aware folds and within-group permutations.
    :param int seed:
        Global seed for center-wise RNG seeding.
    :param int n_perms:
        Number of label permutations to compute for the null distribution.
        Default is 250. Set with general_settings.n_permutations in config file.     
    :param str tmpdir:
        Directory for temporary memmaps and intermediate artifacts
        (defaults to ``~/.joblib_cache``).
    :param str | None output_file:
        Optional path forwarded to :class:`data.MaskAndFilterBetas` for writing
        intermediate outputs.
    :param int n_jobs:
        Controls the level of parallellization over centers. Default is 1.
        Should be set in the config file under ``"general_settings.n_jobs"``.
    :param kwargs:
        Additional keyword arguments forwarded to the per-center runner
        :func:`_one_center` and ultimately :func:`utils._cv_mean_score`
        (e.g., estimator-specific options).

    :returns:
        Dictionary mapping map names to output file paths:

        * ``"observed"`` – NIfTI of observed cross-validated scores
        * ``"null_mean"`` – NIfTI of permutation null mean scores
        * ``"delta"`` – NIfTI of observed − null mean
        * ``"pvalue"`` – NIfTI of empirical p-values
        * ``"nfeatures"`` – NIfTI of neighborhood sizes per center
    :rtype:
        dict[str, str]

    **Workflow**
        1. Build features and labels within ``ROI ∩ valid`` using
           :class:`data.MaskAndFilterBetas`.
        2. Create a voxel-to-column index volume and enumerate centers (ROI voxels).
        3. Generate outer folds via :func:`_folds_for_labels` to mirror the ROI path.
        4. Memmap the full ROI feature matrix once; compute searchlight offsets with
           :func:`_neighbors_ball_mm`.
        5. For each center, call :func:`_one_center` to compute observed score and
           permutation null (optionally in parallel with joblib).
        6. Assemble result volumes and write NIfTIs aligned to the resampled mask grid.
        7. Save a JSON sidecar with key metadata (radius, permutations, CV, etc.).

    **Parallelization**
        - Controlled by ``cfg["parallel"]["n_jobs"]``; when ``> 1``,
          centers are processed with ``joblib.Parallel`` using the ``loky`` backend.
        - Center seeds are drawn from a global RNG initialized by ``seed``.

    **Output Files**
        - ``searchlight_observed.nii.gz``
        - ``searchlight_null_mean.nii.gz``
        - ``searchlight_delta.nii.gz``
        - ``searchlight_pvalue.nii.gz``
        - ``searchlight_nfeatures.nii.gz``
        - ``searchlight_desc-metadata.json`` (radius, permutations, CV config, etc.)

    **Example**
        .. code-block:: python

            out = permutation_searchlight(
                betas_img=betas,
                mask_img=mask,
                trial_list=trials,
                label_mapper={'CS-': 0, 'CS+': 1},
                cfg=cfg['decoding_settings'],
                groups=runs,
                seed=123,
                tmpdir="/tmp/sl-cache",
                save_dir="results/sub-01"
            )
            print(out["delta"])  # path to delta map

    .. note::
       - The maps are written on the same grid/affine as the mask resampled to betas.
       - Neighborhoods respect anisotropic voxel sizes via :func:`_neighbors_ball_mm`.
       - Empirical p-values use the standard ``(+1)/(+1)`` bias correction.
       - The function expects linear (coef_)-based classifiers for interpretability,
         but will still compute scores with non-linear models.
    """

    if groups is not None:
        logger.info(f"Groups = {groups}")

    # 0) Extract settings from cfg
    early_stop_alpha = cfg.get("early_stop_alpha", None)
    sl_cfg = cfg.get("searchlight", {})
    par_cfg = cfg.get("parallel", {})
    radius_mm = sl_cfg.get("radius_mm", 6)
    locked_params = sl_cfg.get("locked", None)
    alpha = sl_cfg.get("alpha", 0.05)

    # 1) Extract X/labels inside ROI∩valid using your existing path
    mf = data.MaskAndFilterBetas(
        betas_img, mask_img,
        trial_list=trial_list,
        label_mapper=label_mapper,
        output_file=output_file
    )

    X = mf.X.astype("float32", copy=False)
    y = np.asarray(mf.labels)
    vol_shape = mf.mask_resampled_to_betas.shape[:3]
    col_index_vol = np.full(vol_shape, -1, np.int32)

    # No guessing about flattening order: just place columns where they belong
    coords = np.unravel_index(np.asarray(mf.roi_linidx, dtype=np.int64), vol_shape, order="C")
    col_index_vol[coords] = np.arange(len(mf.roi_linidx), dtype=np.int32)

    # Centers = every voxel that is actually a feature column
    centers = np.column_stack(np.where(col_index_vol >= 0))  # shape (N, 3)

    # 2) folds exactly like your ROI path
    folds = utils._folds_for_labels(cfg, y, groups)

    # 3) memmap X once
    tmpdir = os.path.expanduser(tmpdir)
    os.makedirs(tmpdir, exist_ok=True)
    tmpd = tempfile.mkdtemp(dir=tmpdir)

    X_path = dump(
        X,
        opj(tmpd, f"Xsl_full_{uuid.uuid4().hex}.joblib"),
        compress=0
    )[0]

    zooms = mf.mask_resampled_to_betas.header.get_zooms()
    offs  = _neighbors_ball_mm(zooms, radius_mm)

    # 5) run per-center
    rng = np.random.default_rng(seed)
    center_seeds = rng.integers(0, 2**32 - 1, size=len(centers), dtype=np.uint32)

    save_dir = tmpdir
    if "save_dir" in kwargs:
        save_dir = opj(kwargs.pop("save_dir"), "searchlight")

    logger.info(f"Storing searchlight information in {save_dir}")
    n_perms = cfg.get("n_permutations", 1000)
    n_jobs = par_cfg.get("n_jobs", 1)
    def _run(ix):
        return _one_center(
            centers[ix], offs, col_index_vol, vol_shape, mf.roi_linidx,
            X_path, y, folds, cfg,
            groups=groups,
            n_perms=n_perms,
            seed=int(center_seeds[ix]),
            locked_params=locked_params,
            searchlight=True,
            save_dir=save_dir,
            **kwargs
        )

    logger.info(f"Centers={len(centers)} | r={radius_mm}mm | perms={n_perms} | jobs={n_jobs}")

    if n_jobs == 1:
        out = [
            _run(i)
            for i in tqdm(
                range(len(centers)),
                total=len(centers),
                disable=utils.tqdm_disabled(),
            )
        ]
    else:
        with tqdm_joblib(
            tqdm(
                total=len(centers),
                disable=utils.tqdm_disabled(),
            )
        ):
            out = Parallel(
                n_jobs=n_jobs,
                backend=par_cfg.get("backend", "loky"),
                prefer=par_cfg.get("prefer", "processes"),
                batch_size=par_cfg.get("batch_size", 16),
                verbose=par_cfg.get("verbose", 0)
            )([delayed(_run)(i) for i in range(len(centers))])

    # 6) assemble maps
    obs_map   = np.full(vol_shape, np.nan, dtype=np.float32)
    null_map  = np.full(vol_shape, np.nan, dtype=np.float32)
    delta_map = np.full(vol_shape, np.nan, dtype=np.float32)
    p_map     = np.full(vol_shape, np.nan, dtype=np.float32)
    nfeat_map = np.zeros(vol_shape, dtype=np.int32)
    nperms_run_map = np.zeros(vol_shape, dtype=np.int32)
    stopped_map    = np.zeros(vol_shape, dtype=np.uint8)
    stop_code_map  = np.zeros(vol_shape, dtype=np.uint8)

    for (x,y,z), obs, nullm, dlt, p, nf, nrun, stopped, stop_code in out:
        obs_map[x,y,z]   = obs
        null_map[x,y,z]  = nullm
        delta_map[x,y,z] = dlt
        p_map[x,y,z]     = p
        nfeat_map[x,y,z] = nf
        nperms_run_map[x,y,z] = nrun
        stopped_map[x,y,z]    = stopped
        stop_code_map[x,y,z]  = stop_code

    # 7) save NIfTIs
    os.makedirs(save_dir, exist_ok=True)
    base = opj(save_dir or tmpdir, "searchlight")

    ref = mf.mask_resampled_to_betas  # SAME grid/affine as vol_shape
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
        "early_stop_alpha": float(early_stop_alpha) if early_stop_alpha is not None else None,
        "n_centers": int(np.isfinite(p_map).sum()),
        "stopped_total": int(np.sum(stopped_map)),
        "stopped_pmin": int(np.sum(stop_code_map == 1)),
        "stopped_pmax": int(np.sum(stop_code_map == 2)),
        "median_nperms_run": float(np.median(nperms_run_map[nperms_run_map > 0])) if (nperms_run_map > 0).any() else 0.0,        
    }

    with open(f"{base}_desc-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

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
    n_perms=None
):
    """
    Save and summarize searchlight decoding results to NIfTI images.

    This function writes the core searchlight maps (``observed``, ``null_mean``,
    ``delta``, ``pvalue``, ``nfeatures``) using the spatial reference of
    ``ref_img``. It also performs voxelwise **FDR correction** of the p-value
    map (within an optional mask), creates visualization-friendly ``-log10(p)``
    maps, **and** saves diagnostics of the early-stopping rule used during
    permutations (number of permutations actually run, whether early-stop
    triggered, and the stop reason).

    **Primary maps**
        - ``observed`` — Mean cross-validated decoding accuracy (often balanced
          accuracy) with *true* labels at each sphere center.

        - ``null_mean`` — Mean decoding accuracy across all *permuted-label*
          repetitions at the same center, estimating the empirical null.

        - ``delta`` — Difference ``observed - null_mean``; an effect-size-like
          contrast indicating how much above null the observed score is.

        - ``pvalue`` — Uncorrected voxelwise p-value computed from the
          permutation distribution (typically one-sided: P[ null ≥ observed ]).
          Values lie in [0, 1]; smaller is stronger evidence.

        - ``nfeatures`` — Number of voxel features included in each sphere.
          Useful to diagnose edge effects or masks that yield too few features.

    **Diagnostic maps (early-stop)**
        - ``nperms_run`` — The **actual** number of permutations executed per
          voxel (≤ the requested maximum, due to early stopping).

        - ``stopped`` — Binary indicator (0/1) whether early stopping triggered
          for that voxel.

        - ``stop_code`` — Encodes the early-stop reason:
          0 = not stopped, 1 = ``p_min > alpha`` (cannot become significant),
          2 = ``p_max < alpha`` (already clearly significant).

    **Derived maps**
        - ``pvalue_fdr`` — Benjamini–Hochberg FDR-corrected p-values within the
          provided mask (or across all finite voxels if no mask is given).

        - ``neglogp`` — ``-log10(pvalue)`` transform of the uncorrected p-map,
          convenient for visualization (e.g., 1.3 ≈ p=0.05, 2 ≈ 0.01, 3 ≈ 0.001).

        - ``neglogp_fdr`` — ``-log10(pvalue_fdr)`` for visualizing corrected
          results.

        - ``sig_uncorrected`` — Thresholded **delta** map where ``pvalue < fdr_alpha``
          (voxels failing the threshold are set to NaN).

        - ``sig_fdr`` — Thresholded **delta** map where ``pvalue_fdr < fdr_alpha``
          (recommended for reporting significant decoding).

    All images are written as NIfTI files that share the header/affine of
    ``ref_img``. Filenames follow ``<base_path>_<mapname>.nii.gz``.

    Parameters
    ----------
    base_path : str
        Output prefix. If ``base_path == '/tmp/sub-01_searchlight'``, the p-map
        is written to ``/tmp/sub-01_searchlight_pvalue.nii.gz`` (and so on).
    ref_img : :class:`nibabel.Nifti1Image`
        Reference image defining the target grid and affine for all outputs
        (typically the resampled mask in beta space).
    observed_map : ndarray, shape (X, Y, Z)
        Observed (true-label) mean CV score per center.
    null_mean_map : ndarray, shape (X, Y, Z)
        Mean permuted-label CV score (empirical null) per center.
    delta_map : ndarray, shape (X, Y, Z)
        Difference ``observed_map - null_mean_map``.
    pvalue_map : ndarray, shape (X, Y, Z)
        Uncorrected voxelwise permutation p-values in [0, 1].
    nfeatures_map : ndarray, shape (X, Y, Z)
        Number of features included in each sphere.
    nperms_run : ndarray, shape (X, Y, Z)
        Number of permutations actually executed per voxel (diagnostic).
    stopped : ndarray, shape (X, Y, Z)
        0/1 map indicating whether early stopping triggered (diagnostic).
    stop_code : ndarray, shape (X, Y, Z)
        0 = not stopped; 1 = ``p_min > alpha``; 2 = ``p_max < alpha`` (diagnostic).
    mask_img : :class:`nibabel.Nifti1Image`, optional
        Binary ROI/brain mask. If provided, FDR correction is applied only
        within this mask; otherwise across all finite p-values.
    n_perms : int or None:
        Number of label permutations performed for the null distribution.
        Used to calculate the percentage of permutations runs based on ``nperms_run``
    fdr_alpha : float, optional
        Target FDR level for both ``pvalue_fdr`` and significance masking
        (default = 0.05).

    Returns
    -------
    out_files : dict
        Mapping from map name to output path for all saved images. Always
        includes: ``'observed'``, ``'null_mean'``, ``'delta'``, ``'pvalue'``,
        ``'nfeatures'``, ``'nperms_run'``, ``'stopped'``, ``'stop_code'``;
        and, when applicable: ``nperms_run_frac``, ``'pvalue_fdr'``, ``'neglogp'``,
        ``'neglogp_fdr'``.

    Notes
    -----
    - P-values are empirical; the smallest attainable value is
      ``1 / (n_permutations_run + 1)`` at each voxel.
    - ``delta`` is convenient for effect visualization; use ``pvalue_fdr`` (or
      ``sig_fdr``) for inferential statements.
    - Early-stopping diagnostics help quantify computational savings and verify
      that stopping criteria behaved as expected.

    Examples
    --------
    >>> ref = mf.mask_resampled_to_betas
    >>> base = os.path.join(output_dir, "searchlight")
    >>> out = save_searchlight_maps(
    ...     base,
    ...     ref_img=ref,
    ...     observed_map=obs_map,
    ...     null_mean_map=null_map,
    ...     delta_map=delta_map,
    ...     pvalue_map=p_map,
    ...     nfeatures_map=nfeat_map,
    ...     nperms_run=nperms_run_map,
    ...     stopped=stopped_map,
    ...     stop_code=stop_code_map,
    ...     mask_img=brain_mask,
    ...     n_perms=n_perms,
    ...     fdr_alpha=0.05,
    ... )
    >>> out["sig_fdr"]
    '/tmp/searchlight/sub-01_searchlight_sig_fdr.nii.gz'
    """

    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    # 1) Save primary maps
    imgs = {
        "observed":   image.new_img_like(ref_img, observed_map.astype(np.float32),  copy_header=True),
        "null_mean":  image.new_img_like(ref_img, null_mean_map.astype(np.float32), copy_header=True),
        "delta":      image.new_img_like(ref_img, delta_map.astype(np.float32),     copy_header=True),
        "pvalue":     image.new_img_like(ref_img, pvalue_map.astype(np.float32),    copy_header=True),
        "nfeatures":  image.new_img_like(ref_img, nfeatures_map.astype(np.float32), copy_header=True),
        "nperms_run": image.new_img_like(ref_img, nperms_run.astype(np.float32), copy_header=True),
        "stopped":    image.new_img_like(ref_img, stopped.astype(np.float32),    copy_header=True),
        "stop_code":  image.new_img_like(ref_img, stop_code.astype(np.float32),  copy_header=True),        
    }

    if n_perms is not None:
        nperms_run_frac = nperms_run/n_perms
        imgs["nperms_run_frac"] = image.new_img_like(ref_img, nperms_run_frac.astype(np.float32), copy_header=True)

    out_files = {}
    for k, v in imgs.items():
        f = f"{base_path}_{k}.nii.gz"
        v.to_filename(f)
        out_files[k] = f

    # 2) Build mask for correction (resample to ref grid if provided)
    if mask_img is not None:
        m = image.resample_to_img(
            mask_img,
            ref_img,
            interpolation="nearest",
            force_resample=True
        ).get_fdata() > 0.5
    else:
        m = np.isfinite(pvalue_map)

    # 3) FDR correction within mask
    p = pvalue_map
    mask_valid = m & np.isfinite(p)
    p_vec = p[mask_valid]

    if p_vec.size > 0:
        rej, p_fdr_vec = fdrcorrection(p_vec, alpha=fdr_alpha)
        p_fdr = np.full_like(p, np.nan, dtype=np.float32)
        p_fdr[mask_valid] = p_fdr_vec

        # 4) -log10 versions (nice for visualization)
        eps = np.finfo(np.float32).tiny
        neglogp     = -np.log10(np.clip(p.astype(np.float32),     eps, 1.0))
        neglogp_fdr = -np.log10(np.clip(p_fdr.astype(np.float32), eps, 1.0))

        # 5) Save derived maps
        extras = {
            "pvalue_fdr": image.new_img_like(ref_img, p_fdr,       copy_header=True),
            "neglogp":    image.new_img_like(ref_img, neglogp,     copy_header=True),
            "neglogp_fdr":image.new_img_like(ref_img, neglogp_fdr, copy_header=True),
        }
        for k, v in extras.items():
            f = f"{base_path}_{k}.nii.gz"
            v.to_filename(f)
            out_files[k] = f

    return out_files