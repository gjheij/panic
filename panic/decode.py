# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import shutil
import numpy as np
import pandas as pd

import tempfile, uuid
from tqdm import tqdm
from joblib import Parallel, delayed, dump, load

from panic import data
from panic.pipeline import create_outer_folds
from panic.utils import (
    tqdm_disabled,
    load_yaml,
    dump_yaml,
    make_analysis_id
)
from panic.logger import get_logger, tqdm_joblib
from panic.searchlight import permutation_searchlight
from panic.errors import EmptyMaskError, NoFeaturesSelectedError
from panic.plugins import core

from lazyfmri.utils import FindFiles, update_kwargs

logger = get_logger(__name__)
opj = os.path.join


def run_decoding_with_permutation(
        X,
        labels,
        folds,
        cfg,
        label_mapper=None,
        seed=0,
        tmpdir=None,
        groups=None,
        **kwargs
    ):
    """
    Run decoding with permutation testing on ROI-level data.

    This function performs cross-validated decoding on a feature matrix ``X``
    to compute the **observed accuracy** and a **null distribution** of accuracies
    obtained by label permutation. The implementation mirrors
    :func:`utils._cv_mean_score` and is optimized for reproducibility and parallelization.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Feature matrix. Typically contains beta values or trialwise features.
    labels : array_like, shape (n_samples,)
        Class labels corresponding to each row of ``X``.
    folds : list of tuple
        List of outer CV splits as tuples ``(train_idx, test_idx)``.
    cfg : dict
        Decoding configuration dictionary used by :func:`utils._cv_mean_score`.
        Should contain entries for ``"estimator"``, ``"cv"``, and optionally
        ``"permute_within_groups"``.
    n_perms : int, optional
        Maximum number of label permutations to compute for the null distribution.
        Default is 1000.
    label_mapper : dict
        Mapping from trial labels (strings) to integer class labels,
        e.g. ``{'CS-': 0, 'CS+': 1}``.        
    n_jobs : int, optional
        Number of parallel workers for permutation testing. Default is 1 (serial execution).
    seed : int, optional
        Master random seed for permutation reproducibility.
    tmpdir : str, optional
        Directory used to store temporary memory-mapped arrays
        (default: ``~/.joblib_cache``).
    groups : array_like, optional
        Optional group vector (e.g., run or session IDs) for group-aware CV
        and within-group permutations.
    **kwargs :
        Additional keyword arguments passed to :func:`utils._cv_mean_score`
        (e.g., ``save_dir`` for saving fold-specific models).

    Returns
    -------
    dict
        Dictionary containing observed and permuted decoding results:

        * ``"observed"`` – Observed mean CV score (float)
        * ``"permuted"`` – Array of permutation scores (shape ``[n_run]``)
        * ``"mean_permuted"`` – Mean permutation score (float)
        * ``"delta"`` – Observed − null mean difference (float)
        * ``"p"`` – Empirical one-tailed p-value
        * ``"n_run"`` – Number of permutations executed
        * ``"n_perms"`` – Number of requested permutations

    Workflow
    ---------
    1. Dumps the input feature matrix ``X`` to a temporary uncompressed
    ``joblib`` file for memory-mapped access.
    2. Computes the **observed** mean cross-validated score using
    :func:`utils._cv_mean_score`.
    3. Generates ``n_perms`` random seeds from a reproducible master RNG.
    4. Recomputes CV scores under label permutation, either serially or
    in parallel via :class:`joblib.Parallel`.
    5. Aggregates permutation scores and computes empirical p-value:
    ::

        p = (sum(perm >= obs) + 1) / (n_run + 1)

    Statistical Outputs
    -------------------
    - **Observed** – Mean CV accuracy for the true (non-permuted) labels.
    - **Permuted** – Null distribution of accuracies from permuted labels.
    - **Mean permuted** – Average null accuracy (expected under the null).
    - **Δ (delta)** – Observed − null mean difference (effect size).
    - **p-value** – Empirical one-tailed significance level.
    - **n_run** – Number of permutations executed. In the ROI decoder this is
      normally equal to ``n_perms`` because all requested permutations are run.

    Null-Distribution Behavior
    --------------------------
    - The ROI decoder uses a fixed-count permutation procedure: every requested
      permutation is evaluated and retained.
    - This makes the null mean, empirical p-value, and saved permutation table
      directly interpretable because they are based on the same number of
      permutations for every ROI.
    - Searchlight decoding follows the same fixed-count null-mean principle in
      Bach-style mode, but usually with far fewer permutations and group-level
      inference downstream.

    Example
    -------
    .. code-block:: python

        result = run_decoding_with_permutation(
            X=X,
            labels=y,
            folds=folds,
            cfg=cfg,
            groups=runs,
            seed=42,
            save_dir="results/sub-01"
        )

        print(result["observed"], result["p"], result["n_run"])
        # 0.73, 0.012, 840

    Notes
    -----
    - The function uses ``np.random.default_rng`` for reproducible random streams.
    - All permutations share the same CV folds as the observed analysis.
    - Empirical p-values use the standard ``(+1)/(+1)`` finite-sample correction.
    - When ``n_jobs > 1``, batches are processed in parallel via Joblib's ``loky`` backend.
    - Ideal for ROI- or whole-brain decoding when voxelwise searchlight is unnecessary.
    """

    if tmpdir is None:
        tmpdir = os.environ.get("TMPDIR", "~/.joblib_cache")

    tmpdir = os.path.expanduser(tmpdir)
    os.makedirs(tmpdir, exist_ok=True)
    
    # read settings for parallellization
    par_cfg = cfg.get("parallel", {})

    # compress=0 keeps it as a plain .npy-like file for fast mmap
    with tempfile.TemporaryDirectory(dir=tmpdir, prefix="panic_") as tmpd:
        X_path = dump(
            X,
            os.path.join(tmpd, f"X_mm_{uuid.uuid4().hex}.joblib"),
            compress=0
        )[0]

        labels = np.asarray(labels)
        labels_path = dump(
            labels,
            os.path.join(tmpd, f"labels_{uuid.uuid4().hex}.joblib"),
            compress=0
        )[0]

        X_mm = load(X_path, mmap_mode="r")
        logger.info(f"X={X_mm.shape} (n_samples, n_features)")

        # config takes precedence over the presence of groups
        if not cfg.get("permute_within_groups", False):
            if groups is not None:
                logger.info("Groups (or runs) were detected, but permute_within_groups=False, so ignoring groups..")

            groups = None
        
        # log them
        if groups is not None:
            logger.info(f"Groups = {groups}")

        groups = None if groups is None else np.asarray(groups)

        # observed
        if "save_dir" in kwargs:
            logger.info(f"Storing fold information in {kwargs['save_dir']}")

        plugin, plugin_kwargs = core.get_analysis_plugin(
            cfg,
            label_dict=label_mapper
        )

        analysis_cfg = cfg.get("analysis", {})
        score_name = analysis_cfg.get("type", "decoding")

        logger.info(f"Running plugin [{score_name}]: {plugin} with args: {plugin_kwargs}")
        do_permutations = bool(analysis_cfg.get("permutations", True))
        higher_is_better = bool(analysis_cfg.get("higher_is_better", True))

        try:
            result  = plugin(
                X_path,
                labels,
                cfg=cfg,
                folds=folds,
                groups=groups,
                permute=False,
                return_artifacts=True,
                **plugin_kwargs,
                **kwargs,
            )

            observed_acc, artifacts = core.unpack_plugin_result(result)

        except NoFeaturesSelectedError as e:
            logger.warning(f"Observed analysis failed (no features): {e}. Skipping ROI.")
            return None

        logger.info("%s observed after %d folds: %.4f", score_name, len(folds), observed_acc)

        # save plugin-specific artifacts
        if "save_dir" in kwargs:
            core.save_analysis_artifacts(
                kwargs["save_dir"],
                artifacts,
                roi_linidx=kwargs.get("roi_linidx"),
            )
        
        # check if we should do permutations
        n_perms = int(cfg.get("n_permutations", 1000))
        n_jobs = par_cfg.get("n_jobs", 1)

        if do_permutations and n_perms > 0:
            rng = np.random.default_rng(seed)
            seeds = rng.integers(0, 2**32 - 1, size=n_perms, dtype=np.uint32)

            def _one_perm(seed):
                return plugin(
                    X_path,
                    labels,
                    cfg=cfg,
                    folds=folds,
                    groups=groups,
                    permute=True,
                    rng=np.random.default_rng(int(seed)),
                    **plugin_kwargs,
                    **kwargs,
                )

            logger.info("Starting permutation testing: n_perms=%d, n_jobs=%d", n_perms, n_jobs)

            if n_jobs == 1:
                permuted_acc = [
                    _one_perm(s)
                    for s in tqdm(seeds, total=n_perms, disable=tqdm_disabled())
                ]
            else:
                with tqdm_joblib(tqdm(total=n_perms, disable=tqdm_disabled())):
                    permuted_acc = Parallel(
                        n_jobs=n_jobs,
                        backend=par_cfg.get("backend", "loky"),
                        prefer=par_cfg.get("prefer", "processes"),
                        batch_size=par_cfg.get("batch_size", 16),
                        verbose=par_cfg.get("verbose", 0),
                    )([delayed(_one_perm)(s) for s in seeds])
        else:
            permuted_acc = []

        permuted = np.asarray(permuted_acc, dtype=float)
        n_run = len(permuted)

        mean_permuted = float(np.nanmean(permuted)) if n_run else float("nan")
        delta = float(observed_acc - mean_permuted) if n_run else float("nan")

        if n_run:
            if higher_is_better:
                p_val = (np.sum(permuted >= observed_acc) + 1) / (n_run + 1)
            else:
                p_val = (np.sum(permuted <= observed_acc) + 1) / (n_run + 1)
        else:
            p_val = float("nan")

        logger.info(
            "%s complete. Observed=%.4f | Null=%.4f | Δ=%.4f | p=%s | n_run=%d",
            score_name,
            observed_acc,
            mean_permuted,
            delta,
            f"{p_val:.4f}" if np.isfinite(p_val) else "nan",
            n_run,
        )

        ddict = {
            "analysis": score_name,
            "observed": float(observed_acc),
            "permuted": permuted,
            "mean_permuted": mean_permuted,
            "delta": delta,
            "p": float(p_val),
            "n_run": int(n_run),
            "n_perms": int(n_perms) if do_permutations else 0,
            "permutations": bool(do_permutations),
            "higher_is_better": bool(higher_is_better),
        }

        return ddict


def _searchlight_outputs_exist(sl_dir, hemi_key=None):

    if hemi_key is None or hemi_key == "uni":
        base = "searchlight"
    else:
        base = f"searchlight_hemi-{hemi_key}"

    expected = [
        f"{base}_observed.nii.gz",
        f"{base}_null_mean.nii.gz",
        f"{base}_delta.nii.gz",
        f"{base}_pvalue.nii.gz",
        f"{base}_nfeatures.nii.gz",
        f"{base}_desc-metadata.json",
    ]

    all_exist = all(os.path.exists(opj(sl_dir, f)) for f in expected)
    return all_exist, [opj(sl_dir, f) for f in expected]
        
        
class ClassifySubject(data.PrepareBetas):

    """
    High-level interface to run decoding for a single subject.

    This class ties together configuration loading, data preparation, and
    decoding options (ROI vs. searchlight). It inherits from
    :class:`data.PrepareBetas` so that beta preparation utilities are available
    on the instance. The constructor reads a YAML config, prepares directories,
    and stores decoding settings for later steps (e.g., ``_init_betas()``,
    ``decode_single_mask()``, searchlight routines).

    :param str subject:
        Subject identifier (e.g., ``"sub-01"``).
    :param str config_file:
        Path to a YAML configuration file containing ``general_settings`` and
        ``decoding_settings`` sections.
    :param bool save_imgs:
        If ``True``, intermediate/merged images may be written to disk by
        downstream steps. Default: ``False``.
    :param bool searchlight:
        If ``True``, enables searchlight decoding mode (feature selection is
        disabled within the searchlight pipeline). Default: ``False``.
    :param kwargs:
        Reserved for future extensions; forwarded where appropriate in
        downstream methods.

    **Configuration**
        The YAML file is expected to define:
        - ``general_settings``: includes paths (e.g., ``save_dir``), method
          choices (e.g., LSA/LSS), standardization flags, etc.
        - ``decoding_settings``: estimator configuration, CV specs, feature
          selection, variance threshold, and (optionally) searchlight
          parameters.

    **Attributes**
        - ``subject`` : str  
          The subject identifier.
        - ``config_file`` : str  
          Path to the YAML configuration file.
        - ``bids_id`` : str  
          Subject numeric/string suffix (``subject.split("-")[-1]``).
        - ``save_imgs`` : bool  
          Whether downstream steps should persist images.
        - ``searchlight`` : bool  
          Toggle for searchlight decoding mode.
        - ``cfg`` : dict  
          Parsed YAML configuration.
        - ``gen_settings`` : dict  
          ``cfg["general_settings"]`` convenience reference.
        - ``dec_settings`` : dict  
          ``cfg["decoding_settings"]`` convenience reference.
        - ``dec_settings_searchlight`` : dict  
          Copy of ``dec_settings`` adapted for searchlight runs: ROI-level
          feature selection and grid search are disabled, and fail-fast
          permutation stopping is disabled for fixed-count null estimation.
        - ``save_dir`` : str  
          Output directory for the subject (``<save_dir>/<subject>``), created
          if missing.

    **Example**
        .. code-block:: python

            clf = ClassifySubject(
                subject="sub-01",
                config_file="config.yml",
                save_imgs=True,
                searchlight=False
            )

            # Later in the workflow:
            clf._init_betas()           # prepare betas from config
            results = clf.decode_single_mask(
                betas=clf.betas,
                mask="roi_amygdala.nii.gz",
                trial_list=clf.trial_list,
                label_mapper=clf.cfg["label_dict"],
                groups=getattr(clf, "groups", None),
                save_dir=clf.save_dir
            )

    **Notes**
        - ``dec_settings_searchlight`` is used to ensure that searchlight
          decoding uses all voxels in each sphere, fixed estimator settings,
          and fixed-count permutations for null-mean estimation.
        - Logging provides a reproducible record of the configuration used for
          each subject, including the full decoding settings dictionary.
    """

    def __init__(
            self,
            subject,
            config_file,
            save_imgs=False,
            searchlight=False,
            **kwargs
        ):
        
        # init
        self.subject = subject
        self.config_file = config_file
        self.bids_id = str(self.subject.split("-")[-1])
        self.save_imgs = save_imgs
        self.searchlight = searchlight

        # load settings
        logger.info(f"Running decoding for {self.subject}")
        logger.info(f"Loading settings from {self.config_file}")
        self.cfg = load_yaml(self.config_file)

        # append subject to save_dir
        self.gen_settings = self.cfg["general_settings"]
        self.dec_settings = self.cfg["decoding_settings"]
        self.roi_settings = self.cfg["roi_settings"]

        # Searchlight uses local spherical neighborhoods as the feature definition.
        # Disable ROI-level feature selection and grid search so each center uses
        # all voxels in the sphere with fixed estimator settings.  Permutations
        # are fixed-count null-mean estimates; fail-fast is intentionally disabled
        # for searchlight because Bach-style inference is performed at group level.
        self.dec_settings_searchlight = {
            **self.dec_settings,
            "feature_selection": None,
            "gridsearch": None
        }

        # set output directory based on analysis name
        # e.g., 'decoding'/'cs_us_similarity'/'dimensionality'
        
        analysis = self.dec_settings["analysis"]

        # name can be different
        self.analysis_name = analysis.get("name") or analysis.get("type")
        self.analysis_type = analysis.get("type")

        # create unique hash based on settings
        self.analysis_id = make_analysis_id(
            analysis_name=self.analysis_name,
            analysis_type=self.analysis_type,
            source=self.gen_settings["source"],
            method=self.gen_settings["method"],
            standardize=self.gen_settings.get("standardize"),
        )
        logger.info(f"Analysis ID: {self.analysis_id}")

        self.save_dir = opj(
            self.gen_settings["save_dir"],
            self.analysis_name,
            self.subject,
        )

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
        logger.info(f"Decoding configuration:")
        logger.info(self.dec_settings)
        if self.searchlight:
            logger.info("Searchlight configuration (effective):")
            logger.info(self.dec_settings_searchlight)


    def _fit(self, **kwargs):
        """
        Run the end-to-end decoding workflow for the current subject.

        This internal driver method initializes beta data, harmonizes decoding
        keyword arguments (injecting ``standardize`` and ``groups`` from the
        object’s state), and then calls :meth:`decode_masks`. Results are stored
        on ``self.results``. Any exception is logged with traceback and re-raised.

        :param kwargs:
            Additional keyword arguments forwarded to :meth:`decode_masks`
            (e.g., ``save_dir``, estimator/pipeline options). If ``standardize``
            or ``groups`` are not present in ``kwargs``, they are filled from
            ``self.do_standardization`` and ``self.groups`` respectively.

        :returns:
            ``None``. Side effect: sets ``self.results`` to the output of
            :meth:`decode_masks`.
        :rtype:
            None

        **Behavior**
            1. Initialize betas and related state via :meth:`_init_betas`.
            2. Prepare a minimal arg set:
            - ``standardize`` ← ``self.do_standardization``
            - ``groups`` ← ``self.groups``
            and merge into ``kwargs`` using :func:`update_kwargs` (existing user
            values are preserved).
            3. Execute :meth:`decode_masks(**kwargs)`` and assign to ``self.results``.
            4. Log success; on failure, log the full exception and re-raise.

        **Side Effects**
            - Populates ``self.results`` with the decoding outputs.
            - Emits informative log messages for progress and errors.

        **Raises**
            - Propagates any exception raised during initialization or decoding
            after logging via ``logger.exception``.
        """

        try:
            # load betas
            self._init_betas()

            # decode hemispheres
            set_kwargs = {
                "standardize": self.do_standardization,
                "groups": self.groups
            }

            for key, val in set_kwargs.items():
                kwargs = update_kwargs(
                    kwargs,
                    key,
                    val
                )

            self.results = self.decode_masks(**kwargs)
            logger.info(f"Decoding {self.subject} complete")
        except Exception as e:
            logger.exception(f"Decoding failed with errors: {e}")
            raise


    def _init_betas(self, **kwargs):
        """
        Initialize and prepare subject-specific beta images for decoding.

        This internal helper constructs a complete keyword argument dictionary
        describing how to locate and preprocess trialwise beta estimates, then
        delegates loading and sanitization to :class:`data.PrepareBetas`.

        The resulting beta image, trial list, and metadata (e.g., run groups and
        standardization flag) are attached to the current object for downstream
        decoding.

        :param kwargs:
            Optional keyword arguments to override or extend default values derived
            from ``self.subject``, ``self.gen_settings``, and ``self.cfg``.
            These may include:
            - ``subject`` – subject identifier
            - ``beta_dir`` – root directory containing beta files
            - ``derivative`` – whether to use temporal derivative estimates
            - ``model`` – model type (e.g., ``"lsa"`` or ``"lss"``)
            - ``standardize`` – whether to standardize beta values
            - ``label_mapper`` – mapping from condition labels to numeric codes

        :returns:
            ``None``. This method initializes the parent :class:`PrepareBetas`
            class, populating its attributes such as:
            - ``self.betas`` (4D NIfTI image of betas)
            - ``self.trial_list`` (trial names per beta volume)
            - ``self.do_standardization`` (flag for further scaling)
            - ``self.groups`` (optional run grouping)
        :rtype:
            None

        **Processing Steps**
            1. Construct a default configuration dictionary ``ddict`` containing:
            - Project- and subject-specific paths and parameters from
                ``self.gen_settings`` and ``self.cfg``.
            2. Merge these defaults into any user-provided ``kwargs`` via
            :func:`update_kwargs` (user-specified keys take precedence).
            3. Initialize :class:`data.PrepareBetas` directly with the merged
            parameters, which handles beta loading, concatenation, and
            sanitization.

        **Example**
            .. code-block:: python

                self._init_betas(
                    subject="sub-01",
                    model="lsa",
                    label_mapper={"CS-": 0, "CS+": 1}
                )
                print(self.betas.shape, len(self.trial_list))

        **Notes**
            - This method is typically invoked inside :meth:`_fit`.
            - Ensures that the decoding object inherits all attributes of
            :class:`PrepareBetas` (by direct class initialization).
            - Project directory and source type are pulled from
            ``self.gen_settings["project_dir"]`` and
            ``self.gen_settings["source"]``.
        """

        beta_dir = opj(
            self.gen_settings["project_dir"],
            "derivatives",
            self.gen_settings["source"]
        )

        assert os.path.exists(beta_dir), FileNotFoundError(f"Beta directory '{beta_dir}' does not exist")

        ddict = {
            "subject": self.subject,
            "beta_dir": beta_dir,
            "derivative": self.cfg.get("fitted_derivative", False),
            "model": self.gen_settings.get("method", "lsa"),
            "standardize": self.gen_settings.get("standardize", False),
            "label_mapper": self.cfg.get("label_dict"),
            "filters": self.gen_settings.get("filters", None),
            "save_imgs": self.gen_settings.get("save_imgs", False),
        }

        for key, val in ddict.items():
            kwargs = update_kwargs(
                kwargs,
                key,
                val
            )

        data.PrepareBetas.__init__(self, **kwargs)

        # save trial order csv
        self.trial_order_file = opj(self.save_dir, f"{self.subject}_desc-trial_order.csv")
        if hasattr(self, "events_df") and self.events_df is not None:
            self.events_df.to_csv(self.trial_order_file, index=False)
            logger.info(f"Saved trial order CSV to {self.trial_order_file}")
        else:
            logger.warning("No events_df available to save trial order CSV.")


    def decode_single_mask(
            self,
            betas,
            mask,
            trial_list=None,
            label_mapper=None,
            output_file=None,
            groups=None,
            **kwargs
        ):
        """
        Run decoding analysis on a single ROI mask or perform searchlight decoding.

        This method serves as a unified entry point for both ROI-level and
        searchlight-level decoding workflows. Depending on the object's
        configuration (``self.searchlight``), it either:
        
        - Executes ROI-based decoding with permutation testing using
        :func:`run_decoding_with_permutation`, or
        - Performs voxelwise searchlight decoding using
        :func:`permutation_searchlight`.

        :param nibabel.Nifti1Image | str betas:
            4D beta image containing trialwise activation maps (shape: X×Y×Z×N).
        :param nibabel.Nifti1Image | str mask:
            Binary ROI or brain mask aligned to the betas image.  
            For searchlight decoding, this defines the volume of possible centers.
        :param list[str] | numpy.ndarray trial_list:
            List of trial or condition identifiers, one per beta volume.
        :param dict label_mapper:
            Dictionary mapping trial labels (e.g., condition names) to integer class labels,  
            such as ``{'CS-': 0, 'CS+': 1}``.
        :param str | None output_file:
            Optional path to save intermediate outputs or extracted features.
        :param array_like groups:
            Optional group labels (e.g., run or session indices) for group-aware
            cross-validation and permutations.
        :param kwargs:
            Additional keyword arguments passed through to
            :func:`run_decoding_with_permutation` or
            :func:`permutation_searchlight`.

        :returns:
            Dictionary containing decoding results.
            
            * For ROI decoding: output of :func:`run_decoding_with_permutation`
            (observed accuracy, permutation scores, delta, p-value, etc.)
            * For searchlight decoding: output of :func:`permutation_searchlight`
            (paths to observed, null, delta, and p-value NIfTI maps)
        :rtype:
            dict

        **Behavior**
            - If ``self.searchlight`` is ``False``:
            1. Extracts voxel features and labels within the ROI using
                :func:`extract_betas_from_rois`.
            2. Defines outer cross-validation folds:
                * If ``permute_within_groups=True`` in ``self.dec_settings``,
                folds are generated by :func:`factory.cv_from_config`.
                * Otherwise, falls back to the legacy "every-third" within-class
                folding scheme via :func:`create_outer_folds`.
            3. Runs decoding with label permutation testing via
                :func:`run_decoding_with_permutation`.
            - If ``self.searchlight`` is ``True``:
            Performs voxelwise searchlight decoding with permutation testing via
            :func:`permutation_searchlight`.

        **Example**
            .. code-block:: python

                # ROI-based decoding
                results = decoder.decode_single_mask(
                    betas="sub-01_betas.nii.gz",
                    mask="roi_amygdala.nii.gz",
                    trial_list=trials,
                    label_mapper={"CS-": 0, "CS+": 1},
                    groups=runs,
                    save_dir="results/sub-01"
                )
                print(results["observed"], results["p"])

            .. code-block:: python

                # Searchlight decoding
                decoder.searchlight = True
                maps = decoder.decode_single_mask(
                    betas="sub-01_betas.nii.gz",
                    mask="brain_mask.nii.gz",
                    trial_list=trials,
                    label_mapper={"CS-": 0, "CS+": 1},
                    groups=runs,
                    save_dir="results/sub-01"
                )
                print(maps["delta"])  # path to delta NIfTI

        .. note::
        - The decoding mode (ROI vs. searchlight) is determined by
            ``self.searchlight``.
        - Cross-validation is defined by ``self.dec_settings["outer_cv"]``.
        - If ``permute_within_groups=True``, groups must be provided.
        """

        # standard ROI analysis
        if not self.searchlight:

            try:
                extract = self.extract_betas_from_rois(
                    betas,
                    mask,
                    trial_list=trial_list,
                    label_mapper=label_mapper,
                    output_file=output_file
                )
            except (EmptyMaskError, NoFeaturesSelectedError, ValueError) as e:
                logger.warning(f"Skipping ROI: {e}")
                return None, None

            # outer folds
            folds = create_outer_folds(
                self.dec_settings,
                extract.labels,
                groups=groups
            )

            # run classifier
            logger.info(f"Feature mapper: {label_mapper}")
            logger.info(f"Labels: {np.unique(extract.labels)} features (n={len(extract.labels)})")

            try:
                ddict = run_decoding_with_permutation(
                    extract.X,
                    extract.labels,
                    folds,
                    self.dec_settings,
                    label_mapper=label_mapper,
                    groups=groups,
                    mask=extract.mask_resampled_to_betas,
                    trial_order_path=self.trial_order_file,
                    roi_linidx=extract.roi_linidx,
                    **kwargs
                )
            except (NoFeaturesSelectedError, ValueError) as e:
                # expected-ish failures for tiny ROIs / strict selection / degenerate folds
                logger.warning(f"Skipping ROI: {e}")
                return None, extract
            except Exception:
                # unexpected failure: log full traceback but still skip mask
                logger.exception("Decoding failed unexpectedly; skipping ROI")
                return None, extract
        else:
            # Searchlight (same same, but different)
            try:
                extract = None
                ddict = permutation_searchlight(
                    betas,
                    mask,
                    trial_list,
                    label_mapper, 
                    self.dec_settings_searchlight,
                    groups=groups,
                    output_file=output_file,
                    **kwargs
                )
            except (NoFeaturesSelectedError, ValueError) as e:
                # expected-ish failures for tiny ROIs / strict selection / degenerate folds
                logger.warning(f"Skipping ROI: {e}")
                return ddict, None
            except Exception:
                logger.exception("Searchlight failed unexpectedly; skipping mask")
                return ddict, None            

        return ddict, extract
    

    def define_mask_inputs(self):

        # roi_dict options:
        #   1. directory -> *.nii.gz files
        #   2. dict -> FreeSurfer labels
        #   3. file -> assume mask

        is_labels = False
        roi_dict = self.cfg["roi_dict"]
        run_dict = {}
        if isinstance(roi_dict, str):
            assert os.path.exists(roi_dict), FileNotFoundError(f"Input ROI-directory '{roi_dict}' does not exist")
            if os.path.isdir(roi_dict):
                logger.info(f"Defining ROIs from directory: '{roi_dict}'")
                extension = self.cfg["roi_settings"].get("extension", ".nii.gz")
                mask_files = FindFiles(
                    roi_dict,
                    extension=extension,
                    maxdepth=0
                ).files

                if isinstance(mask_files, str):
                    mask_files = [mask_files]

                if not isinstance(mask_files, list):
                    raise TypeError(f"We should have a list by now, not {type(mask_files)}..")
                else:
                    if len(mask_files)<1:
                        raise ValueError(f"ROI-list from '{roi_dict}' with extension '{extension}' is empty..")
                    
                n_digits = len(str(len(mask_files)))
                for ix, m in enumerate(mask_files):
                    lbl = f"mask_{str(ix+1).zfill(n_digits)}"
                    run_dict[lbl] = m
            else:
                logger.info(f"Defining single ROI from file: '{roi_dict}'")
                run_dict = {
                    "mask_1": self.cfg["roi_dict"]
                }
        elif isinstance(roi_dict, dict):
            logger.info(f"Defining ROIs dictionary: '{roi_dict}'")
            run_dict = roi_dict
            is_labels = True                    
        else:
            raise TypeError(f"roi_dict input must be a dict or str, not {type(roi_dict)}")
        
        assert len(run_dict)>0, ValueError(f"No ROIs were detected using: '{roi_dict}'")
        return run_dict, is_labels
    

    def decode_masks(
        self,
        hemi_key="hemi",
        **kwargs,
    ):
        """Decode all configured ROIs or searchlight masks.

        Existing ROI-level result files are handled incrementally. When a previous
        results CSV exists and ``overwrite=False``, each ROI is checked against the
        rows already present in that file. Completed ROIs are skipped, while newly
        added ROIs are decoded and appended to the existing results.

        Parameters
        ----------
        hemi_key : str, default="hemi"
            Column name used for hemisphere information.

        **kwargs
            Additional keyword arguments forwarded to
            :meth:`decode_single_mask`.

        Returns
        -------
        pandas.DataFrame or dict
            ROI-level results DataFrame when ``self.searchlight=False``.
            Searchlight output mapping when ``self.searchlight=True``.
        """
        out_files = {}
        results = []
        null = []

        overwrite = bool(
            getattr(
                self,
                "overwrite",
                self.gen_settings.get("overwrite", False),
            )
        )

        # Existing ROI-level outputs
        existing_results = pd.DataFrame()
        completed_rois = set()

        if not self.searchlight:
            res_file = self.generate_filename(
                desc="results",
                ext="csv",
            )

            if os.path.exists(res_file) and not overwrite:
                logger.info(
                    "Loading existing ROI results from '%s'",
                    res_file,
                )

                # ensure representation doesn't change; not using str causes "015" -> "15"
                existing_results = pd.read_csv(
                    res_file,
                    dtype={
                        "subject": str,
                        "roi": str,
                        "source": str,
                        "method": str,
                        "analysis": str,
                        "analysis_type": str,
                        "analysis_id": str
                    },
                )

                required = {
                    "subject",
                    "analysis_id",
                    "roi",
                    hemi_key
                }

                missing = required - set(existing_results.columns)

                if missing:
                    logger.warning(
                        "Existing results file is missing columns required "
                        "for incremental ROI matching: %s. Existing rows will "
                        "be preserved, but no ROIs will be assumed complete.",
                        sorted(missing),
                    )
                else:
                    for _, row in existing_results.iterrows():
                        completed_rois.add(
                            (
                                str(row["subject"]),
                                str(row["analysis_id"]),
                                str(row["roi"]),
                                str(row[hemi_key]),
                            )
                        )

                    logger.info(
                        "Found %d completed ROI entries",
                        len(completed_rois),
                    )

        roi_dict, is_labels = self.define_mask_inputs()

        for r_key, r_val in roi_dict.items():
            logger.info(
                "Processing '%s' (labels|file=%s)",
                r_key,
                r_val,
            )

            roi_name = r_key if is_labels else None

            _, roi_masks = self.prepare_rois(
                r_val,
                roi_name=roi_name,
            )

            for h_key, h_val in roi_masks.items():
                roi_label = str(h_val[0])
                hemi_label = str(h_key)
                source = str(self.gen_settings["source"])
                method = str(self.gen_settings["method"])

                # Incremental ROI skipping
                current_key = (
                    str(self.subject),
                    self.analysis_id,
                    roi_label,
                    hemi_label,
                )

                if (
                    not self.searchlight
                    and not overwrite
                    and current_key in completed_rois
                ):
                    logger.warning(
                        "ROI already present in existing results; skipping: "
                        "roi=%s | %s=%s [entry=%s]",
                        roi_label,
                        hemi_key,
                        hemi_label,
                        current_key
                    )
                    continue

                logger.info(
                    "hemi-key=%s | roi-key=%s",
                    hemi_label,
                    roi_label,
                )

                model_src_dir = opj(
                    self.save_dir,
                    f"model-{method}",
                    f"source-{source}",
                )

                roi_dir = opj(
                    model_src_dir,
                    f"roi-{roi_label}",
                )

                # Searchlight completion check
                if self.searchlight:
                    sl_dir = opj(
                        roi_dir,
                        self.dec_settings["searchlight"].get(
                            "basepath",
                            "searchlight",
                        ),
                    )

                    all_exist, sl_files = _searchlight_outputs_exist(
                        sl_dir,
                        hemi_key=h_key,
                    )

                    if all_exist and not overwrite:
                        logger.warning(
                            "Searchlight outputs already exist for ROI "
                            "'%s'; skipping: %s",
                            roi_label,
                            sl_dir,
                        )

                        out_files[roi_label] = sl_files
                        continue

                    logger.info(
                        "Searchlight outputs missing or overwrite=True "
                        "for ROI '%s'; running decoding.",
                        roi_label,
                    )

                # Optional resampled mask output
                fname = None

                if self.save_imgs:
                    resampled_dir = opj(
                        self.save_dir,
                        "rois",
                    )

                    os.makedirs(
                        resampled_dir,
                        exist_ok=True,
                    )

                    fname = opj(
                        resampled_dir,
                        (
                            f"{self.subject}"
                            f"_roi-{roi_label}"
                            f"_hemi-{hemi_label}"
                            f"_desc-valid_mask.nii.gz"
                        ),
                    )

                # Decode
                ddict, extracted = self.decode_single_mask(
                    self.betas,
                    h_val[1],
                    trial_list=self.trial_list,
                    label_mapper=self.cfg["label_dict"],
                    output_file=fname,
                    save_dir=roi_dir,
                    tmpdir=self.gen_settings["tmp_dir"],
                    hemi_key=h_key,
                    **kwargs,
                )

                if ddict is None:
                    continue

                # ROI results
                if not self.searchlight:
                    results_dict = {
                        "subject": str(self.subject),
                        "observed_acc": ddict["observed"],
                        "null_mean": ddict["mean_permuted"],
                        "delta": ddict["delta"],
                        "p_value": ddict["p"],
                        hemi_key: hemi_label,
                        "roi": roi_label,
                        "source": source,
                        "method": method,
                        **extracted.roi_metrics,
                    }

                    results_dict = self._extend_results_dict(
                        results_dict,
                        extracted=extracted,
                        ddict=ddict,
                        groups=self.groups,
                    )

                    results.append(results_dict)

                    null.extend(
                        {
                            "subject": str(self.subject),
                            "analysis": str(self.analysis_name),
                            "analysis_type": str(self.dec_settings["analysis"]["type"]),
                            "analysis_id": str(self.analysis_id),
                            "roi": roi_label,
                            hemi_key: hemi_label,
                            "perm": i,
                            "acc": float(a),
                            "source": source,
                            "method": method,
                            "n_samples": int(extracted.X.shape[0]),
                            "n_features": int(extracted.X.shape[1]),
                            "outer_cv_mode": self.dec_settings.get(
                                "outer_cv", {}
                            ).get("mode", "sklearn"),
                            "outer_cv_name": self.dec_settings.get(
                                "outer_cv", {}
                            ).get("name"),
                            "scoring": self.dec_settings.get(
                                "scoring",
                                "balanced_accuracy",
                            ),
                        }
                        for i, a in enumerate(ddict["permuted"])
                    )

                else:
                    out_files[roi_label] = ddict

        # Save config
        cfg_file = self.generate_filename(
            id=self.analysis_id,
            desc="config",
            ext="yml",
        )

        self.cfg["analysis_id"] = self.analysis_id
        dump_yaml(
            self.cfg,
            cfg_file,
        )

        shutil.copymode(
            self.config_file,
            cfg_file,
        )

        logger.info(
            "Writing config to: '%s'",
            cfg_file,
        )

        # Save ROI-level outputs
        if not self.searchlight:
            new_results = pd.DataFrame(results)

            if not existing_results.empty:
                res_df = pd.concat(
                    [
                        existing_results,
                        new_results,
                    ],
                    ignore_index=True,
                )
            else:
                res_df = new_results

            logger.info(
                "Writing results to: '%s'",
                res_file,
            )

            res_df.to_csv(
                res_file,
                index=False,
            )

            shutil.copymode(
                self.config_file,
                res_file,
            )

            # Null distributions
            null_file = self.generate_filename(
                desc="null_distribution",
                ext="csv",
            )

            new_null = pd.DataFrame(null)

            if (
                os.path.exists(null_file)
                and not overwrite
            ):
                existing_null = pd.read_csv(
                    null_file
                )

                null_df = pd.concat(
                    [
                        existing_null,
                        new_null,
                    ],
                    ignore_index=True,
                )

            else:
                null_df = new_null

            logger.info(
                "Writing null-distributions to '%s'",
                null_file,
            )

            null_df.to_csv(
                null_file,
                index=False,
            )

            shutil.copymode(
                self.config_file,
                null_file,
            )

            return res_df

        return out_files


    def generate_filename(
        self,
        *,
        desc=None,
        ext="csv",
        **entities,
    ):
        """Generate an output filename from arbitrary analysis entities.

        Parameters
        ----------
        desc : str, optional
            Description appended as the ``desc`` entity.

        ext : str, default="csv"
            File extension, with or without a leading period.

        **entities
            Arbitrary key-value entities to include in the filename, for example
            ``analysis``, ``model``, ``source``, ``method``, ``roi``, or ``hemi``.

            Entities with a value of ``None`` are omitted. Values are added in
            insertion order.

        Returns
        -------
        str
            Full path inside ``self.save_dir``.

        Examples
        --------
        >>> self.generate_filename(
        ...     analysis="SVM_CSm_v_CSpu",
        ...     model="LSA",
        ...     source="stglm",
        ...     desc="results",
        ... )
        '.../sub-015_analysis-SVM_CSm_v_CSpu_model-LSA_source-stglm_desc-results.csv'

        >>> self.generate_filename(
        ...     analysis_type="classification",
        ...     analysis_name="SVM_CSm_v_CSpu",
        ...     roi="bl",
        ...     hemi="left",
        ...     desc="results",
        ... )
        '.../sub-015_analysis_type-classification_analysis_name-SVM_CSm_v_CSpu_roi-bl_hemi-left_desc-results.csv'
        """
        parts = [str(self.subject)]

        for key, value in entities.items():
            if value is not None:
                parts.append(f"{key}-{value}")

        if desc is not None:
            parts.append(f"desc-{desc}")

        ext = str(ext).lstrip(".")

        return opj(
            self.save_dir,
            f"{'_'.join(parts)}.{ext}",
        )

    
    @classmethod
    def extract_betas_from_rois(
            self,
            *args,
            **kwargs
        ):

        return data.MaskAndFilterBetas(
            *args,
            **kwargs
        )


    def _extend_results_dict(
        self,
        results_dict,
        extracted,
        ddict,
        groups=None,
    ):
        """Extend a decoding result row with sample, model, CV, and QC metadata.

        Parameters
        ----------
        results_dict : dict
            Base result dictionary containing subject, ROI, decoding statistics,
            and ROI spatial QC information.

        extracted : object
            ROI extraction result. Expected to expose ``X`` and ``labels``.
            ``X`` must have shape ``(n_samples, n_features)``.

        ddict : dict
            Dictionary returned by the decoding/permutation procedure. Optional
            metadata such as the number of permutations is included when present.

        groups : array-like, optional
            Group identifiers aligned with the decoded samples, such as run
            identifiers.

        Returns
        -------
        dict
            The input dictionary extended with sample counts, class counts,
            cross-validation settings, estimator information, and group metadata.

        Notes
        -----
        Only scalar values suitable for tabular storage are added. Large objects
        such as feature matrices, fitted estimators, masks, permutation arrays,
        and full configuration dictionaries are deliberately excluded.
        """
        labels = np.asarray(extracted.labels)
        outer_cfg = self.cfg["decoding_settings"].get("outer_cv", {})
        inner_cfg = self.cfg["decoding_settings"].get("inner_cv", {})
        estimator_cfg = self.cfg["decoding_settings"].get("estimator", {})

        outer_args = outer_cfg.get("args", {})
        inner_args = inner_cfg.get("args", {})

        # Sample / feature information
        results_dict.update({
            "analysis": str(self.analysis_name),
            "analysis_type": str(self.analysis_type),
            "analysis_id": str(self.analysis_id),
            "n_samples": int(extracted.X.shape[0]),
            "n_features": int(extracted.X.shape[1]),
            "n_classes": int(np.unique(labels).size),
        })

        # Store included labels and class counts.
        classes, counts = np.unique(labels, return_counts=True)

        label_mapper = self.cfg["label_dict"]
        if label_mapper:
            # Reverse mapping: encoded label -> original class name.
            inverse_mapper = {
                value: key
                for key, value in label_mapper.items()
            }

            class_names = [
                inverse_mapper.get(label, str(label))
                for label in classes
            ]

            results_dict["class_names"] = ",".join(
                map(str, class_names)
            )

            for label, class_name, count in zip(
                classes,
                class_names,
                counts,
            ):
                results_dict[f"n_{class_name}"] = int(count)

        else:
            results_dict["class_names"] = ",".join(
                map(str, classes)
            )

            for label, count in zip(classes, counts):
                results_dict[f"n_class_{label}"] = int(count)

        # --------------------------------------------------------------
        # Permutation information
        if "n_perms" in ddict:
            results_dict["n_perms"] = int(ddict["n_perms"])

        if "n_run" in ddict:
            results_dict["n_run"] = int(ddict["n_run"])

        # --------------------------------------------------------------
        # Outer cross-validation
        outer_mode = outer_cfg.get("mode", "sklearn")

        results_dict.update({
            "outer_cv_mode": outer_mode,
            "outer_cv_name": outer_cfg.get("name"),
            "outer_fold_interval": outer_args.get("fold_interval"),
        })

        # --------------------------------------------------------------
        # Inner cross-validation
        results_dict.update({
            "inner_cv_name": inner_cfg.get("name"),
            "inner_n_splits": inner_args.get("n_splits"),
        })

        # --------------------------------------------------------------
        # Estimator / scoring
        results_dict.update({
            "estimator": estimator_cfg.get("name"),
            "scoring": self.cfg["decoding_settings"].get(
                "scoring",
                "balanced_accuracy",
            ),
        })

        # --------------------------------------------------------------
        # Group information
        if groups is not None and self.cfg.get("permute_within_groups", False):
            groups = np.asarray(groups)

            results_dict.update({
                "group_aware": True,
                "n_groups": int(np.unique(groups).size),
            })
        else:
            results_dict.update({
                "group_aware": False,
                "n_groups": 0,
            })

        # --------------------------------------------------------------
        # Convenient ROI QC flags
        metrics = extracted.roi_metrics

        n_voxels = metrics.get("n_voxels")
        coverage = metrics.get("volume_coverage_ratio")
        valid_fraction = metrics.get("valid_voxel_fraction")

        results_dict.update({
            "tiny_roi": (
                bool(n_voxels < 5)
                if n_voxels is not None
                else None
            ),
            "low_volume_coverage": (
                bool(coverage < 0.5)
                if coverage is not None and np.isfinite(coverage)
                else None
            ),
            "low_valid_voxel_fraction": (
                bool(valid_fraction < 0.8)
                if valid_fraction is not None
                and np.isfinite(valid_fraction)
                else None
            ),
        })

        return results_dict


    def prepare_rois(
            self,
            roi_labels,
            roi_name=None
        ):

        # prepare ROIs
        obj = data.PrepareROIs(
            subject=self.subject,
            roi_labels=roi_labels,
            roi_name=roi_name,
            project_dir=self.gen_settings["project_dir"],
            **self.roi_settings
        )

        # now a dict: {'left': Nift1Image, 'right': Nift1Image}
        roi_masks = obj.return_masks()

        return obj, roi_masks
