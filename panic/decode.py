# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import json
import shutil
import numpy as np
import pandas as pd

import logging
import tempfile, uuid
from tqdm import tqdm
from joblib import Parallel, delayed, dump, load

from panic import data, utils, factory
from panic.logger import get_logger, tqdm_joblib
from lazyfmri.utils import FindFiles, update_kwargs
from panic.searchlight import permutation_searchlight

logger = get_logger(__name__, level=logging.INFO, use_tqdm=True)
opj = os.path.join


def run_decoding_with_permutation(
    X,
    labels,
    folds,
    cfg,
    seed=0,
    tmpdir="~/.joblib_cache",
    groups=None,
    **kwargs
):
    """
    Run decoding with permutation testing on ROI-level data, with optional *fail-fast* early stopping.

    This function performs cross-validated decoding on a feature matrix ``X``
    to compute the **observed accuracy** and a **null distribution** of accuracies
    obtained by label permutation. The implementation mirrors
    :func:`utils._cv_mean_score` and is optimized for reproducibility and parallelization.

    Unlike symmetric early stopping (which can terminate for both early significance
    and non-significance), the **fail-fast** variant only halts permutations when it becomes
    mathematically impossible for the final empirical p-value to cross the threshold ``alpha``.
    This ensures *conservative inference* — never stopping early in favor of significance,
    only when continued computation cannot change a non-significant outcome.

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
    early_stop_alpha : float or None, optional
        Significance threshold (e.g., 0.05) for fail-fast stopping. If set,
        the procedure terminates once the smallest attainable empirical p-value
        (given the number of observed exceedances so far) exceeds ``alpha``.
        If ``None`` (default), all ``n_perms`` permutations are executed.
    early_stop_batch : int, optional
        Number of permutations to evaluate per batch before re-checking the
        early-stopping condition. Smaller batches allow more frequent checks but
        higher overhead. Default is 32.
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
        * ``"n_run"`` – Number of permutations actually executed
        * ``"early_stop_alpha"`` – Early-stop threshold used (or ``None`` if disabled)

    Workflow
    ---------
    1. Dumps the input feature matrix ``X`` to a temporary uncompressed
    ``joblib`` file for memory-mapped access.
    2. Computes the **observed** mean cross-validated score using
    :func:`utils._cv_mean_score`.
    3. Generates ``n_perms`` random seeds from a reproducible master RNG.
    4. Recomputes CV scores under label permutation, either serially or
    in parallel via :class:`joblib.Parallel`.
    5. Applies fail-fast early stopping if enabled:
    after each batch, compute:
    ::

        p_min = (count_exceed + 1) / (n_run + 1)

    and terminate if ``p_min > alpha`` and ``n_run >= ceil(1/alpha) - 1``.
    This guarantees that even in the most optimistic case,
    the final empirical p-value could not fall below ``alpha``.
    6. Aggregates permutation scores and computes empirical p-value:
    ::

        p = (sum(perm >= obs) + 1) / (n_run + 1)

    Statistical Outputs
    -------------------
    - **Observed** – Mean CV accuracy for the true (non-permuted) labels.
    - **Permuted** – Null distribution of accuracies from permuted labels.
    - **Mean permuted** – Average null accuracy (expected under the null).
    - **Δ (delta)** – Observed − null mean difference (effect size).
    - **p-value** – Empirical one-tailed significance level.
    - **n_run** – Number of permutations actually executed
    (≤ ``n_perms`` when early stopping is active).

    Early-Stopping Behavior
    -----------------------
    - Stops **only** when continued permutations cannot yield significance
    (i.e., ``p_min > alpha``).
    - Never stops early for apparent significance, ensuring conservative inference.
    - The batch size determines how often stopping is checked:
    smaller batches → more precise, larger batches → more efficient.
    - Parallelized batches maintain full reproducibility through explicit seeding.

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
            early_stop_alpha=0.05,
            early_stop_batch=32,
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

    tmpdir = os.path.expanduser(tmpdir)
    os.makedirs(tmpdir, exist_ok=True)
    tmpd = tempfile.mkdtemp(dir=tmpdir)
    
    # read settings for parallellization
    par_cfg = cfg.get("parallel", {})

    # compress=0 keeps it as a plain .npy-like file for fast mmap
    X_path = dump(
        X,
        os.path.join(tmpd, f"X_mm_{uuid.uuid4().hex}.joblib"),
        compress=0
    )[0]

    X_mm = load(X_path, mmap_mode="r")
    logger.info(f"X={X_mm.shape} (n_samples, n_features)")

    if groups is not None:
        logger.info(f"Groups = {groups}")

    labels = np.asarray(labels)
    groups = None if groups is None else np.asarray(groups)

    # observed
    if "save_dir" in kwargs:
        logger.info(f"Storing fold information in {kwargs['save_dir']}")

    observed_acc = utils._cv_mean_score(
        X_path, labels, folds, cfg,
        groups=groups,
        permute=False,
        **kwargs
    )

    logger.info("Observed accuracy after %d folds: %.4f", len(folds), observed_acc)

    # permutations in parallel with progress bar
    n_perms = cfg.get("n_permutations", 1000)
    n_jobs = par_cfg.get("n_jobs", 1)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, size=n_perms, dtype=np.uint32)

    # helper for one permutation draw
    def _one_perm(seed):
        return utils._cv_mean_score(
            X_path, labels, folds, cfg,
            groups=groups,
            permute=True,
            rng=np.random.default_rng(int(seed)),
            **kwargs
        )

    # get early stop alpha
    early_stop_alpha = cfg.get("early_stop_alpha", None)
    early_stop_batch = cfg.get("early_stop_batch", 32)
    logger.info("Starting permutation testing: n_perms=%d, n_jobs=%d", n_perms, n_jobs)
    if early_stop_alpha is None:


        if n_jobs == 1:
            permuted_acc = [
                _one_perm(s)
                for s in tqdm(
                    seeds,
                    total=n_perms,
                    disable=utils.tqdm_disabled()
                )
            ]
        else:
            with tqdm_joblib(
                tqdm(
                    total=n_perms,
                    disable=utils.tqdm_disabled()
                )
            ):
                permuted_acc = Parallel(
                    n_jobs=n_jobs,
                    backend=par_cfg.get("backend", "loky"),
                    prefer=par_cfg.get("prefer", "processes"),
                    batch_size=par_cfg.get("batch_size", 16),
                    verbose=par_cfg.get("verbose", 0)
                )([delayed(_one_perm)(s) for s in seeds])
        permuted_acc = np.asarray(permuted_acc, dtype=float)
        n_run = len(permuted_acc)

    # --- EARLY-STOP ON: batched loop with fail-fast rule ---
    else:
        n_batches = int(np.ceil(n_perms / early_stop_batch))
        logger.info(f"Fail-fast enabled with α={early_stop_alpha}, testing each {int(early_stop_batch)} permutations if significance can be reached (total #batches={n_batches})")
        J0 = int(np.ceil(1.0 / early_stop_alpha) - 1)  # minimum perms before we can possibly detect p<α
        count_exceed = 0
        permuted_list = []
        j = 0

        # iterator over seeds in batches
        def _batches(seq, k):
            for i in range(0, len(seq), k):
                yield seq[i:i+k]

        for batch in tqdm(
            _batches(seeds, early_stop_batch),
            total=n_batches,
            disable=utils.tqdm_disabled()
        ):
            if n_jobs == 1:
                vals = [_one_perm(s) for s in batch]
            else:
                # parallel within batch
                vals = Parallel(
                    n_jobs=n_jobs,
                    backend=par_cfg.get("backend", "loky"),
                    prefer=par_cfg.get("prefer", "processes"),
                    batch_size=par_cfg.get("batch_size", 16),
                    verbose=par_cfg.get("verbose", 0)
                )([delayed(_one_perm)(s) for s in batch])

            vals = np.asarray(vals, float)
            permuted_list.append(vals)
            # update state
            count_exceed += int(np.sum(vals >= observed_acc))
            j += len(vals)

            # fail-fast only: stop if p cannot become < alpha
            if j >= J0:
                p_min = (count_exceed + 1) / (j + 1)
                if p_min > early_stop_alpha:
                    break

            # also stop if we already hit all perms
            if j >= n_perms:
                break

        permuted_acc = np.concatenate(permuted_list, dtype=float) if permuted_list else np.empty(0, float)
        n_run = len(permuted_acc)

    # aggregate
    mean_permuted_acc = float(np.mean(permuted_acc)) if n_run else float("nan")
    delta = float(observed_acc - mean_permuted_acc)
    p_val = (np.sum(permuted_acc >= observed_acc) + 1) / (n_run + 1) if n_run else 1.0

    logger.info("Permutation complete. Null acc: %.4f | Observed acc: %.4f | Δ=%.4f | p=%.4f | n_run=%d",
                np.nanmean(permuted_acc) if n_run else float("nan"),
                observed_acc, delta, p_val, n_run)

    ddict = {
        "observed": float(observed_acc),
        "permuted": permuted_acc,
        "mean_permuted": float(mean_permuted_acc),
        "delta": float(delta),
        "p": float(p_val),
        "n_run": int(n_run),
        "n_perms": int(n_perms),
        "early_stop_alpha": None if early_stop_alpha is None else float(early_stop_alpha),
    }

    return ddict

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
        - ``dec_settings_no_feature_selection`` : dict  
          Copy of ``dec_settings`` with ``feature_selection=None`` for
          searchlight runs.
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
        - ``dec_settings_no_feature_selection`` is used to ensure that
          searchlight decoding does not perform ROI-level feature selection.
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
        self.cfg = utils.load_yaml(self.config_file)

        # append subject to save_dir
        self.gen_settings = self.cfg["general_settings"]
        self.dec_settings = self.cfg["decoding_settings"]

        # turn off feature selection within searchlight
        self.dec_settings_no_feature_selection = {**self.dec_settings, "feature_selection": None}
        self.save_dir = opj(self.gen_settings["save_dir"], self.subject)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
        logger.info(f"Decoding configuration:")
        logger.info(self.dec_settings)

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

        ddict = {
            "subject": self.subject,
            "beta_dir": opj(
                self.gen_settings["project_dir"],
                "derivatives",
                self.gen_settings["source"]
            ),
            "derivative": self.cfg["fitted_derivative"],
            "model": self.gen_settings["method"],
            "standardize": self.gen_settings["standardize"],
            "label_mapper": self.cfg["label_dict"]
        }

        for key, val in ddict.items():
            kwargs = update_kwargs(
                kwargs,
                key,
                val
            )

        data.PrepareBetas.__init__(self, **kwargs)

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
                folding scheme via :func:`define_folds`.
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

            extract = self.extract_betas_from_rois(
                betas,
                mask,
                trial_list=trial_list,
                label_mapper=label_mapper,
                output_file=output_file
            )

            # outer folds
            if self.dec_settings.get("permute_within_groups"):

                outer = factory.cv_from_config(self.dec_settings["outer_cv"])
                folds = [
                    (tr, te)
                    for tr, te in outer.split(
                        np.zeros_like(extract.labels),
                        extract.labels,
                        groups
                    )
                ]
            else:
                # fallback to your old every-third scheme (within-class)
                folds = self.define_folds(extract)   # or paste that logic here and delete the method
            
            # run classifier
            logger.info(f"Feature mapper: {label_mapper}")
            logger.info(f"Labels: {np.unique(extract.labels)} features (n={len(extract.labels)})")
            try:
                ddict = run_decoding_with_permutation(
                    extract.X,
                    extract.labels,
                    folds,
                    self.dec_settings,
                    groups=groups,
                    roi_linidx=extract.roi_linidx,
                    **kwargs
                )
            except Exception as e:
                raise RuntimeError(f"Decoding failed with message: {e}")
        else:
            # Searchlight (same same, but different)
            try:
                ddict = permutation_searchlight(
                    betas,
                    mask,
                    trial_list,
                    label_mapper,
                    self.dec_settings_no_feature_selection,
                    groups=groups,
                    **kwargs
                )
            except Exception as e:
                raise RuntimeError(f"Searchlight failed with message: {e}")

        return ddict
    
    def define_mask_inputs(self):

        # roi_dict options:
        #   1. dict -> FreeSurfer labels
        #   2. directory -> *.nii.gz files
        #   3. file -> assume mask

        if isinstance(self.cfg["roi_dict"], dict):
            run_dict = self.cfg["roi_dict"]
        elif isinstance(self.cfg["roi_dict"], str):
            if os.path.isdir(self.cfg["roi_dict"]):
                mask_files = FindFiles(
                    self.cfg["roi_dict"],
                    extension="nii.gz"
                ).files

                if isinstance(mask_files, str):
                    mask_files = [mask_files]

                if not isinstance(mask_files, list):
                    raise TypeError(f"We should have a list by now..")
                
                run_dict = {}
                n_digits = len(str(len(mask_files)))
                for ix, m in enumerate(mask_files):
                    lbl = f"mask_{str(ix+1).zfill(n_digits)}"
                    run_dict[lbl] = m
            else:
                run_dict = {
                    "mask_1": self.cfg["roi_dict"]
                }
        
        return run_dict

    def decode_masks(self, hemi_key="hemi", **kwargs):
        
        out_files = {}
        results = []
        null = []

        roi_dict = self.define_mask_inputs()
        for r_key, r_val in roi_dict.items():
            logger.info(f"Processing '{r_key}' (labels|file={r_val})")

            # prepare ROIs for beta-series extraction
            _, roi_masks = self.prepare_rois(r_val)

            # extract betas
            for h_key, h_val in roi_masks.items():
                logger.info(f"hemi-key={h_key} | roi-key={h_val[0]}..")

                model_src_dir = opj(
                    self.save_dir,
                    f"model-{self.gen_settings['method']}",
                    f"source-{self.gen_settings['source']}"
                )

                # extract beta-series
                fname = None
                roi_dir = None
                if self.save_imgs:
                    resampled_dir = opj(self.save_dir, "rois")
                    if not os.path.exists(resampled_dir):
                        os.makedirs(resampled_dir, exist_ok=True)

                    fname = opj(
                        resampled_dir,
                        f"{self.subject}_roi-{h_val[0]}_hemi-{h_key}_desc-valid_mask.nii.gz"
                    )
                
                roi_dir = opj(model_src_dir, f"roi-{h_val[0]}")
                ddict = self.decode_single_mask(
                    self.betas,
                    h_val[1],
                    trial_list=self.trial_list,
                    label_mapper=self.cfg["label_dict"],
                    output_file=fname,
                    save_dir=roi_dir,
                    **kwargs
                )

                # Store
                if not self.searchlight:
                    results_dict = {
                        "subject": str(self.bids_id),
                        "observed_acc": ddict["observed"],
                        "null_mean": ddict["mean_permuted"],
                        "delta": ddict["delta"],
                        "p_value": ddict["p"],
                        hemi_key: str(h_key),
                        "roi": str(h_val[0]),
                        "source": str(self.gen_settings["source"]),
                        "method": str(self.gen_settings["method"])
                    }
                    
                    if hemi_key == "hemi":
                        results_dict[hemi_key] = str(h_key)

                    results.append(results_dict)

                    # Store null perms in tidy form (one row per permutation)
                    null.extend(
                        {
                            "subject": str(self.bids_id),
                            "roi": str(h_val[0]),
                            "perm": i,
                            "acc": float(a),
                            "source": str(self.gen_settings["source"]),
                            "method": str(self.gen_settings["method"]),                        
                            **({hemi_key: str(h_key)} if hemi_key == "hemi" else {}),
                        }
                        for i, a in enumerate(ddict["permuted"])
                    )
                else:
                    out_files[h_val[0]] = ddict

        if not self.searchlight:
            # write null-distributions
            null_df = pd.DataFrame(null)
            null_file = self.generate_filename(
                desc="null_distribution",
                ext="csv"
            )

            logger.info(f"Writing null-distributions to '{null_file}'")
            null_df.to_csv(null_file, index=False)
            shutil.copymode(self.config_file, null_file)

            # write results to dataframe and save
            res_df = pd.DataFrame(results)
            res_file = self.generate_filename(
                desc="results",
                ext="csv"
            )

            logger.info(f"Writing results to: '{res_file}'")
            res_df.to_csv(res_file, index=False)
            shutil.copymode(self.config_file, res_file)

            # copy config file
            cfg_file = self.generate_filename(
                desc="config",
                ext="yml"
            )

            # Writing the data to a YAML file
            utils.dump_yaml(self.cfg, cfg_file)
            shutil.copymode(self.config_file, cfg_file)

            # return results
            return res_df

        else:
            # return results
            return out_files

    def generate_filename(self, desc=None, ext="csv"):
        base_name = f"{self.subject}_model-{self.gen_settings['method']}_source-{self.gen_settings['source']}"

        if desc is not None:
            base_name += f"_desc-{desc}"

        return opj(self.save_dir, f"{base_name}.{ext}")

    def define_folds(self, obj):

        labels = obj.labels
        trials = obj.trials
        n_trials = len(trials)

        # Get indices for CS- and CS+ trials
        trial_zero = np.where(labels == 0)[0]
        trial_one = np.where(labels == 1)[0]

        folds = []
        rotate_fold = self.dec_settings["fold_interval"]
        for offset in range(rotate_fold):
            # Every 3rd trial for test set (rotated)
            test_idx = np.concatenate(
                [
                    trial_zero[offset::rotate_fold],
                    trial_one[offset::rotate_fold]
                ]
            )

            train_idx = np.setdiff1d(np.arange(n_trials), test_idx)
            folds.append((train_idx, test_idx))

        return folds
    
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

    def prepare_rois(self, roi_labels, **kwargs):
        
        obj = data.PrepareROIs(
                subject=self.subject,
                roi_labels=roi_labels,
                project_dir=self.gen_settings["project_dir"],
                **kwargs
            )

        # now a dict: {'left': Nift1Image, 'right': Nift1Image}
        roi_masks = obj.return_masks()

        return obj, roi_masks