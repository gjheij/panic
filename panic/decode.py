# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import sys
import shutil
import numpy as np
import pandas as pd

import logging
import tempfile, uuid
from tqdm import tqdm
from joblib import Parallel, delayed, dump, load

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    VarianceThreshold
)

from panic import data, utils, factory
from panic.logger import get_logger, tqdm_joblib
from lazyfmri.utils import FindFiles, update_kwargs

logger = get_logger(__name__, level=logging.INFO, use_tqdm=True)
opj = os.path.join


def _pipeline(cfg, standardize=False):

    scaler = factory.scaler_from_config(cfg.get("scaler"))
    inner_cv = factory.cv_from_config(cfg["cv"])
    estimator = factory.estimator_from_config(cfg["estimator"])
    selector = factory.selector_from_config(
        cfg["feature_selection"],
        estimator_factory=factory.estimator_from_config,
        task="classification",
        random_state=0,
    )

    thr = float(cfg.get("variance_threshold", 1e-12))
    pipe = Pipeline([
        ("var", VarianceThreshold(threshold=thr)),
        ("scaler", scaler if standardize else "passthrough"),
        ("select", selector),
        ("clf", estimator),
    ])

    grid = factory.search_from_config(
        pipe,
        inner_cv,
        cfg["gridsearch"]
    )

    return grid

def _perm_mean_acc(
    X_path,
    labels,
    folds,
    cfg,
    perm_seed,
    **kwargs
    ):

    rng = np.random.default_rng(perm_seed)
    # memmap read-only to avoid large copies
    X_mm = load(X_path, mmap_mode="r")

    fold_accs = []
    for train_idx, test_idx in folds:
        y_train = labels[train_idx]
        y_test  = labels[test_idx]

        # independent permutations within each set
        y_train_perm = rng.permutation(y_train)
        y_test_perm  = rng.permutation(y_test)

        grid = _pipeline(cfg, **kwargs)

        grid.fit(X_mm[train_idx], y_train_perm)
        score = grid.score(X_mm[test_idx], y_test_perm)
        fold_accs.append(score)

    return float(np.mean(fold_accs))

def tqdm_disabled():
    return (not sys.stderr.isatty()) # or bool(os.environ.get("PYTEST_CURRENT_TEST"))

def run_decoding_with_permutation(
    X,
    labels,
    folds,
    cfg,
    n_perms=1000,
    n_jobs=1,
    seed=0,
    standardize=True,
    tmpdir="~/.joblib_cache"
):
    """
    Returns:
        observed_acc (float), perm_accs (np.ndarray, shape [n_perms])
    """
    os.makedirs(tmpdir, exist_ok=True)
    tmpd = tempfile.mkdtemp(dir=tmpdir)

    # compress=0 keeps it as a plain .npy-like file for fast mmap
    X_path = dump(
        X,
        os.path.join(tmpd, f"X_mm_{uuid.uuid4().hex}.joblib"),
        compress=0
    )[0]

    # ---- observed accuracy (no parallel here) ----
    X_mm = load(X_path, mmap_mode="r")
    logger.info(f"X={X_mm.shape} (n_features, n_samples)")

    fold_accs = []
    for train_idx, test_idx in folds:
        grid = _pipeline(cfg, standardize=standardize)
        grid.fit(X_mm[train_idx], labels[train_idx])
        y_true = labels[test_idx]
        score = grid.score(X_mm[test_idx], y_true)
        fold_accs.append(score)

    observed_acc = float(np.mean(fold_accs))
    logger.info("Observed accuracy after %d folds: %.4f", len(fold_accs), observed_acc)

    # ---- permutations in parallel with progress bar ----
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, size=n_perms, dtype=np.uint32)

    logger.info("Starting permutation testing: n_perms=%d, n_jobs=%d", n_perms, n_jobs)

    if n_jobs == 1:
        perm_accs = [
            _perm_mean_acc(
                X_path,
                labels,
                folds,
                cfg,
                int(s),
                standardize=standardize
            )
            for s in tqdm(
                seeds,
                total=n_perms,
                desc="[INFO] panic.decode - Permutation testing",
                disable=tqdm_disabled()
            )
        ]
    else:
        # Parallel path => per-permutation updates
        with tqdm_joblib(
            tqdm(
                total=n_perms,
                desc="[INFO] panic.decode - Permutation testing",
                disable=tqdm_disabled()
            )
            ):
            perm_accs = Parallel(
                n_jobs=n_jobs,
                backend="loky",
                prefer="processes",
                batch_size=1,
                verbose=0
            )(
                delayed(_perm_mean_acc)(
                    X_path,
                    labels,
                    folds,
                    cfg,
                    int(s),
                    standardize=standardize
                )
                for s in seeds
            )

    perm_accs = np.asarray(perm_accs, dtype=float)
    perm_acc = np.mean(perm_accs)

    delta = observed_acc - perm_acc
    logger.info("Permutation complete. Null acc: %.4f | Observed acc: %.4f | Δ=%.4f", perm_acc, observed_acc, delta)
    
    return observed_acc, perm_accs, delta

class ClassifySubject(data.PrepareBetas):

    def __init__(
        self,
        subject,
        config_file,
        save_imgs=False,
        **kwargs
        ):
        
        # init
        self.subject = subject
        self.config_file = config_file
        self.bids_id = str(self.subject.split("-")[-1])
        self.save_imgs = save_imgs

        # load settings
        logger.info(f"Running decoding for {self.subject}")
        logger.info(f"Loading settings from {self.config_file}")
        self.cfg = utils.load_yaml(self.config_file)

        # append subject to save_dir
        self.gen_settings = self.cfg["general_settings"]
        self.dec_settings = self.cfg["decoding_settings"]

        self.save_dir = opj(self.gen_settings["save_dir"], self.subject)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
        logger.info(f"Decoding configuration:")
        logger.info(self.dec_settings)

    def _fit(self, **kwargs):
        try:
            # load betas
            self._init_betas()
        
            # decode hemispheres
            kwargs = update_kwargs(
                kwargs,
                "standardize",
                self.do_standardization
            )

            self.results = self.decode_masks(**kwargs)
            logger.info(f"Decoding {self.subject} complete")
        except Exception as e:
            logger.exception(f"Decoding failed with errors: {e}")
            raise

    def _init_betas(self, **kwargs):
        ddict = {
            "subject": self.subject,
            "beta_dir": opj(
                self.gen_settings["project_dir"],
                "derivatives",
                self.gen_settings["source"]
            ),
            "derivative": self.cfg["fitted_derivative"],
            "model": self.gen_settings["method"],
            "standardize": self.gen_settings["standardize"]
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
        **kwargs
        ):

        extract = self.extract_betas_from_rois(
            betas,
            mask,
            trial_list=trial_list,
            label_mapper=label_mapper,
            output_file=output_file
        )

        # define folds
        folds = self.define_folds(extract)

        # run classifier
        logger.info(f"Feature mapper: {label_mapper}")
        logger.info(f"Labels: {np.unique(extract.labels)} features (n={len(extract.labels)})")

        obs_acc, null_accs, delta = run_decoding_with_permutation(
            extract.X,
            extract.labels,
            folds,
            self.dec_settings,
            **kwargs
        )
        
        # Within-subject p-value
        p_val = (np.sum(null_accs >= obs_acc) + 1) / (len(null_accs) + 1)

        return obs_acc, null_accs, delta, p_val
    
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

                # extract beta-series
                fname = None
                if self.save_imgs:
                    resampled_dir = opj(self.save_dir, "rois")
                    if not os.path.exists(resampled_dir):
                        os.makedirs(resampled_dir, exist_ok=True)

                    fname = opj(
                        resampled_dir,
                        f"{self.subject}_roi-{h_val[0]}_hemi-{h_key}_desc-valid_mask.nii.gz"
                    )

                obs_acc, null_accs, delta, p_val = self.decode_single_mask(
                    self.betas,
                    h_val[1],
                    trial_list=self.trial_list,
                    label_mapper=self.cfg["label_dict"],
                    n_perms=self.dec_settings["n_permutations"],
                    n_jobs=self.gen_settings["n_jobs"],
                    output_file=fname,
                    **kwargs
                )

                # Store
                results_dict = {
                    "subject": str(self.bids_id),
                    "observed_acc": float(obs_acc),
                    "null_mean": float(null_accs.mean()),
                    "delta": float(delta),
                    "p_value": float(p_val),
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
                    for i, a in enumerate(null_accs)
                )

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
        cs_minus_idx = np.where(labels == 0)[0]
        cs_plus_idx = np.where(labels == 1)[0]

        folds = []
        rotate_fold = self.dec_settings["fold_interval"]
        for offset in range(rotate_fold):
            # Every 3rd trial for test set (rotated)
            test_idx = np.concatenate(
                [
                    cs_minus_idx[offset::rotate_fold],
                    cs_plus_idx[offset::rotate_fold]
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