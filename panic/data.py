# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import json
import numpy as np
from nilearn import (
    image,
    maskers
)
import nibabel as nib
from lazyfmri import utils
from panic.logger import get_logger

opj = os.path.join

logger = get_logger(__name__)
                    
class PrepareROIs():

    def __init__(
        self,
        subject=None,
        project_dir="/mnt/d/fMRI/HRA",
        roi_dir=None,
        roi_name="hippoAmygLabels.mgz",
        roi_labels=None,
        roi_base="hippo-amygdala",
        src="freesurfer",
        extension="mgz",
        ):
        
        self.subject = subject
        self.project_dir = project_dir
        self.roi_dir = roi_dir
        self.roi_base = roi_base
        self.roi_name = roi_name
        self.roi_labels = roi_labels
        self.src = src
        self.extension = extension
        
        # derive mask dir from components or straight directory
        self.mask_dir = self.derive_mask_dir()

        # load masks
        if isinstance(self.roi_labels, list):
            self.roi_masks = self._from_labels(self.mask_dir)
        elif isinstance(self.roi_labels, str):
            if os.path.isdir(self.roi_labels):
                self.roi_masks = self._from_directory(self.roi_labels)
            elif os.path.isfile(self.roi_labels):
                self.roi_masks = self._from_file(self.roi_labels)
        else:
            raise TypeError(f"roi_labels must be a list of FreeSurfer-compatible labels, a directory with *.nii.gz files, or an actual .nii.gz file, not '{self.roi_labels}'")

    def derive_mask_dir(self):

        if self.roi_dir is None:
            return opj(
                self.project_dir,
                "derivatives",
                self.src,
                self.subject,
                self.roi_base
            )
        else:
            return self.roi_dir

    def return_masks(self):
        return self.roi_masks

    def _from_directory(self, directory, fill=0.0, out_dtype=np.uint8):
        
        ddict = {}
        all_imgs = utils.FindFiles(
            directory,
            extension=".nii.gz"
        ).files

        if isinstance(all_imgs, str):
            all_imgs = [all_imgs]

        if not isinstance(all_imgs, list):
            raise TypeError(f"We should have a list of images by now, not {all_imgs} of type {type(all_imgs)}")
        
        ddict = {}
        for i in all_imgs:
            logger.info(f" {i}")
            img = nib.load(i)
            data = self._sanitize(img, fill=fill)
            out_img = nib.Nifti1Image(
                data.astype(out_dtype, copy=False),
                img.affine
            )
            
            lbl, roi_name = self._fetch_key(i)

            ddict[lbl] = [roi_name, out_img]

        return ddict

    def _from_atlas(self, img, fill=0.0):
        raise NotImplementedError(f"To-be-implemented: extract from actual atlas file; for now, use the label options")

    def _from_file(self, i, fill=0.0, out_dtype=np.uint8):

        img = nib.load(i)
        data = self._sanitize(img, fill=fill)
        out_img = nib.Nifti1Image(
            data.astype(out_dtype, copy=False),
            img.affine
        )
        lbl, roi_name = self._fetch_key(i)

        return {
            lbl: [roi_name, out_img]
        }
        
    def _from_labels(self, mask_dir):
        
        logger.info(f"Loading masks from '{mask_dir}'")
        masks = utils.get_file_from_substring(
            self.roi_name,
            utils.FindFiles(
                mask_dir,
                extension=self.extension,
            ).files
        )
        
        if not isinstance(masks, list):
            masks = [masks]

        for ix, m in enumerate(masks):
            logger.info(f" #{ix+1}: {m}")

        return_masks = {}
        for i in masks:

            # Create a binary mask from labeled mgz
            tmp_mask = self.select_labels_from_mgh(
                input_file=i,
                labels=self.roi_labels,
                mode="nifti",
                binary=True,
            )

            lbl, roi_n = self._fetch_key(i, roi_name=self.roi_name)
            return_masks[lbl] = [roi_n, tmp_mask]

        return return_masks

    @classmethod
    def _fetch_key(self, i, roi_name=None):
        path_base = os.path.basename(i).lower()
        if path_base.startswith("lh.") or "hemi-l" in path_base:
            lbl = "left"
        elif os.path.basename(i).startswith("rh.") or "hemi-r" in path_base:
            lbl = "right"
        else:
            lbl = "uni"

        if roi_name is None:
            try:
                bids_comps = utils.split_bids_components(i)
            except Exception:
                logger.warning(f"{i} is not in BIDS-format. Cannot extract 'roi-<roi_name>' key, so defaulting to 'roi' as roi_name. This can result in clashing names!")
                roi_name = "roi"

            if roi_name is None:
                if "roi" in bids_comps:
                    roi_name = bids_comps["roi"]
                else:
                    logger.warning(f"{i} does not have 'roi-' key, defaulting to 'roi' as roi name. This can result in clashing names!")

        return lbl, roi_name.split(".")[0]

    @classmethod
    def _sanitize(self, img, fill=0.0):
        
        if isinstance(img, str):
            img = nib.load(img)

        # replace NaNs/±∞ BEFORE integer casting
        data = img.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            data = np.nan_to_num(
                data,
                nan=fill,
                posinf=fill,
                neginf=fill
            )

        return data

    @classmethod
    def select_labels_from_mgh(
        self,
        input_file,
        labels,
        mode="mgz",         
        binary=False,
        fill=0.0
    ):
        """
        Extract specified label values from a FreeSurfer .mgz/.mgh image and optionally binarize.

        - Sanitizes input (NaN/±∞ -> `fill`) to avoid resampling warnings.
        - Uses nearest-neighbor-friendly integer dtypes (uint8 for binary, int32 for labels).
        """

        img = nib.load(input_file)

        # 1) Sanitize: replace NaNs/±∞ BEFORE integer casting
        data = self._sanitize(img, fill=fill)

        # 2) Work in integer label space
        data = np.rint(data).astype(np.int32, copy=False)
        labels = np.asarray(sorted(set(int(l) for l in labels)), dtype=np.int32)

        if binary:
            masked = np.isin(data, labels).astype(np.uint8, copy=False)
            out_dtype = np.uint8
        else:
            masked = np.zeros_like(data, dtype=np.int32)
            for lab in labels:
                masked[data == lab] = lab
            out_dtype = np.int32

        # 3) Create a fresh image (don’t reuse header with a changed dtype)
        m = mode.lower()
        if m == "mgz":
            out_img = nib.MGHImage(masked.astype(out_dtype, copy=False), img.affine)
        elif m in ("nifti", "nii", "nii.gz"):
            out_img = nib.Nifti1Image(masked.astype(out_dtype, copy=False), img.affine)
            out_img.set_data_dtype(out_dtype)
        else:
            raise ValueError("mode must be 'mgz' or 'nifti'")

        return out_img
    
class PrepareBetas():

    def __init__(
        self,
        beta_file: str=None,
        trial_list: list=None,
        **kwargs
        ):

        if isinstance(beta_file, str):
            logger.info(f"Beta-file: '{beta_file}'")
            self.betas, self.do_standardization = self.sanitize_img(beta_file, **kwargs)
            
            if not isinstance(trial_list, (list, np.ndarray)):
                logger.exception("Please specify a list representing the trials")

            self.trial_list = trial_list
        else:
            self.betas, self.trial_list, self.do_standardization = self.load_and_merge_betas(
                **kwargs
            )

    def return_trials(self):
        return self.trial_list

    @classmethod
    def sanitize_img(self, img, fill=0.0, clip=1e6, standardize=None, **kwargs):
        """
        Replace NaN/±inf with `fill` and optionally clip extreme values.
        Works for 3D or 4D images; preserves affine/header.
        """

        img = image.load_img(img)
        data = img.get_fdata(dtype=np.float32)
        # replace NaN/±inf
        data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)
        # optional safety clip to avoid huge outliers
        if clip is not None:
            data = np.clip(data, -clip, clip)

        do_standardization = True
        if standardize == "zscore":
            logger.info("Z-scoring betas before decoding; removing 'StandardScalar' from pipeline")

            mu = data.mean(axis=-1, keepdims=True)
            sigma = data.std(axis=-1, ddof=1, keepdims=True)
            zdata = (data - mu) / np.where(sigma == 0, 1, sigma)

            data = np.nan_to_num(zdata, copy=False)
            do_standardization = False
            
        return image.new_img_like(img, data, copy_header=True), do_standardization
    
    @classmethod
    def load_and_merge_betas(
        self,
        subject,
        beta_dir="/mnt/d/fMRI/HRA/derivatives/stglm",
        save_imgs=False,
        output_dir=None,
        derivative=False,
        model="lsa",
        **kwargs
        ):

        # stglm | halfpipe | glmsingle
        src = os.path.basename(beta_dir)
        logger.info(f"Loading betas from: '{opj(beta_dir, subject)}'")

        model_mapper = {
            "lss": "typeb",
            "lsa": "typed"
        }
            
        # stglm | glmsingle need to be concatenated; HALFpipe comes concatenated
        if src in ["stglm", "glmsingle", "bach"]:
            search = f"desc-{model}"

            if src == "glmsingle":
                search = f"model-{model_mapper[model]}_beta-"
            elif src == "bach":
                search = "beta_"

            m_files = utils.FindFiles(
                opj(beta_dir, subject),
                extension=".nii.gz",
                filters=search
            ).files

            if derivative:
                logger.info("Directory contains trialwise estimates of the temporal derivative. Selecting every other beta file.")

                m_files = m_files[0::2]

            logger.info(f"Found {len(m_files)} files for {model}-model")
            
            logger.info("Loading and concatenating these files into single 4D-object")
            niimgs = []
            for f in m_files:
                niimgs.append(nib.load(f))

            beta_imgs = image.concat_imgs(niimgs)
        else:

            # find statmap files
            beta_files = utils.FindFiles(
                opj(beta_dir, subject),
                extension="stat-effect_statmap.nii.gz"
            )

            # beta files should have 'model-{lss|lsa}'-tag
            model_file = utils.get_file_from_substring(
                [f"model-{model}"],
                beta_files
            )
            
            logger.info(f"{model}: {model_file}")
            beta_imgs = nib.load(model_file)

        # glmsingle has separate trial_list.txt file in directory; others have json sidecar
        if src in ["glmsingle", "bach"]:
            txt_files = utils.FindFiles(
                opj(beta_dir, subject),
                extension=".txt"
            ).files

            trial_file = utils.get_file_from_substring(
                "trial_list",
                txt_files
            )

            trials = np.loadtxt(trial_file, dtype=str)
        else:

            json_files = utils.FindFiles(
                opj(beta_dir, subject),
                extension=".json"
            ).files

            # stglm is run-wise
            if src == "stglm":

                trials = []
                for j in json_files:
                    with open(j, "r") as j_file:
                        metadata = json.load(j_file)

                    trials += metadata["TrialList"]
            else:
                # HALFpipe output has single json for LSS/LSA models; should have model-tag in directory name
                json_model = utils.get_file_from_substring(
                    f"model-{model}",
                    json_files
                )

                with open(json_model, "r") as j_file:
                    metadata = json.load(j_file)
                
                trials = metadata["TrialList"]
        
        logger.info("Sanitizing beta images (e.g., remove NaN/inf and set float)")
        beta_imgs, is_standardized = self.sanitize_img(beta_imgs, **kwargs)

        if save_imgs:
            if output_dir is None:
                output_dir = opj(beta_dir, subject)

            fname = opj(
                output_dir,
                f"{subject}_model-{model}_source-{src}_desc-merged_betas.nii.gz"
            )

            logger.info(f"Saving merged beta image as '{fname}'")
            beta_imgs.to_filename(fname)

        return beta_imgs, trials, is_standardized

class MaskAndFilterBetas():

    def __init__(
        self,
        betas,
        mask,
        trial_list=None,
        label_mapper=None,
        **kwargs
    ):
        
        self.betas = betas
        self.mask = mask
        self.trial_list = trial_list
        self.label_mapper = label_mapper

        # extract betas from mask
        self.betas_in_mask = self._masker(**kwargs)

        # filter trials and set labels
        self.X, self.trials, self.labels = self.filter_trials()


    def filter_trials(self):

        filtered_trials = []
        labels = []
        trial_indices = []

        for i, t in enumerate(self.trial_list):
            for key, val in self.label_mapper.items():
                if t.startswith(key):
                    filtered_trials.append(key)
                    labels.append(val)
                    trial_indices.append(i)

        labels = np.array(labels)
        n_trials = len(labels)

        parts = [
            f"{k}: {(labels == v).sum()}"
            for k, v in self.label_mapper.items()
        ]

        logger.info(f"Total included trials: {n_trials} ({', '.join(parts)})")

        # Filter beta images and corresponding trial names
        X_filtered = self.betas_in_mask[trial_indices]
        filtered_trial_list = [self.trial_list[i] for i in trial_indices]

        logger.info(f"Filtered beta images: {X_filtered.shape[0]}")
        return X_filtered, filtered_trial_list, labels


    def return_betas(self):
        return self.betas_in_mask
    
    def _masker(
        self,
        interpolation="nearest",
        output_file=None,
        var_thr=0.0,
        masker_kws={},
        fit_kws={}
        ):

        # resample mask to betas
        betas_first_vol = image.index_img(self.betas, 0)
        logger.info(f"Resampling mask {self.mask.shape} to affine of beta-image {betas_first_vol.shape}")

        self.mask_resampled_to_betas = image.resample_to_img(
            self.mask,
            betas_first_vol,
            interpolation=interpolation,
            force_resample=True,
            copy_header=True
        )

        # find intersection with betas
        logger.info(f"Identifying valid voxels (var>{var_thr})")
        B = self.betas.get_fdata(dtype=np.float32)

        eps = max(float(var_thr), np.finfo(np.float32).eps)
        varying = B.var(axis=3) > eps
        support = np.isfinite(B).all(axis=3) & varying

        support_img = image.new_img_like(
            self.mask_resampled_to_betas,
            support.astype(np.uint8),
            copy_header=True
        )


        # 3) intersect: only keep ROI voxels inside beta support
        roi_in_fov = image.math_img(
            "roi & sup",
            roi=self.mask_resampled_to_betas,
            sup=support_img
        )

        vox_before = int((self.mask_resampled_to_betas.get_fdata() > 0.5).sum())
        vox_after = int(roi_in_fov.get_fdata().sum())
        logger.info(f"{vox_after}/{vox_before} voxels inside beta FOV")

        if output_file is not None:
            logger.info(f"Saving resampled mask: {output_file}")
            nib.save(roi_in_fov, output_file)

        logger.info(f"Extract betas with resampled/validated mask")

        masker = maskers.NiftiMasker(
            mask_img=self.mask_resampled_to_betas,
            dtype="float32",
            **masker_kws
        )

        return masker.fit_transform(
            self.betas,
            **fit_kws
        )