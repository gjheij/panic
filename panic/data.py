# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import re
import os
import numpy as np
from nilearn import (
    image,
    maskers
)
import nibabel as nib
from lazyfmri import utils
from collections import Counter
from panic.logger import get_logger
from panic.errors import EmptyMaskError

opj = os.path.join

logger = get_logger(__name__)


class PrepareROIs:
    """
    Discover, load, and sanitize ROI masks for a subject from directories,
    single files, or labeled FreeSurfer volumes.

    The class resolves a canonical *mask directory* (via :meth:`derive_mask_dir`)
    and then loads ROI masks depending on the type of ``roi_labels``:

    - **list[int]** → extract those integer labels from a FreeSurfer ``.mgz/.mgh``
      file matching ``roi_src`` using :meth:`_from_labels`
      (internally calls :meth:`select_labels_from_mgh`).
    - **str (directory path)** → bulk-load all ``.nii.gz`` masks in the folder
      via :meth:`_from_directory`.
    - **str (file path)** → load a single ``.nii.gz`` mask via :meth:`_from_file`.

    On success, ``self.roi_masks`` holds a dictionary mapping hemisphere/side labels
    (e.g., ``"left"``, ``"right"``, ``"uni"``) to ``[roi_name, NIfTI image]`` pairs.

    :param str | None subject:
        Subject identifier (e.g., ``"sub-01"``). Used when deriving the mask directory.
    :param str project_dir:
        Project root directory. Defaults to ``"/mnt/d/fMRI/HRA"``.
    :param str | None roi_dir:
        Custom directory where ROI/mask files live. If provided, used as-is.
        Otherwise, the directory is constructed as
        ``<project_dir>/derivatives/<src>/<subject>/<roi_base>``.
    :param str roi_src:
        Filename substring to select the source labeled volume (e.g., ``"aseg.mgz"`` or
        a custom FreeSurfer label file) when ``roi_labels`` is a **list**.
        Default: ``"hippoAmygLabels.mgz"``.
    :param list[int] | str | None roi_labels:
        - List of FreeSurfer label integers to extract (uses :meth:`_from_labels`)
        - Directory path with ``.nii.gz`` masks (uses :meth:`_from_directory`)
        - Single ``.nii.gz`` file path (uses :meth:`_from_file`)
    :param str roi_base:
        Base folder name under the derived mask directory (default: ``"hippo-amygdala"``).
    :param str src:
        Data source or derivative family (e.g., ``"freesurfer"``). Used in path derivation.
    :param str extension:
        File extension used when searching for source files in :meth:`_from_labels`
        (default: ``"mgz"``).

    **Attributes**
        - ``mask_dir`` : str  
          Resolved directory used to search for ROIs.
        - ``roi_masks`` : dict  
          Mapping ``{hemisphere_label: [roi_name, nibabel.Nifti1Image]}``.

    **Raises**
        - :class:`TypeError` if ``roi_labels`` is not a list of ints, a directory
          with masks, or a single mask filepath.

    **Typical usage**
        .. code-block:: python

            # 1) From FreeSurfer labels in an MGZ file
            rois = PrepareROIs(
                subject="sub-01",
                project_dir="/proj",
                roi_base="masks",
                src="freesurfer",
                roi_src="aseg.mgz",
                roi_labels=[17, 53]  # hippocampus L/R
            )
            roi_dict = rois.return_masks()

            # 2) From a directory of ready-made NIfTI masks
            rois = PrepareROIs(
                roi_dir="/proj/derivatives/masks/sub-01",
                roi_labels="/proj/derivatives/masks/sub-01"  # directory path
            )

            # 3) From a single NIfTI mask file
            rois = PrepareROIs(
                roi_labels="/proj/derivatives/masks/sub-01_roi-amygdala.nii.gz"
            )

    .. note::
       - Hemisphere labels are inferred from filenames (``lh.``, ``rh.``, ``hemi-L/R``)
         by :meth:`_fetch_key`. Unknown patterns default to ``"uni"``.
       - When extracting labels from FreeSurfer volumes, voxel data are sanitized
         and cast to integer-friendly dtypes to support nearest-neighbor resampling.
       - All loaders sanitize NaN/±∞ values before casting (see :meth:`_sanitize`).
    """

    def __init__(
            self,
            subject=None,
            project_dir="/mnt/d/fMRI/HRA",
            roi_dir=None,
            roi_src="hippoAmygLabels.mgz",
            roi_name=None,
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
        self.roi_src = roi_src
        self.roi_labels = roi_labels
        self.src = src
        self.extension = extension
        
        # load masks
        if isinstance(self.roi_labels, list):
            # derive mask dir from components or straight directory
            self.mask_dir = self.derive_mask_dir()

            assert os.path.exists(self.mask_dir), FileNotFoundError(f"Mask directory '{self.mask_dir}' does not exist")

            self.roi_masks = self._from_labels(self.mask_dir)
        elif isinstance(self.roi_labels, str):
            if os.path.isdir(self.roi_labels):
                self.roi_masks = self._from_directory(self.roi_labels)
            elif os.path.isfile(self.roi_labels):
                self.roi_masks = self._from_file(self.roi_labels)
        else:
            raise TypeError(f"roi_labels must be a list of FreeSurfer-compatible labels, a directory with *.nii.gz files, or an actual .nii.gz file, not '{self.roi_labels}'")


    def derive_mask_dir(self):
        """
        Determine the directory path for ROI or mask files.

        This helper constructs the canonical mask directory path for the
        current subject and project context. If a custom ROI directory
        has been explicitly defined (``self.roi_dir``), it is returned
        directly; otherwise, the path is derived from the project and
        subject hierarchy.

        :returns:
            The absolute path to the directory containing ROI or mask files.
        :rtype:
            str

        **Path Resolution Logic**
            - If ``self.roi_dir`` is not ``None``:
            Returns the user-specified directory directly.
            - Otherwise:
            Constructs the default path as:

            ``<project_dir>/derivatives/<src>/<subject>/<roi_base>``

            where:
            * ``self.project_dir`` – root of the decoding project  
            * ``self.src`` – data source or analysis type (e.g., "fmriprep")  
            * ``self.subject`` – subject identifier (e.g., "sub-01")  
            * ``self.roi_base`` – base folder for ROI definitions (e.g., "masks")

        **Example**
            .. code-block:: python

                # Case 1: using default project structure
                decoder.roi_dir = None
                path = decoder.derive_mask_dir()
                print(path)
                # /project/derivatives/fmriprep/sub-01/masks

                # Case 2: using a custom ROI directory
                decoder.roi_dir = "/custom/rois/sub-01"
                print(decoder.derive_mask_dir())
                # /custom/rois/sub-01

        .. note::
        - Ensures consistent directory structure for ROI and mask loading.
        - Typically used internally before file operations like
            :func:`decode_single_mask`.
        - Relies on class attributes: ``self.project_dir``, ``self.src``,
            ``self.subject``, and ``self.roi_base``.
        """

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
        """
        Load and sanitize all NIfTI images from a directory, returning them as a dictionary.

        This internal helper scans a directory for ``.nii.gz`` files, loads each image,
        applies a sanitation step (e.g., NaN/Inf replacement, type casting), and returns
        a dictionary mapping label identifiers to tuples of
        ``[roi_name, nibabel.Nifti1Image]``.

        :param str directory:
            Path to the directory containing NIfTI mask or ROI files.
        :param float fill:
            Replacement value for invalid voxels (e.g., NaNs or infs). Default is ``0.0``.
        :param numpy.dtype out_dtype:
            Output data type for the resulting images (default: ``numpy.uint8``).

        :returns:
            Dictionary where each key is a label identifier (parsed from filename)
            and each value is a list ``[roi_name, nibabel.Nifti1Image]`` containing the
            ROI name and the sanitized image object.
        :rtype:
            dict[str, list[Union[str, nibabel.Nifti1Image]]]

        **Processing Steps**
            1. Find all files in ``directory`` with the extension ``.nii.gz`` using
            :func:`utils.FindFiles` (0 recursion, looks only in specified path).
            2. Load each image via :func:`nibabel.load`.
            3. Sanitize image data via :func:`self._sanitize`, replacing invalid
            voxel values with ``fill``.
            4. Convert voxel intensities to ``out_dtype`` for memory efficiency.
            5. Parse a label and ROI name from each filename using :func:`self._fetch_key`.
            6. Store the result in the output dictionary keyed by label.

        **Example**
            .. code-block:: python

                rois = decoder._from_directory(
                    directory="/data/derivatives/masks/sub-01",
                    fill=0.0,
                    out_dtype=np.uint8
                )

                for label, (roi_name, img) in rois.items():
                    print(f"{label}: {roi_name}, shape={img.shape}")

        **Example Output**
            ::
                ROI_01: amygdala_L, shape=(64, 64, 36)
                ROI_02: hippocampus_R, shape=(64, 64, 36)

        .. note::
        - Relies on :func:`utils.FindFiles` to locate NIfTI files.
        - The helper :func:`self._sanitize` ensures all loaded images have valid
            voxel values and uniform dtype.
        - The method expects filenames to encode label and ROI name in a format
            recognizable by :func:`self._fetch_key`.
        - Commonly used to bulk-load ROI masks before decoding or visualization.
        """
            
        ddict = {}
        all_imgs = utils.FindFiles(
            directory,
            maxdepth=0,
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
        """
        Load and sanitize a single NIfTI ROI or mask file.

        This internal helper loads one ``.nii.gz`` file, cleans its voxel data,
        converts it to a desired numeric type, and returns it in a dictionary
        structure consistent with :func:`_from_directory`.

        :param str i:
            Path to a NIfTI file (``.nii.gz``) to load.
        :param float fill:
            Value used to replace invalid voxels (e.g., NaNs or infinities). Default is ``0.0``.
        :param numpy.dtype out_dtype:
            Output data type for the image data array (default: ``numpy.uint8``).

        :returns:
            A dictionary with a single key–value pair:
            
            ``{label: [roi_name, nibabel.Nifti1Image]}``
            
            where:
            * ``label`` – identifier parsed from the filename  
            * ``roi_name`` – human-readable ROI name  
            * ``nibabel.Nifti1Image`` – sanitized and type-cast NIfTI image
        :rtype:
            dict[str, list[Union[str, nibabel.Nifti1Image]]]

        **Processing Steps**
            1. Load the NIfTI file using :func:`nibabel.load`.
            2. Sanitize voxel data via :func:`self._sanitize`, replacing invalid values
            with the specified ``fill`` value.
            3. Cast the voxel array to ``out_dtype`` for compact storage.
            4. Parse a label and ROI name from the filename using :func:`self._fetch_key`.
            5. Return a dictionary entry consistent with the format used by
            :func:`_from_directory`.

        **Example**
            .. code-block:: python

                entry = decoder._from_file(
                    "/data/derivatives/masks/sub-01_roi-amygdala.nii.gz",
                    fill=0.0,
                    out_dtype=np.uint8
                )

                print(entry)
                # {'ROI_01': ['amygdala', <nibabel.nifti1.Nifti1Image object>]}

        .. note::
        - This method is used internally by :func:`_from_directory` when loading
            multiple mask files.
        - Ensures consistent output formatting and voxel data sanitization.
        - Filenames must follow a naming pattern recognizable by
            :func:`self._fetch_key` to extract label and ROI name correctly.
        """

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
        """
        Load and extract ROI masks from labeled volumetric files (e.g., FreeSurfer .mgz).

        This internal helper searches for labeled segmentation files in the specified
        directory, extracts one or more ROIs based on label indices, converts them
        into binary NIfTI masks, and returns a dictionary mapping label identifiers
        to ``[roi_name, NIfTI image]`` pairs.

        :param str mask_dir:
            Path to the directory containing labeled mask or segmentation files
            (e.g., FreeSurfer ``aseg.mgz`` or parcellation volumes).

        :returns:
            Dictionary mapping label identifiers to a list containing:
            
            * ``roi_src`` – the human-readable ROI name
            * ``nibabel.Nifti1Image`` – binary ROI mask image in NIfTI format
        :rtype:
            dict[str, list[Union[str, nibabel.Nifti1Image]]]

        **Workflow**
            1. Search ``mask_dir`` for all files matching ``self.extension`` using
            :func:`utils.FindFiles`.
            2. Filter results to include only files containing the target substring
            ``self.roi_src`` via :func:`utils.get_file_from_substring`.
            3. For each matching file:
            - Log the filename and index.
            - Call :func:`self.select_labels_from_mgh` to extract the specified
                labels from the input segmentation file (``.mgz`` or similar).
            - Convert extracted data into a binary NIfTI mask (``mode='nifti'``,
                ``binary=True``).
            - Parse a label and ROI name via :func:`self._fetch_key`.
            4. Collect all results in a dictionary keyed by label.

        **Example**
            .. code-block:: python

                rois = decoder._from_labels("/data/freesurfer/sub-01/mri")

                for label, (roi_name, img) in rois.items():
                    print(f"{label}: {roi_name}, shape={img.shape}")

            Example output::

                #1: /data/freesurfer/sub-01/mri/aseg.mgz
                ROI_01: amygdala_L, shape=(256, 256, 256)
                ROI_02: hippocampus_R, shape=(256, 256, 256)

        .. note::
        - Designed for FreeSurfer-style parcellation volumes where integer labels
            correspond to anatomical structures.
        - The extracted ROIs are converted to binary NIfTI masks for use in
            downstream decoding or visualization pipelines.
        - Requires that ``self.roi_labels`` (list of integers) and
            ``self.roi_src`` are defined in the calling object.
        - Depends on:
            * :func:`utils.FindFiles` – to locate files
            * :func:`utils.get_file_from_substring` – to filter by substring
            * :func:`self.select_labels_from_mgh` – to extract label indices
            * :func:`self._fetch_key` – to parse ROI metadata
        """
        
        logger.info(f"Loading masks from '{mask_dir}'")
        masks = utils.get_file_from_substring(
            self.roi_src,
            utils.FindFiles(
                mask_dir,
                extension=self.extension,
            ).files
        )
        
        if not isinstance(masks, list):
            masks = [masks]

        return_masks = {}
        for ix, m in enumerate(masks):
            logger.info(f" #{ix+1}: {m}")
            # Create a binary mask from labeled mgz
            tmp_mask = self.select_labels_from_mgh(
                input_file=m,
                labels=self.roi_labels,
                mode="nifti",
                binary=True,
            )

            lbl, roi_n = self._fetch_key(m, roi_name=self.roi_name)
            return_masks[lbl] = [roi_n, tmp_mask]

        return return_masks


    @classmethod
    def _fetch_key(self, i, roi_name=None):
        """
        Derive a hemisphere label and ROI name from a file path.

        This internal helper parses the provided mask or ROI filename to infer
        both the **hemisphere** (``left``, ``right``, or ``uni``) and the
        **ROI name** (from BIDS-style key–value pairs, e.g. ``roi-amygdala``).

        :param str i:
            Path to a NIfTI or MGH file whose name encodes hemisphere and/or ROI information.
        :param str | None roi_name:
            Optional override for the ROI name. If provided, it is returned directly
            without parsing the file path. If ``None``, the function attempts to infer
            the name from the filename using BIDS conventions.

        :returns:
            A tuple ``(lbl, roi_name)`` containing:
            
            * ``lbl`` – hemisphere label; one of ``{"left", "right", "uni"}``
            * ``roi_name`` – extracted or default ROI name
        :rtype:
            tuple[str, str]

        **Logic**
            1. The hemisphere is determined based on filename content:
            - Starts with ``lh.`` or contains ``hemi-l`` → ``"left"``
            - Starts with ``rh.`` or contains ``hemi-r`` → ``"right"``
            - Otherwise → ``"uni"`` (unilateral or unspecified)
            2. The ROI name is parsed if not explicitly provided:
            - Attempts to extract BIDS-style components via :func:`utils.split_bids_components`.
            - If the filename includes a ``roi-`` key, that value is used.
            - If parsing fails or no key is found, defaults to ``"roi"`` or ``"roi1"``.
            3. Emits a warning if the file is not BIDS-compliant or if default
            naming may cause collisions.

        **Example**
            .. code-block:: python

                lbl, roi_name = decoder._fetch_key("sub-01_hemi-L_roi-amygdala_mask.nii.gz")
                print(lbl, roi_name)
                # left amygdala

            .. code-block:: python

                lbl, roi_name = decoder._fetch_key("/data/masks/custom_mask.nii.gz")
                # [WARNING] not BIDS-format → defaults to 'roi'
                print(lbl, roi_name)
                # uni roi

        .. note::
        - The method expects filenames to follow BIDS-like conventions
            (e.g., ``sub-01_hemi-L_roi-hippocampus_mask.nii.gz``).
        - Returns safe defaults if parsing fails, but name collisions are possible.
        - Depends on :func:`utils.split_bids_components` for BIDS key extraction.
        - Typically used by ROI and mask loading helpers
            (:func:`_from_file`, :func:`_from_directory`, :func:`_from_labels`).
        """

        path_base = os.path.basename(i).lower()
        if path_base.startswith("lh.") or "hemi-l" in path_base:
            lbl = "left"
        elif os.path.basename(i).startswith("rh.") or "hemi-r" in path_base:
            lbl = "right"
        else:
            lbl = "uni"

        if roi_name is None:
            try:
                bids_comps = utils.split_bids_components(i, add_elements='roi')
            except Exception:
                logger.warning(f"{i} is not in BIDS-format. Cannot extract 'roi-<roi_name>' key, so defaulting to 'roi' as roi_name. This can result in clashing names!")
                roi_name = "roi"

            if roi_name is None:
                if "roi" in bids_comps:
                    roi_name = bids_comps["roi"]
                    roi_name.split(".")[0]
                else:
                    logger.warning(f"{i} does not have 'roi-' key, defaulting to 'roi' as roi name. This can result in clashing names!")
                    roi_name = "roi1"

        return lbl, roi_name


    @classmethod
    def _sanitize(self, img, fill=0.0):
        """
        Load and sanitize a NIfTI image by replacing invalid voxel values.

        This class method ensures that voxel data from a NIfTI image (or a file path)
        is numerically stable and free of NaN or infinite values prior to further
        processing or integer casting.

        :param str | nibabel.Nifti1Image img:
            Either a path to a NIfTI image file (``.nii``/``.nii.gz``) or an already
            loaded :class:`nibabel.Nifti1Image` object.
        :param float fill:
            Replacement value for invalid voxels (NaN, +Inf, −Inf).  
            Default is ``0.0``.

        :returns:
            A sanitized voxel array of type ``float32`` with all invalid values
            replaced by ``fill``.
        :rtype:
            numpy.ndarray

        **Processing Steps**
            1. If ``img`` is a file path, it is loaded using :func:`nibabel.load`.
            2. The image data is read into memory as a ``float32`` NumPy array
            via :meth:`nibabel.Nifti1Image.get_fdata`.
            3. All non-finite values (NaN, +∞, −∞) are replaced with ``fill``
            using :func:`numpy.nan_to_num`.
            4. Returns the cleaned array for downstream use.

        **Example**
            .. code-block:: python

                arr = Decoder._sanitize("sub-01_roi-mask.nii.gz", fill=0)
                print(arr.shape, np.isfinite(arr).all())
                # (64, 64, 36) True

            .. code-block:: python

                img = nib.load("roi_mask.nii.gz")
                data = Decoder._sanitize(img, fill=-1)
                print(np.unique(data))
                # [-1.  0.  1.]

        .. note::
        - The method operates entirely in memory; it does **not** modify the
            original file on disk.
        - Ensures compatibility with downstream integer conversion steps
            (e.g., mask binarization).
        - Typically used internally by ROI and mask loading utilities such as
            :func:`_from_file` and :func:`_from_directory`.
        """

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
        Extract one or more label values from a FreeSurfer ``.mgz``/``.mgh`` file and
        return a new image containing only those regions.

        This method sanitizes and filters a labeled anatomical or parcellation volume,
        optionally converting it into a binary mask suitable for decoding or region-based
        analyses.

        :param str input_file:
            Path to a FreeSurfer ``.mgz`` or ``.mgh`` segmentation file (e.g., ``aseg.mgz``).
        :param sequence[int] labels:
            Sequence of integer label values to extract from the volume.
        :param str mode:
            Output image format. Options:
            - ``"mgz"`` (default): returns an :class:`nibabel.MGHImage`
            - ``"nifti"`` / ``"nii"`` / ``"nii.gz"``: returns a :class:`nibabel.Nifti1Image`
        :param bool binary:
            If ``True``, converts the extracted labels into a binary mask (1 = label voxel, 0 = background).  
            If ``False`` (default), preserves original label values.
        :param float fill:
            Replacement value for invalid voxels (NaN, +∞, −∞). Default is ``0.0``.

        :returns:
            A new NIfTI or MGH image containing only the selected labels (or a binary mask).
        :rtype:
            nibabel.Nifti1Image | nibabel.MGHImage

        **Processing Steps**
            1. Load the input image with :func:`nibabel.load`.
            2. Sanitize voxel data via :func:`_sanitize` (replace NaN/±∞ with ``fill``).
            3. Convert the image data to integer label space using ``np.int32``.
            4. Extract the specified labels:
            - If ``binary=True``, output is ``uint8`` with 1/0 values.
            - If ``binary=False``, output retains integer label values.
            5. Construct a new image using the same affine but a fresh header and dtype.
            6. Return the cleaned and filtered image in the desired format (MGZ or NIfTI).

        **Example**
            .. code-block:: python

                img = Decoder.select_labels_from_mgh(
                    input_file="aseg.mgz",
                    labels=[17, 53],   # Left/Right hippocampus
                    mode="nifti",
                    binary=True
                )
                nib.save(img, "hippocampus_mask.nii.gz")

        **Example Output**
            ::
                Extracted labels: [17, 53]
                Output dtype: uint8
                Output shape: (256, 256, 256)
                Saved as: hippocampus_mask.nii.gz

        .. note::
        - Automatically replaces NaNs or ±∞ values before integer conversion.
        - Uses integer-friendly dtypes for compatibility with nearest-neighbor resampling:
            * ``uint8`` for binary masks  
            * ``int32`` for label-preserving outputs
        - The returned image **does not reuse** the original header, ensuring consistency
            with the new data type.
        - Supports both FreeSurfer’s native ``.mgz`` format and NIfTI output for
            downstream tools.
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
    

class PrepareBetas:
    """
    Prepare trialwise beta images for decoding analyses.

    The :class:`PrepareBetas` class handles loading, sanitizing, and merging
    trialwise beta images across multiple preprocessing pipelines
    (e.g., **GLMsingle**, **stGLM**, **Bach**, or **HALFpipe**).  
    It standardizes voxel data, removes invalid values, and produces
    a clean 4D image with associated trial labels and optional run group
    identifiers.

    The class supports two main initialization modes:

    1. **Direct mode** – when a single pre-merged NIfTI image and trial list
       are provided via ``beta_file`` and ``trial_list``.
    2. **Automatic mode** – when only configuration arguments are passed;
       it will invoke :meth:`load_and_merge_betas` to locate and merge
       source files automatically.

    **Typical Workflow**

        .. code-block:: python

            prep = PrepareBetas(
                beta_file="sub-01_model-lsa_desc-merged_betas.nii.gz",
                trial_list=["CS-", "CS+", "CS-", "CS+"],
                standardize="zscore"
            )

            betas_img = prep.betas
            trials = prep.trial_list
            print(prep.do_standardization)  # False (if z-scored)

        or automatically:

        .. code-block:: python

            prep = PrepareBetas(
                subject="sub-01",
                beta_dir="/data/derivatives/stglm",
                model="lsa",
                save_imgs=True
            )

            betas_img, trials, is_std, groups = prep.betas, prep.trial_list, prep.do_standardization, prep.groups

    :param str | None beta_file:
        Path to a single merged 4D beta image. If provided, ``trial_list`` must
        also be given. If omitted, the class automatically loads betas from disk
        using :meth:`load_and_merge_betas`.
    :param list | numpy.ndarray | None trial_list:
        List of condition or event names (one per beta volume).  
        Required when ``beta_file`` is specified.
    :param kwargs:
        Additional parameters forwarded to :meth:`sanitize_img` or
        :meth:`load_and_merge_betas`.

    **Attributes**
        - ``betas`` : nibabel.Nifti1Image  
          The cleaned and optionally merged beta image.
        - ``trial_list`` : list[str]  
          The list of trial identifiers corresponding to beta volumes.
        - ``do_standardization`` : bool  
          Whether downstream scaling (e.g., `StandardScaler`) is still needed.
        - ``groups`` : list[int] | None  
          Optional run identifiers if parsed from filenames.

    **Main Methods**
        - :meth:`sanitize_img` — replace invalid voxels, clip, and optionally z-score.
        - :meth:`load_and_merge_betas` — load and concatenate trialwise betas,
          auto-detecting file structure based on source type.
        - :meth:`return_trials` — return the list of trials for convenience.

    **Notes**
        - Compatible with trialwise beta extraction pipelines including:
          * **GLMsingle** (`model-typeb`, `model-typed`)
          * **stGLM** (JSON sidecars with TrialList)
          * **HALFpipe** (condition-wise 4D betas + label mapper)
          * **Bach** (optionally includes temporal derivative images)
        - Produces standardized, analysis-ready input for decoding and ROI analyses.
    """

    def __init__(
            self,
            beta_file: str=None,
            trial_list: list=None,
            **kwargs
        ):
                
        if isinstance(beta_file, str):
            logger.info(f"Beta-file: '{beta_file}'")
            self.betas, self.do_standardization = self.sanitize_img(
                beta_file,
                **kwargs
            )
            
            if not isinstance(trial_list, (list, np.ndarray)):
                logger.exception("Please specify a list representing the trials")

            self.trial_list = trial_list
        else:
            self.betas, self.trial_list, self.do_standardization, self.groups = self.load_and_merge_betas(
                **kwargs
            )

    def return_trials(self):
        return self.trial_list

    @classmethod
    def sanitize_img(
        self,
        img,
        fill=0.0,
        clip=1e6,
        standardize=None,
        **kwargs
    ):
        
        """
        Sanitize an image by replacing invalid values, optionally clipping outliers,
        and performing z-score standardization.

        This method loads a NIfTI image (3D or 4D), replaces non-finite voxel values
        (NaN, +∞, −∞) with a specified fill value, optionally clips extreme values,
        and can standardize each voxel’s time course or beta series using z-scoring.
        The output image preserves the input’s affine and header.

        :param str | nibabel.Nifti1Image img:
            Path to a NIfTI image file (``.nii``/``.nii.gz``) or an already loaded
            :class:`nibabel.Nifti1Image` object.
        :param float fill:
            Replacement value for invalid voxel values (NaN, +∞, −∞). Default is ``0.0``.
        :param float | None clip:
            Optional threshold for clipping absolute voxel intensities to the range
            ``[-clip, clip]``. If ``None``, clipping is disabled. Default is ``1e6``.
        :param str | None standardize:
            If set to ``"zscore"``, performs voxelwise z-score normalization across the
            last dimension (e.g., time or trial). If set to ``"range"``, it does not
            divide by standard deviation, but by its range. Other values disable
            standardization.
        :param kwargs:
            Additional keyword arguments (currently unused; reserved for extension).

        :returns:
            A tuple ``(clean_img, do_standardization)`` where:

            * ``clean_img`` – a new NIfTI image with sanitized data  
            * ``do_standardization`` – bool flag indicating whether further
            pipeline-level standardization (e.g., `StandardScaler`) should be applied
            downstream
        :rtype:
            tuple[nibabel.Nifti1Image, bool]

        **Processing Steps**
            1. Load the input image using :func:`nilearn.image.load_img`.
            2. Replace NaN/±∞ values with ``fill`` via :func:`numpy.nan_to_num`.
            3. Optionally clip voxel intensities to the range ``[-clip, clip]``.
            4. If ``standardize='zscore'`` or ``standardize='range'``:
            - Compute voxelwise mean and standard deviation across the last axis.
            - Apply normalization ``(x - μ) / σ`` or ``(x - μ) / (max - min)``.
            - Replace any residual NaN/Inf with zeros.
            - Disable further standardization in the decoding pipeline.
            5. Return a new NIfTI image via :func:`nilearn.image.new_img_like`.

        **Example**
            .. code-block:: python

                clean_img, do_std = Decoder.sanitize_img(
                    "sub-01_betas.nii.gz",
                    fill=0,
                    clip=5e4,
                    standardize="zscore"
                )

                print(clean_img.shape, do_std)
                # (64, 64, 36, 120) False

        **Notes**
            - Works with both 3D and 4D NIfTI images.
            - Preserves the original affine and header.
            - Returns ``do_standardization=False`` when z-scoring is applied internally,
            allowing pipeline configuration to skip redundant scaling steps.
            - Designed for preprocessing beta images prior to ROI or searchlight decoding.
        """

        img = image.load_img(img)
        data = img.get_fdata(dtype=np.float32)

        # replace NaN/±inf
        data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)

        # optional safety clip to avoid huge outliers
        if clip is not None:
            data = np.clip(data, -clip, clip)

        do_standardization = True
        if standardize in ["zscore", "range"]:
            do_standardization = False
        
            if standardize == "zscore":
                logger.info("Z-scoring betas before decoding; removing 'StandardScalar' from pipeline")

                mu = data.mean(axis=-1, keepdims=True)
                sigma = data.std(axis=-1, ddof=1, keepdims=True)
                zdata = (data - mu) / np.where(sigma == 0, 1, sigma)
                data = np.nan_to_num(zdata, copy=False)

            elif standardize == "range":
                logger.info("Normalizing betas by range before decoding; removing 'StandardScalar' from pipeline")

                # mean across trials (axis = -1 because PANIC stores betas as [voxels × trials])
                mu = data.mean(axis=-1, keepdims=True)

                # compute range across trials
                data_min = data.min(axis=-1, keepdims=True)
                data_max = data.max(axis=-1, keepdims=True)
                denom = data_max - data_min

                # avoid divide-by-zero (flat voxels)
                denom = np.where(denom == 0, 1, denom)

                # mean-center + divide by range
                rdata = (data - mu) / denom

                # replace NaN / Inf just like z-score code does
                data = np.nan_to_num(rdata, copy=False)

            else:
                logger.error(f"'standardize' must be on of 'range' or 'zscore', not '{standardize}'")

        return image.new_img_like(img, data, copy_header=True), do_standardization
    

    @classmethod
    def load_and_merge_betas(
            self,
            subject,
            beta_dir="/mnt/d/fMRI/HRA/derivatives/stglm",
            save_imgs=False,
            output_dir=None,
            derivative=False,
            label_mapper=None,
            model="lsa",
            filters=None,
            **kwargs
        ):
        """
        Load, sanitize, and concatenate trialwise beta images for a subject from
        different preprocessing pipelines, returning a single 4D image plus
        trial labels and (optional) run groups.

        Supported sources are inferred from ``beta_dir``'s basename:
        - **"glmsingle"** – expects per-run/per-condition beta files and a
        ``trial_list.txt``; model name is mapped to GLMsingle types:
        ``"lss" → "typeb"``, ``"lsa" → "typed"``.
        - **"stglm"** – expects per-run beta files with JSON sidecars carrying
        ``"TrialList"``; groups are parsed from filenames (``run-XX``).
        - **"bach"** – D. Bach format; may contain derivative estimates (see below).
        - **"halfpipe"** – condition-wise single-trial files; requires
        ``label_mapper`` to select conditions.

        Optionally, trialwise derivative images can be selected for Bach-format
        directories by taking every other file.

        :param str subject:
            Subject identifier (e.g., ``"sub-01"``) whose beta directory is
            ``<beta_dir>/<subject>``.
        :param str beta_dir:
            Root directory of beta images. Its basename determines the source
            format (``"stglm"``, ``"glmsingle"``, ``"halfpipe"``, or ``"bach"``).
            Default: ``"/mnt/d/fMRI/HRA/derivatives/stglm"``.
        :param bool save_imgs:
            If ``True``, writes the merged 4D image to disk.
        :param str | None output_dir:
            Output directory used when ``save_imgs=True``. If ``None``, defaults
            to ``<beta_dir>/<subject>``.
        :param bool derivative:
            For Bach format, select every other beta file (temporal derivative).
            Ignored for other sources.
        :param dict | None label_mapper:
            Required for **halfpipe** to specify which conditions to include
            (keys are condition names). Ignored for other sources.
        :param str model:
            Model identifier. For GLMsingle, mapped via
            ``{"lss": "typeb", "lsa": "typed"}``. For other sources, used to
            filter filenames (e.g., ``"desc-lsa"``).
        :param kwargs:
            Passed to :meth:`sanitize_img` (e.g., ``standardize="zscore"``, ``fill``, ``clip``).

        :returns:
            A tuple ``(beta_imgs, trials, is_standardized, groups)``:
            
            * ``beta_imgs`` – 4D NIfTI image of concatenated betas
            * ``trials`` – list/array of trial labels (length equals last dim of ``beta_imgs``)
            * ``is_standardized`` – ``bool`` flag from :meth:`sanitize_img`
            indicating whether pipeline-level standardization should still occur
            * ``groups`` – list of run IDs per trial or ``None`` if not applicable
        :rtype:
            tuple[nibabel.Nifti1Image, list[str] | numpy.ndarray, bool, list[int] | None]

        **Workflow**
            1. Discover beta files under ``<beta_dir>/<subject>`` based on the
            source format and requested ``model``.
            2. (Bach) If ``derivative=True``, select every other file.
            3. Load images and track trial counts; for **stglm** and **halfpipe**,
            parse ``run-XX`` from filenames to construct ``groups`` (with a
            fallback when parsing fails).
            4. Concatenate all beta images into a single 4D image via
            :func:`nilearn.image.concat_imgs`.
            5. Determine the trial list:
            - **glmsingle / bach**: read from ``trial_list.txt`` in subject dir
            - **stglm**: concatenate ``"TrialList"`` from JSON sidecars
            - **halfpipe**: accumulated earlier while selecting files
            6. Sanitize the merged image using :meth:`sanitize_img` (replace NaN/±∞,
            optional clipping, optional z-score standardization).
            7. Optionally save the merged image if ``save_imgs=True``.

        **Saved filename (when ``save_imgs=True``)**
            ``{subject}_model-{model}_source-{src}_desc-merged_betas.nii.gz``

        **Notes**
            - Validates that the last dimension of ``beta_imgs`` matches the number
            of ``trials`` and logs an error if not.
            - ``groups`` is ``None`` unless a source provides run parsing (e.g., stGLM,
            halfpipe); for halfpipe, conditions are filtered by ``label_mapper``.
            - The return flag ``is_standardized`` allows a downstream pipeline to
            disable redundant scaling if z-scoring was already applied here.
        """

        # helper to parse run-XX from filenames like "...run-02_..."
        run_re = re.compile(r"run-(\d+)", flags=re.IGNORECASE)
        groups = None

        #-----------------------------------------------------------------------
        # stglm | halfpipe | glmsingle
        src = os.path.basename(beta_dir)
        subject_betas = opj(beta_dir, subject)

        assert os.path.exists(subject_betas), FileNotFoundError(f"Input beta directory '{subject_betas}' does not exist")
        logger.info(f"Loading betas from: '{subject_betas}'")

        #-----------------------------------------------------------------------
        # define GLMsingle mapper
        model_mapper = {
            "lss": "typeb",
            "lsa": "typed"
        }

        #-----------------------------------------------------------------------
        # stglm | glmsingle need to be concatenated; HALFpipe comes concatenated
        allowed = ['glmsingle', 'bach', 'halfpipe', 'stglm']
        if src == "glmsingle":
            search = f"model-{model_mapper[model]}_beta-"
        elif src == "bach":
            search = "beta_"
        elif src in ["halfpipe", "stglm"]:
            search = [f"feature-{model}_condition-", "effect_statmap.nii.gz"]
        else:
            raise ValueError(f"Source must be on of {allowed}, not '{src}'")
        
        # add custom filters
        if filters is not None:
            if isinstance(filters, str):
                filters = [filters]

            if len(filters)>0:
                search += filters

        logger.info(f"Search criteria: {search}")
        m_files = utils.FindFiles(
            subject_betas,
            extension=".nii.gz",
            filters=search
        ).files

        assert len(m_files)>0, ValueError(f"No *.nii.gz files in '{subject_betas}'")
    
        #-----------------------------------------------------------------------
        # D. Bach's beta values
        if derivative:
            logger.info("Directory contains trialwise estimates of the temporal derivative. Selecting every other beta file.")
            m_files = m_files[0::2]
        
        #-----------------------------------------------------------------------
        # HALFpipe outputs condition-wise single-trial files > match with "label_dict" in config.yml
        niimgs = []
        trials = []
        groups = []
        if src in ["halfpipe", "stglm"]:

            assert isinstance(label_mapper, dict), f"When HALFpipe is used, 'label_mapper' must be a dictionary with keys representing the stimuli to include, not '{label_mapper}'"

            # select ev-specific files
            include_events = list(label_mapper.keys())
            for i in include_events:

                incl_files = utils.get_file_from_substring(
                    [f"-{i}_stat"],
                    m_files
                )

                if isinstance(incl_files, str):
                    incl_files = [incl_files]

                if not isinstance(incl_files, list):
                    raise TypeError(f"We should have a list of beta-values by now, not {incl_files}")

                # fetch nr of volumes and fill in with label_mapper
                logger.info(f"Found {len(incl_files)} files for '{i}' [model = {model}]")
                for f in incl_files:
                    img = nib.load(f)
                    header = img.header
                    n_vols = header.get("dim")[4]
                    trials.extend([i] * n_vols)
                    niimgs.append(img)

                    # groups (run ID replicated per event in that run)
                    m = run_re.search(f)
                    if m:
                        run_id = int(m.group(1))      # 1..R
                    else:
                        # fallback: index of file within condition list
                        run_id = len(set(groups)) + 1
                        logger.warning(f"Could not parse run-XX from '{f}'. Using fallback run_id={run_id}")

                    groups.extend([run_id] * n_vols)

        else:
            logger.info(f"Found {len(m_files)} files for {model}-model")
            for f in m_files:
                img = nib.load(f)
                header = img.header
                n_vols = header.get("dim")[4]
                # trials.extend([i] * n_vols)
                niimgs.append(img)

        #-----------------------------------------------------------------------
        # concatenate
        logger.info("Concatenating these files into single 4D-object")
        beta_imgs = image.concat_imgs(niimgs)
        
        #-----------------------------------------------------------------------
        # glmsingle has separate trial_list.txt file in directory; stGLM has json sidecar
        # HALFpipe trials are read above
        if src in ["glmsingle", "bach"]:
            txt_files = utils.FindFiles(
                subject_betas,
                extension=".txt"
            ).files

            trial_file = utils.get_file_from_substring(
                "trial_list",
                txt_files
            )

            trials = np.loadtxt(trial_file, dtype=str)

        #-----------------------------------------------------------------------
        # verify
        if beta_imgs.shape[-1] != len(trials):
            logger.error(f"Number of beta-images ({beta_imgs.shape[-1]}) does not match length of label/trial list ({len(trials)})")

        #-----------------------------------------------------------------------
        # sanitize images
        logger.info("Sanitizing beta images (e.g., remove NaN/inf and set float)")
        beta_imgs, is_standardized = self.sanitize_img(beta_imgs, **kwargs)

        #-----------------------------------------------------------------------
        # save merged?
        if save_imgs:
            if output_dir is None:
                output_dir = subject_betas

            fname = opj(
                output_dir,
                f"{subject}_model-{model}_source-{src}_desc-merged_betas.nii.gz"
            )

            logger.info(f"Saving merged beta image as '{fname}'")
            beta_imgs.to_filename(fname)

        return beta_imgs, trials, is_standardized, groups


class MaskAndFilterBetas:
    """
    Extract and filter voxelwise beta values within an ROI mask.

    The :class:`MaskAndFilterBetas` class combines ROI masking and trial filtering
    into a single preprocessing step. It resamples the mask to match the beta
    image grid, removes non-varying or invalid voxels, extracts beta features,
    and filters trials according to a label mapping for classification.

    This class is typically used directly after :class:`PrepareBetas` to generate
    clean trialwise feature matrices and corresponding label vectors.

    **Typical Workflow**

        .. code-block:: python

            mf = MaskAndFilterBetas(
                betas=prep.betas,
                mask="roi_amygdala.nii.gz",
                trial_list=prep.trial_list,
                label_mapper={"CS-": 0, "CS+": 1}
            )

            X = mf.X               # (n_trials, n_voxels_in_roi)
            y = mf.labels          # integer labels
            roi_idx = mf.roi_linidx

    :param nibabel.Nifti1Image | str betas:
        4D beta image from :class:`PrepareBetas`.
    :param nibabel.Nifti1Image | str mask:
        ROI or brain mask to extract features from.
    :param list | numpy.ndarray trial_list:
        Trial or condition names, one per beta volume.
    :param dict label_mapper:
        Dictionary mapping condition names to integer labels (e.g., ``{"CS-": 0, "CS+": 1}``).
    :param kwargs:
        Additional keyword arguments passed to :meth:`_masker` (e.g., ``var_thr``).

    **Attributes**
        - ``betas`` : nibabel.Nifti1Image  
          The full beta image (unmasked).
        - ``mask`` : nibabel.Nifti1Image  
          The ROI mask image.
        - ``betas_in_mask`` : numpy.ndarray  
          Masked voxel data (n_trials × n_voxels_in_roi).
        - ``roi_linidx`` : numpy.ndarray  
          Linear indices of voxels retained in the ROI mask.
        - ``X`` : numpy.ndarray  
          Filtered feature matrix (subset of betas_in_mask).
        - ``trials`` : list[str]  
          Filtered trial list.
        - ``labels`` : numpy.ndarray  
          Integer labels aligned with X and trials.

    **Main Methods**
        - :meth:`_masker` — resample mask to beta grid, apply variance filtering,
          and extract ROI voxel data.
        - :meth:`filter_trials` — select trials matching the provided label map.
        - :meth:`return_betas` — return the masked (unfiltered) beta data.

    **Notes**
        - Resampling uses ``nilearn.image.resample_to_img`` to align mask and beta
          affine/shape, typically with nearest-neighbor interpolation.
        - Automatically excludes voxels with zero variance or invalid values.
        - Logs the number of retained voxels and included trials per condition.
        - Produces ROI-specific feature matrices suitable for decoding, MVPA,
          or searchlight analyses.
    """

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
        self.betas_in_mask, self.roi_linidx = self._masker(**kwargs)

        # filter trials and set labels
        self.X, self.trials, self.labels = self.filter_trials()


    def filter_trials(self):
        """
        Filter the loaded trials and corresponding beta images based on the
        user-defined label mapping.

        This method selects only those trials whose condition names appear
        in ``self.label_mapper``. It returns the filtered beta images,
        trial names, and corresponding integer class labels, and logs a
        summary of included conditions and trial counts.

        :returns:
            A tuple ``(X_filtered, filtered_trial_list, labels)`` containing:

            * ``X_filtered`` – subset of the beta image data (``n_trials × n_features``)
            * ``filtered_trial_list`` – list of included trial names
            * ``labels`` – integer label array aligned with ``X_filtered`` and ``filtered_trial_list``
        :rtype:
            tuple[numpy.ndarray, list[str], numpy.ndarray]

        **Processing Steps**
            1. Iterate over ``self.trial_list`` (trial names).
            2. Retain trials whose names match keys in ``self.label_mapper``.
            3. Collect their integer labels (``val`` from ``label_mapper``)
            and indices.
            4. Subset ``self.betas_in_mask`` and ``self.trial_list`` using
            the retained indices.
            5. Log the number of included trials per class and overall count.

        **Example**
            .. code-block:: python

                # Example configuration
                decoder.label_mapper = {"CS-": 0, "CS+": 1}
                decoder.trial_list = ["CS-", "CS+", "CS+", "CS-"]
                decoder.betas_in_mask = np.random.randn(4, 500)

                X, trials, labels = decoder.filter_trials()
                print(trials)  # ['CS-', 'CS+', 'CS+', 'CS-']
                print(labels)  # [0, 1, 1, 0]
                print(X.shape) # (4, 500)

        **Logs**
            .. code-block::

                Total included trials: 4 (CS-: 2, CS+: 2)
                Filtered beta images: 4

        **Notes**
            - ``self.label_mapper`` must be a dictionary mapping trial names
            to integer class labels.
            - ``self.trial_list`` and ``self.betas_in_mask`` must have the
            same length along the first dimension.
            - This method is typically called before cross-validation or
            ROI decoding to align input data with the chosen conditions.
        """

        filtered_trials = []
        labels = []
        trial_indices = []
        condition_counts = Counter()

        for i, t in enumerate(self.trial_list):
            for key, val in self.label_mapper.items():
                if key in t:
                    filtered_trials.append(key)
                    labels.append(val)
                    trial_indices.append(i)
                    condition_counts[key] += 1
                    break  # each trial may belong to only one condition

        labels = np.asarray(labels)
        n_trials = len(labels)

        # Count samples per decoded class
        label_counts = Counter(labels.tolist())

        condition_parts = [
            f"{k}: {condition_counts.get(k, 0)}"
            for k in self.label_mapper.keys()
        ]

        label_parts = [
            f"{lab}: {label_counts.get(lab, 0)}"
            for lab in sorted(label_counts)
        ]

        if n_trials == 0:
            msg = (
                "Number of trials equals 0. "
                "Please check trial names to make sure filtering is correct. "
                f"Conditions: {', '.join(condition_parts)} | "
                f"Classes: {', '.join(label_parts)}"
            )
            logger.error(msg)
            raise ValueError(msg)

        logger.info(
            "Total included trials: %d | Conditions: %s | Classes: %s",
            n_trials,
            ", ".join(condition_parts),
            ", ".join(label_parts),
        )

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
            zooms=None,
            masker_kws={},
            fit_kws={}
        ):
        """
        Create and apply a NIfTI masker aligned with the beta images,
        ensuring voxel-level intersection between the ROI mask and valid
        beta signal.

        This method resamples the ROI mask to match the spatial grid of
        the beta images, removes non-varying or invalid voxels, and extracts
        the resulting beta features using :class:`nilearn.maskers.NiftiMasker`.

        :param str interpolation:
            Interpolation method for resampling the ROI mask to the beta image
            grid. Typically ``"nearest"`` (default) to preserve binary values.
        :param str | None output_file:
            Optional file path to save the resampled and intersected mask.
            If ``None``, no file is written.
        :param float var_thr:
            Minimum voxelwise variance threshold (in beta space). Voxels with
            variance ≤ ``var_thr`` are excluded. Default: ``0.0``.
        :param list zooms:
            Instead of using beta images as default for resampling, allow custom
            zoom. This is particularly useful for searchlight, where you may
            want to downsample the data to reduce the number of centers evalua-
            ted.
        :param dict masker_kws:
            Additional keyword arguments passed to
            :class:`nilearn.maskers.NiftiMasker` during initialization.
        :param dict fit_kws:
            Additional keyword arguments passed to
            :meth:`nilearn.maskers.NiftiMasker.fit_transform`.

        :returns:
            A tuple ``(X, roi_linidx)`` where:

            * ``X`` – 2D NumPy array of extracted beta values
            (``n_samples × n_voxels_in_roi``)
            * ``roi_linidx`` – 1D array of linear voxel indices within the
            3D mask volume, corresponding to the selected ROI voxels
        :rtype:
            tuple[numpy.ndarray, numpy.ndarray]

        **Processing Steps**
            1. Resample the ROI mask to match the affine and shape of the
            first beta image volume using :func:`nilearn.image.resample_to_img`.
            2. Compute voxelwise variance across time/trials in the beta image.
            Mask out voxels that are NaN, non-finite, or below ``var_thr``.
            3. Intersect the resampled ROI with this “support” mask to obtain
            the valid in-field-of-view (FOV) voxels.
            4. Compute linear voxel indices (``roi_linidx``) for these valid
            voxels.
            5. Optionally save the resulting intersected mask if
            ``output_file`` is provided.
            6. Initialize a :class:`nilearn.maskers.NiftiMasker` using the
            resampled mask and extract beta values via
            :meth:`fit_transform`.

        **Example**
            .. code-block:: python

                X, linidx = decoder._masker(
                    interpolation="nearest",
                    var_thr=1e-6,
                    output_file="sub-01_resampled_mask.nii.gz"
                )

                print(X.shape)         # (n_trials, n_voxels_in_roi)
                print(len(linidx))     # number of ROI voxels retained

        **Logs**
            .. code-block::

                Resampling mask (64, 64, 36) to affine of beta-image (64, 64, 36)
                Identifying valid voxels (var>1e-06)
                3210/3548 voxels inside beta FOV
                Extract betas with resampled/validated mask

        **Notes**
            - The mask is always resampled to the **first beta volume** to
            ensure spatial alignment.
            - Voxels with zero or near-zero variance are excluded to avoid
            degenerate features.
            - Returns both the extracted feature matrix and voxel indices for
            downstream reconstruction (e.g., mapping weights or contributions
            back into 3D space).
            - Designed for internal use during ROI and searchlight decoding.
        """

        betas_first_vol = image.index_img(self.betas, 0)
        
        # if custom zooms, downsample betas first, then use that as new target
        # for the masks
        if zooms is not None:
            logger.info(f"Applying custom zoom to beta images: {zooms}")
            
            if isinstance(zooms, list):
                if len(zooms)<3:
                    raise ValueError(f"A list of zooms must contain 3 elements, not {len(zooms)}: {zooms}")
                zooms = np.array(zooms)
            elif isinstance(zooms, (int, float)):
                zooms = np.full(3, float(zooms))
            elif isinstance(zooms, np.ndarray):
                pass
            else:
                raise TypeError(f"zooms must be a list of 3 elements, an integer/float, or a numpy array, not type {type(zooms)}")
            
            assert isinstance(zooms, np.ndarray) and len(zooms)==3, f"zooms must be a list of 3 elements, an integer/float, or a numpy array, not {zooms}"

            logger.info(
                "Original beta shape=%s, zooms=%s",
                betas_first_vol.shape,
                betas_first_vol.header.get_zooms()[:3],
            )

            target_affine = nib.affines.rescale_affine(
                betas_first_vol.affine,
                betas_first_vol.shape[:3],
                zooms,
            )

            self.betas = image.resample_img(
                self.betas,
                target_affine=target_affine,
                interpolation="continuous",
                force_resample=True,
                copy_header=True
            )

            # set new template
            betas_first_vol = image.index_img(self.betas, 0)

            logger.info(
                "Resampled beta shape=%s, zooms=%s",
                betas_first_vol.shape,
                betas_first_vol.header.get_zooms()[:3],
            )

        # resample mask to betas
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

        if vox_after == 0:
            raise EmptyMaskError(f"Invalid mask after filtering: voxel count in mask = {vox_after}")
            
        roi_mask_data = roi_in_fov.get_fdata() > 0.5
        roi_linidx = np.flatnonzero(roi_mask_data.ravel())

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
        ), roi_linidx
