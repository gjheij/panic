# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

"""Beta loading helpers.

This module contains a mixin-style implementation intended to be copied into the
class that already provides ``sanitize_img`` and has access to the project-level
``utils`` helper module.
"""

from __future__ import annotations

from collections import defaultdict, deque
import json
import logging
import os
import re
from os.path import join as opj
from typing import Mapping

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image

from lazyfmri import utils

logger = logging.getLogger(__name__)


class BetaLoaderMixin:
    """Mixin implementing beta discovery, loading, event parsing, and merging.

    Notes
    -----
    The surrounding class is expected to provide ``sanitize_img``. The project
    must also expose the existing ``utils.FindFiles`` and
    ``utils.get_file_from_substring`` helpers used by the original implementation.
    """

    _BETA_SOURCES = {"glmsingle", "bach", "halfpipe", "stglm"}
    _GLMSINGLE_MODELS = {"lss": "typeb", "lsa": "typed"}
    _EVENT_COLUMNS = ["onset", "duration", "run_id", "trial_type", "label"]
    _RUN_RE = re.compile(r"run-(\d+)", flags=re.IGNORECASE)

    @classmethod
    def load_and_merge_betas(
        cls,
        subject,
        beta_dir="/mnt/d/fMRI/HRA/derivatives/stglm",
        save_imgs=False,
        output_dir=None,
        derivative=False,
        label_mapper=None,
        model="lsa",
        filters=None,
        **kwargs,
    ):
        """Load, merge, sanitize, and align trialwise beta images.

        The method is intentionally a thin orchestration layer. File discovery,
        source-specific loading, JSON/event parsing, temporal alignment, image
        concatenation, metadata validation, and saving are delegated to dedicated
        helpers so that each behavior can be tested independently.

        The source format is inferred from the basename of ``beta_dir``. The
        supported layouts are ``"glmsingle"``, ``"bach"``, ``"halfpipe"``, and
        ``"stglm"``.

        HALFpipe and stGLM inputs are typically stored condition-wise rather than
        chronologically. For these sources, beta volumes are first loaded in the
        order implied by ``label_mapper`` and the discovered files. Event metadata
        from JSON sidecars is then converted to session-relative onset time and
        sorted chronologically. If valid event metadata is available, the beta
        image, trial labels, and run groups are all reordered with the same
        permutation so that returned metadata continues to describe the returned
        4D image exactly.

        Parameters
        ----------
        subject : str
            Subject identifier, for example ``"sub-01"``. Beta files are searched
            under ``<beta_dir>/<subject>``.
        beta_dir : str, default="/mnt/d/fMRI/HRA/derivatives/stglm"
            Root directory containing subject-level beta directories. Its basename
            determines the source implementation.
        save_imgs : bool, default=False
            If ``True``, write the merged, sanitized, and chronologically aligned
            beta image to disk.
        output_dir : str or os.PathLike or None, default=None
            Destination directory used when ``save_imgs=True``. If omitted, the
            subject beta directory is used. Missing directories are created.
        derivative : bool, default=False
            For Bach input only, select every second discovered beta file starting
            with the first file. For all other sources the flag is ignored and a
            warning is emitted.
        label_mapper : mapping or None, default=None
            Required for HALFpipe/stGLM input. Mapping keys are condition names used
            to identify condition-specific beta files; mapping values are labels
            stored in ``events_df``. The condition names themselves are returned in
            ``trials``.
        model : str, default="lsa"
            Model identifier. For GLMsingle, ``"lss"`` maps to ``"typeb"`` and
            ``"lsa"`` maps to ``"typed"``. For HALFpipe/stGLM, the value is used
            directly in the filename search criteria.
        filters : str or sequence of str or None, default=None
            Additional filename substrings applied during beta-file discovery.
        **kwargs
            Additional keyword arguments forwarded unchanged to ``sanitize_img``.

        Returns
        -------
        beta_imgs : nibabel.spatialimages.SpatialImage
            Concatenated and sanitized 4D beta image. If valid event metadata was
            available, its final dimension is in chronological trial order.
        trials : list
            Trial-type label for each beta volume in ``beta_imgs``. The returned
            sequence is always aligned to the final image order.
        is_standardized : bool
            Standardization flag returned by ``sanitize_img``.
        groups : list[int] or None
            Run identifier for each beta volume for HALFpipe/stGLM input. ``None``
            for sources without run-group metadata. When betas are reordered,
            groups are reordered identically.
        events_df : pandas.DataFrame
            Event table with columns ``onset``, ``duration``, ``run_id``,
            ``trial_type``, and ``label``. For HALFpipe/stGLM, valid onsets are
            converted to session-relative time and sorted chronologically. For
            other sources, or if no valid event metadata is available, the table is
            empty with the expected columns.

        Raises
        ------
        FileNotFoundError
            If the subject directory does not exist, no matching beta files are
            found, no valid images can be loaded, or a required ``trial_list`` file
            is missing.
        TypeError
            If ``filters`` has an unsupported type, a filter is not a string,
            ``label_mapper`` is invalid for HALFpipe/stGLM input, or an internal
            file query returns an unsupported result type.
        ValueError
            If the source or GLMsingle model is unsupported, image concatenation
            fails, image/metadata lengths disagree, event timing cannot be made
            valid, or chronological source and target orders are incompatible.

        Notes
        -----
        JSON sidecar failures do not prevent an otherwise valid beta image from
        loading. They only reduce or eliminate available event metadata. When no
        usable events remain, chronological order cannot be reconstructed and the
        betas are returned in their discovered/source order.
        """
        src = os.path.basename(os.path.normpath(beta_dir)).casefold()
        subject_betas = opj(beta_dir, subject)

        cls._validate_beta_dir(subject_betas)
        cls._validate_source(src)

        logger.info("Loading betas from: '%s'", subject_betas)

        beta_files = cls._find_beta_files(
            subject_betas,
            src=src,
            model=model,
            filters=filters,
            derivative=derivative,
        )

        if src in {"halfpipe", "stglm"}:
            niimgs, trials, groups, events_df = cls._load_statmap_betas(
                beta_files,
                label_mapper=label_mapper,
                model=model,
            )
        else:
            niimgs, trials = cls._load_beta_series(
                beta_files,
                subject_betas=subject_betas,
                model=model,
            )
            groups = None
            events_df = cls._empty_events_df()

        beta_imgs = cls._concat_betas(niimgs)
        cls._validate_beta_metadata(beta_imgs, trials=trials, groups=groups)

        logger.info("Sanitizing beta image (remove NaN/inf and convert dtype)")
        beta_imgs, is_standardized = cls.sanitize_img(beta_imgs, **kwargs)

        if not events_df.empty:
            logger.info("Restoring chronological order of trial structure")
            target_order = events_df["trial_type"].tolist()
            reorder_idx = cls.get_reorder_indices(
                source_order=trials,
                target_order=target_order,
            )

            beta_imgs = cls.reorder_betas(
                beta_imgs,
                source_order=trials,
                target_order=target_order,
            )
            trials = np.asarray(trials)[reorder_idx].tolist()

            if groups is not None:
                groups = np.asarray(groups)[reorder_idx].tolist()
        else:
            trials = list(trials)
            if groups is not None:
                groups = list(groups)

        if save_imgs:
            cls._save_merged_betas(
                beta_imgs,
                subject=subject,
                model=model,
                src=src,
                subject_betas=subject_betas,
                output_dir=output_dir,
            )

        return beta_imgs, trials, is_standardized, groups, events_df

    @classmethod
    def get_reorder_indices(cls, source_order, target_order):
        """Return indices that transform ``source_order`` into ``target_order``.

        Matching is occurrence-aware. This matters for trial sequences because a
        condition label usually occurs many times. The first occurrence of a label
        in ``target_order`` is matched to the first unused occurrence of that label
        in ``source_order``, the second to the second, and so on. Consequently,
        repeated labels retain their within-condition source order while the overall
        sequence is rearranged to match the target.

        Parameters
        ----------
        source_order : sequence
            Labels describing the current order of the objects to be reordered.
        target_order : sequence
            Labels describing the desired order. It must contain exactly the same
            number of occurrences of every label as ``source_order``.

        Returns
        -------
        list[int]
            Positional indices such that ``np.asarray(source_order)[indices]`` is
            equal to ``target_order``.

        Raises
        ------
        ValueError
            If the two sequences have different lengths or different label counts.

        Examples
        --------
        ``source_order = ["A", "A", "B"]`` and
        ``target_order = ["A", "B", "A"]`` produce ``[0, 2, 1]``.
        """
        if len(source_order) != len(target_order):
            raise ValueError(
                "`source_order` and `target_order` must have equal length."
            )

        indices = defaultdict(deque)
        for i, label in enumerate(source_order):
            indices[label].append(i)

        try:
            reorder_idx = [
                indices[label].popleft()
                for label in target_order
            ]
        except (KeyError, IndexError):
            raise ValueError(
                "`source_order` and `target_order` have different label counts."
            ) from None

        if any(indices.values()):
            raise ValueError(
                "`source_order` and `target_order` have different label counts."
            )

        return reorder_idx

    @classmethod
    def reorder_betas(cls, beta_imgs, source_order, target_order):
        """Reorder beta volumes from a source trial order to a target trial order.

        This convenience wrapper validates the image length, computes an
        occurrence-aware permutation with :meth:`get_reorder_indices`, and applies
        it directly to the fourth NIfTI dimension using nibabel's ``slicer``.

        Parameters
        ----------
        beta_imgs : nibabel.spatialimages.SpatialImage
            4D beta image whose final dimension corresponds one-to-one with
            ``source_order``.
        source_order : sequence
            Trial labels describing the current beta-volume order.
        target_order : sequence
            Trial labels describing the requested beta-volume order.

        Returns
        -------
        nibabel.spatialimages.SpatialImage
            Image with the final dimension reordered to ``target_order``.

        Raises
        ------
        ValueError
            If the image is not 4D, its number of volumes differs from
            ``source_order``, or the source and target label counts are incompatible.
        """
        if beta_imgs.ndim != 4:
            raise ValueError(
                f"Expected a 4D beta image, got {beta_imgs.ndim} dimensions."
            )

        if beta_imgs.shape[-1] != len(source_order):
            raise ValueError(
                "Number of beta volumes must match `source_order`."
            )

        reorder_idx = cls.get_reorder_indices(source_order, target_order)

        # SpatialFirstSlicer does not support arbitrary fancy indexing. Slice
        # each requested volume as a length-one 4D image, then concatenate the
        # slices in target order. Using ``i:i + 1`` rather than ``i`` preserves
        # the fourth dimension for every intermediate image.
        reordered = [
            beta_imgs.slicer[..., i:i + 1]
            for i in reorder_idx
        ]
        return nib.concat_images(reordered, axis=3)

    @classmethod
    def _validate_beta_dir(cls, subject_betas):
        """Validate that a subject-level beta directory exists.

        Parameters
        ----------
        subject_betas : str or os.PathLike
            Expected subject-level beta directory.

        Raises
        ------
        FileNotFoundError
            If ``subject_betas`` does not exist or is not a directory.
        """
        if not os.path.isdir(subject_betas):
            raise FileNotFoundError(
                f"Input beta directory '{subject_betas}' does not exist or is not a directory"
            )

    @classmethod
    def _validate_source(cls, src):
        """Validate a source identifier inferred from the beta-directory name."""
        if src not in cls._BETA_SOURCES:
            raise ValueError(
                f"Unsupported source '{src}'. Expected one of {sorted(cls._BETA_SOURCES)}"
            )

    @classmethod
    def _find_beta_files(
        cls,
        subject_betas,
        src,
        model,
        filters=None,
        derivative=False,
    ):
        """Discover beta NIfTI files for one source and model.

        Parameters
        ----------
        subject_betas : str or os.PathLike
            Subject-level directory to search.
        src : {"glmsingle", "bach", "halfpipe", "stglm"}
            Source implementation selected from ``beta_dir``.
        model : str
            Requested model name. GLMsingle model aliases are translated to the
            filename convention used by that package.
        filters : str or sequence of str or None, default=None
            Additional filename substrings that all discovered files must satisfy.
        derivative : bool, default=False
            For Bach input, keep every second discovered file beginning with the
            first. Ignored for other source types.

        Returns
        -------
        list[str]
            Sorted, de-duplicated paths to matching ``.nii.gz`` files.

        Raises
        ------
        TypeError
            If ``filters`` is neither a string nor a supported sequence of strings.
        ValueError
            If an unsupported GLMsingle model is requested.
        FileNotFoundError
            If no matching files are found or Bach derivative selection removes all
            candidates.
        """
        if src == "glmsingle":
            model_key = str(model).casefold()
            if model_key not in cls._GLMSINGLE_MODELS:
                raise ValueError(
                    f"Unsupported GLMsingle model '{model}'. "
                    f"Expected one of {sorted(cls._GLMSINGLE_MODELS)}"
                )
            search = [f"model-{cls._GLMSINGLE_MODELS[model_key]}_beta-"]
        elif src == "bach":
            search = ["beta_"]
        else:
            search = [f"feature-{model}_condition-", "effect_statmap.nii.gz"]

        search.extend(cls._normalize_filters(filters))
        logger.info("Search criteria: %s", search)

        beta_files = utils.FindFiles(
            subject_betas,
            extension=".nii.gz",
            filters=search,
        ).files
        beta_files = cls._as_list(beta_files, name="beta-file query")
        beta_files = sorted(dict.fromkeys(beta_files))

        if not beta_files:
            raise FileNotFoundError(
                f"No matching '*.nii.gz' files found in '{subject_betas}'"
            )

        if derivative:
            if src == "bach":
                logger.info(
                    "Selecting every second Bach beta file as temporal-derivative estimates"
                )
                beta_files = beta_files[0::2]
                if not beta_files:
                    raise FileNotFoundError(
                        "No Bach beta files remain after derivative selection"
                    )
            else:
                logger.warning(
                    "'derivative=True' is only supported for Bach input; ignoring it for '%s'",
                    src,
                )

        return beta_files

    @classmethod
    def _normalize_filters(cls, filters):
        """Normalize optional filename filters to a list of strings."""
        if filters is None:
            return []
        if isinstance(filters, str):
            return [filters]
        if not isinstance(filters, (list, tuple, set)):
            raise TypeError(
                "'filters' must be a string or sequence of strings, "
                f"not {type(filters).__name__}"
            )

        filters = list(filters)
        if not all(isinstance(value, str) for value in filters):
            raise TypeError("Every entry in 'filters' must be a string")
        return filters

    @classmethod
    def _as_list(cls, value, name="query"):
        """Normalize common file-query return types to a list.

        Strings are treated as a single result rather than an iterable of
        characters. ``None`` represents no results. Other iterable objects are
        materialized with ``list``.
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return value
        try:
            return list(value)
        except TypeError as exc:
            raise TypeError(
                f"Expected {name} to return a path or iterable of paths, got {value!r}"
            ) from exc

    @classmethod
    def _load_statmap_betas(cls, beta_files, label_mapper, model):
        """Load condition-wise HALFpipe/stGLM beta maps and event metadata.

        For every condition in ``label_mapper``, matching beta files are loaded and
        expanded into one trial label and run-group entry per image volume. JSON
        sidecars are parsed opportunistically: image loading succeeds even if a
        sidecar is missing or malformed.

        Run durations and valid events are accumulated across condition-specific
        files and later passed to :meth:`_build_events_df`, which converts
        run-relative event onsets into a single session-relative chronology.

        Parameters
        ----------
        beta_files : sequence of str
            Candidate HALFpipe/stGLM beta files.
        label_mapper : mapping
            Condition name to output label mapping. Condition names are used to
            identify files and populate ``trial_type``; mapped values populate the
            event ``label`` column.
        model : str
            Model identifier used only for diagnostic logging.

        Returns
        -------
        niimgs : list[nibabel.spatialimages.SpatialImage]
            Successfully loaded beta images.
        trials : list[str]
            Condition name for every loaded beta volume in source order.
        groups : list[int]
            Parsed/fallback run identifier for every loaded beta volume.
        events_df : pandas.DataFrame
            Chronologically sorted session-relative event metadata.

        Raises
        ------
        TypeError
            If ``label_mapper`` is not a non-empty mapping.
        FileNotFoundError
            If no valid condition-specific beta images can be loaded.
        """
        if not isinstance(label_mapper, Mapping) or not label_mapper:
            raise TypeError(
                "For HALFpipe/stGLM input, 'label_mapper' must be a non-empty dictionary"
            )

        niimgs = []
        trials = []
        groups = []
        rows = []
        run_durations = {}
        run_end_times = {}

        for trial_type, label in label_mapper.items():
            condition_files = utils.get_file_from_substring(
                [f"-{trial_type}_stat"],
                beta_files,
            )
            condition_files = cls._as_list(
                condition_files,
                name="condition-file query",
            )

            logger.info(
                "Found %d files for '%s' [model=%s]",
                len(condition_files),
                trial_type,
                model,
            )

            for file_path in condition_files:
                img = cls._load_nifti(file_path)
                if img is None:
                    continue

                n_vols = int(img.shape[3]) if img.ndim == 4 else 1
                run_id = cls._parse_run_id(file_path, groups)

                niimgs.append(img)
                trials.extend([trial_type] * n_vols)
                groups.extend([run_id] * n_vols)

                metadata = cls._read_json_sidecar(file_path)
                if metadata is None:
                    continue

                cls._collect_events(
                    metadata,
                    img=img,
                    json_file=file_path.replace(".nii.gz", ".json"),
                    run_id=run_id,
                    trial_type=trial_type,
                    label=label,
                    rows=rows,
                    run_durations=run_durations,
                    run_end_times=run_end_times,
                )

        if not niimgs:
            raise FileNotFoundError(
                "No valid HALFpipe/stGLM beta images matched the requested label_mapper"
            )

        events_df = cls._build_events_df(rows, run_durations, run_end_times)
        return niimgs, trials, groups, events_df

    @classmethod
    def _load_beta_series(cls, beta_files, subject_betas, model):
        """Load GLMsingle/Bach beta files and their external trial list.

        These sources use a separate ``trial_list`` text file rather than JSON
        event metadata. All readable NIfTI files are retained, then exactly one
        trial-list file is located and read as strings.

        Returns
        -------
        niimgs : list[nibabel.spatialimages.SpatialImage]
            Successfully loaded beta images.
        trials : numpy.ndarray
            Trial labels read from ``trial_list``.
        """
        logger.info("Found %d files for %s model", len(beta_files), model)

        niimgs = []
        for file_path in beta_files:
            img = cls._load_nifti(file_path)
            if img is not None:
                niimgs.append(img)

        if not niimgs:
            raise FileNotFoundError("No valid NIfTI beta images could be loaded")

        trial_file = cls._find_trial_file(subject_betas)
        try:
            return niimgs, np.loadtxt(trial_file, dtype=str, ndmin=1)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Could not read trial list '{trial_file}': {exc}"
            ) from exc

    @classmethod
    def _find_trial_file(cls, subject_betas):
        """Locate the single ``trial_list`` text file for a beta-series source."""
        txt_files = utils.FindFiles(subject_betas, extension=".txt").files
        txt_files = cls._as_list(txt_files, name="text-file query")

        trial_file = utils.get_file_from_substring("trial_list", txt_files)
        if isinstance(trial_file, (list, tuple)):
            if len(trial_file) != 1:
                raise ValueError(
                    f"Expected exactly one trial_list file, found {len(trial_file)}"
                )
            trial_file = trial_file[0]

        if not isinstance(trial_file, str) or not os.path.isfile(trial_file):
            raise FileNotFoundError(
                f"Could not find a valid trial_list.txt in '{subject_betas}'"
            )
        return trial_file

    @classmethod
    def _load_nifti(cls, file_path):
        """Load a NIfTI image, returning ``None`` after recoverable I/O failure."""
        if not os.path.isfile(file_path):
            logger.warning("Beta file does not exist: '%s'", file_path)
            return None

        try:
            return nib.load(file_path)
        except (OSError, nib.filebasedimages.ImageFileError, ValueError) as exc:
            logger.warning("Could not load beta image '%s': %s", file_path, exc)
            return None

    @classmethod
    def _parse_run_id(cls, file_path, groups):
        """Parse ``run-XX`` from a path, with a deterministic fallback ID.

        The fallback preserves the legacy behavior: if no run can be parsed, use
        run 1 before any groups exist, otherwise use one greater than the largest
        run ID already assigned.
        """
        match = cls._RUN_RE.search(file_path)
        if match:
            return int(match.group(1))

        run_id = 1 if not groups else max(groups) + 1
        logger.warning(
            "Could not parse run-XX from '%s'. Using fallback run_id=%d",
            file_path,
            run_id,
        )
        return run_id

    @classmethod
    def _read_json_sidecar(cls, nifti_file):
        """Read a NIfTI JSON sidecar as a case-preserving dictionary.

        Missing, unreadable, malformed, or non-object JSON sidecars are treated as
        recoverable metadata failures and return ``None``. They never cause the
        already loaded beta image to be discarded.
        """
        json_file = nifti_file.replace(".nii.gz", ".json")
        if not os.path.isfile(json_file):
            logger.warning("No JSON file found for '%s'", nifti_file)
            return None

        try:
            with open(json_file, "r", encoding="utf-8") as fjson:
                metadata = json.load(fjson)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read JSON file '%s': %s", json_file, exc)
            return None

        if not isinstance(metadata, dict):
            logger.warning(
                "Expected a JSON object in '%s', got %s",
                json_file,
                type(metadata).__name__,
            )
            return None

        return metadata

    @classmethod
    def _collect_events(
        cls,
        metadata,
        *,
        img,
        json_file,
        run_id,
        trial_type,
        label,
        rows,
        run_durations,
        run_end_times,
    ):
        """Validate and accumulate event metadata from one JSON sidecar.

        Metadata keys and event keys are matched case-insensitively. A preferred
        run duration is calculated from ``NumberOfVolumes * RepetitionTime`` when
        both values are valid. Event endpoints are also recorded and provide a
        fallback duration if metadata-derived run timing is unavailable.

        Invalid individual events are skipped with warnings. Negative event
        durations are retained in the returned event row, matching the source, but
        are clamped to zero only when calculating the fallback run endpoint.

        Parameters
        ----------
        metadata : mapping
            Parsed JSON metadata for one condition/run image.
        img : nibabel.spatialimages.SpatialImage
            Loaded image. Its header TR is used only as a fallback when the JSON
            sidecar does not define ``RepetitionTime``.
        json_file : str
            Sidecar path used in diagnostic messages.
        run_id : int
            Run associated with the image and events.
        trial_type : str
            Condition name assigned to every valid event from this sidecar.
        label : Any
            Output label associated with ``trial_type``.
        rows : list[dict]
            Mutable accumulator receiving validated event rows.
        run_durations : dict[int, float]
            Mutable mapping receiving preferred metadata-derived run durations.
        run_end_times : dict[int, float]
            Mutable mapping receiving maximum valid event endpoints per run.
        """
        metadata_lower = {
            str(key).casefold(): value
            for key, value in metadata.items()
        }

        events = metadata_lower.get("events")
        number_of_volumes = cls._coerce_positive_int(
            metadata_lower.get("numberofvolumes"),
            name="NumberOfVolumes",
            source=json_file,
        )
        tr = cls._coerce_positive_float(
            metadata_lower.get(
                "repetitiontime",
                float(img.header["pixdim"][4]),
            ),
            name="RepetitionTime/TR",
            source=json_file,
        )

        if number_of_volumes is not None and tr is not None:
            cls._register_run_duration(
                run_durations,
                run_id,
                number_of_volumes * tr,
            )

        if not isinstance(events, list):
            logger.warning("No valid 'Events' list found in '%s'", json_file)
            return

        valid_event_count = 0
        for event_index, event in enumerate(events):
            parsed = cls._parse_event(
                event,
                event_index=event_index,
                json_file=json_file,
            )
            if parsed is None:
                continue

            onset, duration = parsed
            rows.append(
                {
                    "onset": onset,
                    "duration": duration,
                    "run_id": run_id,
                    "trial_type": trial_type,
                    "label": label,
                }
            )

            event_end = onset + max(duration, 0.0)
            run_end_times[run_id] = max(
                run_end_times.get(run_id, 0.0),
                event_end,
            )
            valid_event_count += 1

        if valid_event_count == 0:
            logger.warning("No valid events found in '%s'", json_file)

    @classmethod
    def _parse_event(cls, event, *, event_index, json_file):
        """Parse and validate one JSON event entry.

        Returns ``(onset, duration)`` for a valid event and ``None`` for an entry
        that should be skipped. Onset and duration must be finite numeric values;
        onset must additionally be non-negative.
        """
        if not isinstance(event, dict):
            logger.warning(
                "Skipping event %d in '%s': event is not a JSON object",
                event_index,
                json_file,
            )
            return None

        event_lower = {
            str(key).casefold(): value
            for key, value in event.items()
        }
        onset = event_lower.get("onset")
        duration = event_lower.get("duration", 0.0)

        try:
            onset = float(onset)
            duration = float(duration)
        except (TypeError, ValueError):
            logger.warning(
                "Skipping event %d in '%s': invalid onset=%r or duration=%r",
                event_index,
                json_file,
                onset,
                duration,
            )
            return None

        if not np.isfinite(onset) or not np.isfinite(duration):
            logger.warning(
                "Skipping event %d in '%s': onset or duration is not finite",
                event_index,
                json_file,
            )
            return None

        if onset < 0:
            logger.warning(
                "Skipping event %d in '%s': onset is negative (%g)",
                event_index,
                json_file,
                onset,
            )
            return None

        if duration < 0:
            logger.warning(
                "Event %d in '%s' has negative duration %g; "
                "using 0 for run-end estimation",
                event_index,
                json_file,
                duration,
            )

        return onset, duration

    @classmethod
    def _coerce_positive_float(cls, value, *, name, source):
        """Convert a metadata value to a finite positive float or return ``None``."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            logger.warning("Invalid %s in '%s': %r", name, source, value)
            return None

        if not np.isfinite(value) or value <= 0:
            logger.warning("Invalid %s in '%s': %r", name, source, value)
            return None
        return value

    @classmethod
    def _coerce_positive_int(cls, value, *, name, source):
        """Convert a metadata value to a positive integer or return ``None``."""
        try:
            value = int(value)
        except (TypeError, ValueError):
            logger.warning("Invalid or missing %s in '%s': %r", name, source, value)
            return None

        if value <= 0:
            logger.warning("Invalid or missing %s in '%s': %r", name, source, value)
            return None
        return value

    @classmethod
    def _register_run_duration(cls, run_durations, run_id, duration):
        """Register a metadata-derived duration while detecting conflicts.

        The first valid duration observed for a run is retained. Later conflicting
        condition files produce a warning but do not overwrite it, preserving the
        behavior of the original implementation.
        """
        existing_duration = run_durations.get(run_id)
        if existing_duration is None:
            run_durations[run_id] = float(duration)
            return

        if not np.isclose(existing_duration, duration):
            logger.warning(
                "Conflicting metadata-derived durations for run %s: %g vs %g seconds. "
                "Keeping the first valid value.",
                run_id,
                existing_duration,
                duration,
            )

    @classmethod
    def _build_events_df(cls, rows, run_durations, run_end_times):
        """Build a chronologically sorted, session-relative event table.

        Event onsets stored in sidecars are assumed to be relative to their own run.
        This method computes a cumulative offset for each run, adds that offset to
        every event onset, and performs a stable sort by onset and trial type.

        Preferred run duration comes from ``NumberOfVolumes * RepetitionTime``.
        When that estimate is unavailable or invalid, the maximum observed event
        endpoint for the run is used as a fallback. If neither source provides a
        valid positive duration, session-relative timing cannot be established and
        a ``ValueError`` is raised.

        Parameters
        ----------
        rows : sequence of mapping
            Validated event records with run-relative onsets.
        run_durations : mapping[int, float]
            Preferred metadata-derived run durations.
        run_end_times : mapping[int, float]
            Fallback maximum event endpoints for each run.

        Returns
        -------
        pandas.DataFrame
            DataFrame with ``_EVENT_COLUMNS``. Empty input returns an empty table;
            otherwise onsets are session-relative and rows are chronological.
        """
        events_df = pd.DataFrame(rows, columns=cls._EVENT_COLUMNS)

        if events_df.empty:
            logger.warning(
                "No valid event metadata found; returning an empty events DataFrame; "
                "chronological order cannot be restored."
            )
            return events_df

        run_offsets = {}
        elapsed_time = 0.0
        run_ids = sorted(events_df["run_id"].dropna().unique())

        for run_id in run_ids:
            run_offsets[run_id] = elapsed_time
            run_duration = run_durations.get(run_id)

            if (
                run_duration is None
                or not np.isfinite(run_duration)
                or run_duration <= 0
            ):
                run_duration = run_end_times.get(run_id)
                if (
                    run_duration is not None
                    and np.isfinite(run_duration)
                    and run_duration > 0
                ):
                    logger.warning(
                        "NumberOfVolumes × TR was unavailable for run %s; "
                        "using the final event endpoint (%g s) as a fallback run duration",
                        run_id,
                        run_duration,
                    )
                else:
                    raise ValueError(
                        f"Could not determine a valid duration for run {run_id}. "
                        "Neither NumberOfVolumes × TR nor a valid event endpoint was available."
                    )

            elapsed_time += float(run_duration)

        offset_values = events_df["run_id"].map(run_offsets)
        if offset_values.isna().any():
            missing_runs = sorted(
                events_df.loc[offset_values.isna(), "run_id"].unique()
            )
            raise ValueError(
                f"Could not calculate onset offsets for runs {missing_runs}"
            )

        events_df["onset"] = events_df["onset"] + offset_values.astype(float)
        return (
            events_df
            .sort_values(["onset", "trial_type"], kind="stable")
            .reset_index(drop=True)
        )

    @classmethod
    def _empty_events_df(cls):
        """Return an empty event table with the canonical column schema."""
        return pd.DataFrame(columns=cls._EVENT_COLUMNS)

    @classmethod
    def _concat_betas(cls, niimgs):
        """Concatenate loaded beta images into one 4D NIfTI image.

        Any spatial incompatibility or other concatenation error is re-raised as a
        ``ValueError`` with a consistent public-facing message.
        """
        logger.info("Concatenating %d files into a single 4D object", len(niimgs))
        try:
            return image.concat_imgs(niimgs)
        except Exception as exc:
            raise ValueError(f"Could not concatenate beta images: {exc}") from exc

    @classmethod
    def _validate_beta_metadata(cls, beta_imgs, trials, groups=None):
        """Ensure one trial label and, when present, one group per beta volume."""
        n_beta_volumes = int(beta_imgs.shape[-1])

        if n_beta_volumes != len(trials):
            raise ValueError(
                f"Number of beta images ({n_beta_volumes}) does not match "
                f"the number of trial labels ({len(trials)})"
            )

        if groups is not None and n_beta_volumes != len(groups):
            raise ValueError(
                f"Number of beta images ({n_beta_volumes}) does not match "
                f"the number of run-group entries ({len(groups)})"
            )

    @classmethod
    def _save_merged_betas(
        cls,
        beta_imgs,
        *,
        subject,
        model,
        src,
        subject_betas,
        output_dir=None,
    ):
        """Save a merged beta image using the class's canonical filename scheme.

        Parameters
        ----------
        beta_imgs : nibabel.spatialimages.SpatialImage
            Final merged image to write.
        subject : str
            Subject identifier embedded in the output filename.
        model : str
            Model identifier embedded in the output filename.
        src : str
            Source identifier embedded in the output filename.
        subject_betas : str or os.PathLike
            Default destination when ``output_dir`` is omitted.
        output_dir : str or os.PathLike or None, default=None
            Explicit output directory. Created recursively when necessary.

        Returns
        -------
        str
            Path of the saved NIfTI image.
        """
        if output_dir is None:
            output_dir = subject_betas

        os.makedirs(output_dir, exist_ok=True)
        fname = opj(
            output_dir,
            f"{subject}_model-{model}_source-{src}_desc-merged_betas.nii.gz",
        )
        logger.info("Saving merged beta image as '%s'", fname)
        beta_imgs.to_filename(fname)
        return fname
