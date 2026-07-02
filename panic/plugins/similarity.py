# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from panic.utils import load_feature_matrix


ArrayLike = Union[np.ndarray, Any]
PathLike = Union[str, os.PathLike[str], Path]
ArtifactDict = Dict[str, np.ndarray]
PluginReturn = Union[float, Tuple[float, ArtifactDict]]

def cosine_similarity_plugin(
        X_path: Union[PathLike, ArrayLike],
        labels: ArrayLike,
        cfg: Mapping[str, Any],
        *,
        cols: Optional[np.ndarray] = None,
        cs_label: Optional[Any] = None,
        us_label: Optional[Any] = None,
        return_artifacts: bool = False,
        **kwargs: Any,
    ) -> PluginReturn:
    """
    Compute mean trialwise CS-to-US cosine similarity.

    The US template is the mean pattern across all samples with label
    ``us_label``. The score is the mean cosine similarity between each CS sample
    and that US template.

    Parameters
    ----------
    X_path : str, path-like, or array-like
        Feature matrix or joblib path with shape ``(n_samples, n_features)``.
    labels : array-like
        Labels aligned to the rows of ``X``.
    cfg : mapping
        Analysis configuration. Included for interface consistency; not used
        directly by this plugin.
    cols : array-like of int, optional
        Optional feature subset for searchlight evaluation.
    cs_label : object
        Label identifying CS samples.
    us_label : object
        Label identifying US samples.
    return_artifacts : bool, default=False
        If True, also return feature-wise arrays that can be mapped back to ROI
        space: ``us_template``, ``cs_mean``, ``cs_us_diff``, and per-CS-trial
        similarities.
    **kwargs
        Ignored extra keyword arguments accepted for plugin compatibility.

    Returns
    -------
    float or tuple
        Mean CS-to-US cosine similarity. If ``return_artifacts=True``, returns
        ``(score, artifacts)``.
    """
    X = load_feature_matrix(X_path, cols=cols)
    y = np.asarray(labels)

    CS = X[y == cs_label]
    US = X[y == us_label]

    if CS.size == 0 or US.size == 0:
        return (float("nan"), {}) if return_artifacts else float("nan")

    us_template = US.mean(axis=0)
    cs_mean = CS.mean(axis=0)
    diff = cs_mean - us_template

    denom_us = np.linalg.norm(us_template)
    sims = []
    for x in CS:
        denom = np.linalg.norm(x) * denom_us
        sims.append(np.dot(x, us_template) / denom if denom > 0 else np.nan)

    score = float(np.nanmean(sims))

    if not return_artifacts:
        return score

    artifacts: ArtifactDict = {
        "us_template": np.asarray(us_template),
        "cs_mean": np.asarray(cs_mean),
        "cs_us_diff": np.asarray(diff),
        "cs_us_similarity": np.asarray(sims),
    }
    return score, artifacts
