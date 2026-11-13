# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Analysis plugins for PANIC ROI and searchlight analyses.

PANIC analysis plugins are organized as individual modules under the
``panic.plugins`` package. Each plugin implements a small analysis function
that receives a feature matrix and returns a scalar score (optionally with
artifacts). The same plugin interface is used for both ROI analyses and
searchlight centres.

Built-in plugins are implemented in separate modules, for example::

    panic.plugins.decoding
    panic.plugins.rsa
    panic.plugins.similarity
    panic.plugins.dimensionality

The default ``decoding`` plugin wraps
``panic.pipeline._cv_mean_score``; other plugins can compute
representational metrics such as dimensionality, RSA, or CS-to-US similarity.

Expected plugin signature
-------------------------
Plugins should accept the following core arguments::

    def my_plugin(
        X_path,
        labels,
        cfg,
        *,
        folds=None,
        cols=None,
        groups=None,
        permute=False,
        rng=None,
        return_artifacts=False,
        **kwargs,
    ):
        ...

Parameters are intentionally broad so the same callable works in ROI and
searchlight contexts:

- ``X_path`` is either a path to a joblib-dumped matrix or an array-like
  object. The matrix is expected to have shape
  ``(n_samples, n_features)``.
- ``labels`` has shape ``(n_samples,)``.
- ``cols`` is an optional feature subset, mainly used by searchlight.
  Plugins should evaluate ``X[:, cols]`` when ``cols`` is provided.
- ``groups`` contains optional grouping variables (e.g. runs or subjects)
  for grouped analyses.
- ``permute`` indicates whether the plugin should compute a null/permuted
  score. For label-dependent analyses, use ``rng`` to shuffle labels. For
  analyses where permutation is not meaningful, set
  ``analysis.permutations: false`` in the YAML config.
- ``return_artifacts`` should usually be ``True`` only for observed
  ROI-level analyses. Avoid returning large artifacts for permutations or
  searchlight analyses.

Return value
------------
A plugin may return either::

    score

or::

    score, artifacts

where ``score`` is a float-like scalar and ``artifacts`` is a dictionary of
numpy-saveable arrays. Arrays with shape ``(n_features,)`` can later be
mapped back to brain space using ``roi_linidx``.

Plugin organization
-------------------
Plugins are implemented in dedicated modules under
``panic.plugins``. Shared infrastructure such as plugin registration,
artifact handling, plugin lookup, and label resolution lives in
``panic.plugins.core``.

Current package structure::

    panic/plugins/
    ├── __init__.py
    ├── core.py
    ├── decoding.py
    ├── rsa.py
    ├── similarity.py
    └── dimensionality.py

To add a custom plugin:

1. Create a new module under ``panic/plugins/``.
2. Implement a plugin function using the signature above.
3. Load/materialize the feature matrix with
   ``panic.utils.load_feature_matrix``.
4. Return either a scalar score or ``(score, artifacts)``.
5. Import and register the plugin in ``panic.plugins.core``.

For example::

    # panic/plugins/my_plugin.py

    def my_plugin(...):
        ...
        return score

    # panic/plugins/core.py

    from .my_plugin import my_plugin

    PLUGIN_REGISTRY["my_plugin"] = my_plugin

Selecting a plugin
------------------
Select the desired plugin in the analysis configuration::

    decoding_settings:
      analysis:
        name: my_plugin
        permutations: false
        higher_is_better: true
        args:
          my_argument: 123

The values under ``analysis.args`` are passed directly to the plugin as
keyword arguments by ``get_analysis_plugin``.

Label resolution
----------------
When ``label_dict`` is provided, plugin arguments matching condition names
are automatically converted to their corresponding numeric labels by
``resolve_label_references``.

For example::

    analysis:
      name: cs_us_similarity
      args:
        cs_label: CSpu
        us_label: USp

may be resolved internally to::

    {
        "cs_label": 1,
        "us_label": 0,
    }

depending on the contents of ``label_dict``.
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

from .decoding import (
    decoding_plugin,
    vanilla_nearest_centroid_plugin,
    n_region_nearest_centroid_plugin,
    fixed_n_region_generative_plugin,
)
from .rsa import rsa_plugin
from .similarity import cosine_similarity_plugin
from .dimensionality import dimensionality_plugin


opj = os.path.join

ArrayLike = Union[np.ndarray, Any]
PathLike = Union[str, os.PathLike[str], Path]
ArtifactDict = Dict[str, np.ndarray]
PluginReturn = Union[float, Tuple[float, ArtifactDict]]
PluginCallable = Callable[..., PluginReturn]


def get_analysis_plugin(
        cfg: Mapping[str, Any],
        label_dict: Optional[Mapping[str, int]] = None,
    ) -> Tuple[PluginCallable, Dict[str, Any]]:
    """
    Resolve the configured analysis plugin and plugin-specific arguments.

    Parameters
    ----------
    cfg : mapping
        Decoding settings dictionary. The ``analysis`` section defines the
        analysis plugin and optional analysis-specific arguments. For example::

            analysis:
            type: scikit_decoding
            args: {}

        ``type`` identifies the analysis plugin. Exact names registered in
        ``PLUGIN_REGISTRY`` are supported, as are ``scikit_*`` analysis types,
        which are dispatched to the generic scikit-learn decoding plugin. For
        example::

            analysis:
            type: scikit_svm
            args: {}

        An optional ``name`` may be provided to assign a descriptive identifier
        to the analysis::

            analysis:
            type: scikit_svm
            name: SVM_CSm_v_CSpu
            args: {}

        ``name`` does not determine which plugin is executed. It is used to
        identify the analysis in output paths and result metadata, allowing the
        same analysis name to be used with different plugin types if desired.

        For example::

            analysis:
            type: vanilla_nearest_centroid
            name: SVM_CSm_v_CSpu
            args: {}

        and::

            analysis:
            type: scikit_svm
            name: SVM_CSm_v_CSpu
            args: {}

        represent distinct analysis types while sharing the same descriptive
        analysis name.     

    label_dict : mapping, optional
        Mapping from condition names to integer labels. When provided,
        any plugin argument whose value matches a key in ``label_dict``
        is automatically replaced by the corresponding integer label.

        Example
        -------
        >>> label_dict = {"USp": 0, "CSpu": 1}

        >>> analysis:
        ...   name: cs_us_similarity
        ...   args:
        ...     cs_label: CSpu
        ...     us_label: USp

        becomes::

            {"cs_label": 1, "us_label": 0}

    Returns
    -------
    plugin : callable
        Plugin function from ``PLUGIN_REGISTRY``.
    plugin_kwargs : dict
        Keyword arguments from ``analysis.args`` after label resolution.

    Raises
    ------
    KeyError
        If the requested plugin name is not registered.
    """
    analysis = cfg.get(
        "analysis",
        {
            "type": "scikit_decoding",
            "args": {},
        },
    )

    name = analysis.get("type", "scikit_decoding")

    # Exact match first.
    if name in PLUGIN_REGISTRY:
        plugin = PLUGIN_REGISTRY[name]

    # Prefix-based fallback for scikit-learn analyses.
    elif name.startswith("scikit_"):
        plugin = decoding_plugin

    else:
        available = ", ".join(sorted(PLUGIN_REGISTRY))
        raise KeyError(
            f"Unknown analysis plugin {name!r}. "
            f"Available plugins: {available}, scikit_*"
        )

    plugin_kwargs = analysis.get("args", {}).copy()

    if label_dict is not None:
        plugin_kwargs = resolve_label_references(
            plugin_kwargs,
            label_dict,
        )

    return plugin, plugin_kwargs


def resolve_label_references(plugin_kwargs, label_dict):
    out = plugin_kwargs.copy()

    def resolve(v):
        if isinstance(v, str) and v in label_dict:
            return label_dict[v]
        if isinstance(v, list):
            return [resolve(x) for x in v]
        if isinstance(v, tuple):
            return tuple(resolve(x) for x in v)
        if isinstance(v, dict):
            return {k: resolve(x) for k, x in v.items()}
        return v

    return {k: resolve(v) for k, v in out.items()}


def unpack_plugin_result(result: PluginReturn) -> Tuple[float, ArtifactDict]:
    """
    Normalize plugin output to ``(score, artifacts)``.

    Parameters
    ----------
    result : float or tuple
        Return value from a plugin. Valid forms are ``score`` or
        ``(score, artifacts)``.

    Returns
    -------
    score : float
        Scalar plugin score.
    artifacts : dict
        Artifact dictionary. Empty when the plugin returned only a scalar.
    """
    if isinstance(result, tuple) and len(result) == 2:
        score, artifacts = result
        return float(score), dict(artifacts)

    return float(result), {}


def save_analysis_artifacts(
        output_dir: PathLike,
        artifacts: Mapping[str, Any],
        roi_linidx: Optional[np.ndarray] = None,
    ) -> None:
    """
    Save plugin artifacts and ROI feature-to-voxel mapping.

    Parameters
    ----------
    output_dir : str or path-like
        Directory where artifacts should be written.
    artifacts : mapping
        Dictionary of numpy-saveable artifacts returned by a plugin.
    roi_linidx : numpy.ndarray, optional
        Linear voxel indices mapping feature columns back to the source image or
        ROI mask. This is essential for mapping feature-wise arrays back to
        brain space.

    Returns
    -------
    None
        Files are written to ``output_dir``.
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, val in artifacts.items():
        if val is None:
            continue
        np.save(opj(output_dir, f"{key}.npy"), np.asarray(val))

    if isinstance(roi_linidx, np.ndarray):
        np.save(opj(output_dir, "roi_linidx.npy"), roi_linidx)


PLUGIN_REGISTRY: Dict[str, PluginCallable] = {
    "rsa": rsa_plugin,
    "scikit_decoding": decoding_plugin,
    "dimensionality": dimensionality_plugin,
    "cs_us_similarity": cosine_similarity_plugin,
    "vanilla_nearest_centroid": vanilla_nearest_centroid_plugin,
    "n_region_nearest_centroid": n_region_nearest_centroid_plugin,
    "fixed_n_region_generative": fixed_n_region_generative_plugin,
}
