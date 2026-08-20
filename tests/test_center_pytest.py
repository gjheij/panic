# VERIFIED MASK-REPLAY VERSION: PANIC_TEST_MASK REQUIRED/DEFAULTED
# -*- coding: utf-8 -*-
"""
Targeted pytest regression/debugger for the searchlight stall observed at:

    center index = 58116
    permutation  = 9

This deliberately does NOT run the full searchlight.  It initializes the real
sub-017 data/configuration through ClassifySubject, reconstructs the exact
searchlight center list and RNG stream, materializes only center 58116, and then
runs the observed score followed by permutations 0..9 serially.

Run with:

    pytest -s -vv tests/test_center.py

Useful optional environment variables:

    PANIC_TEST_CENTER_IX=58116
    PANIC_TEST_PERM_IX=9
    PANIC_TEST_GLOBAL_SEED=0
    PANIC_TEST_MASK=/absolute/path/to/exact/searchlight/mask.nii.gz
    PANIC_TEST_EXPECTED_N_CENTERS=221989

PANIC_TEST_MASK should be the exact mask used by the stalled production run.
The test refuses to replay CENTER_IX unless the reconstructed center count
matches PANIC_TEST_EXPECTED_N_CENTERS, because index 58116 is meaningful only
within the identical center enumeration.

The faulthandler timer dumps Python stacks every 30 seconds if a plugin call
wedges.  This is intentional: the test is a debugger/reproducer rather than a
fast unit test.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import time

import nibabel as nib
import numpy as np
import pytest
from joblib import dump

from panic import data
from panic.decode import ClassifySubject
from panic.pipeline import create_outer_folds
from panic.plugins import core
from panic.searchlight import _cols_for_center, _neighbors_ball_mm
from panic.utils import get_config_path, load_feature_matrix


CENTER_IX = int(os.environ.get("PANIC_TEST_CENTER_IX", "58116"))
TARGET_PERM_IX = int(os.environ.get("PANIC_TEST_PERM_IX", "9"))
GLOBAL_SEED = int(os.environ.get("PANIC_TEST_GLOBAL_SEED", "0"))
EXPECTED_N_CENTERS = int(os.environ.get("PANIC_TEST_EXPECTED_N_CENTERS", "221989"))
DEFAULT_MASK = "/mnt/d/fMRI/ENIGMA_FC_SingleTrials/site_data/acquisition/site-Bonn_pi-Bach_sample-01_desc-HRA/derivatives/rois/sub-017/searchlight/sub-017_roi-brain_desc-orig.nii.gz"


def _select_searchlight_mask(decoder: ClassifySubject):
    """Load the exact searchlight mask used by the stalled production run."""
    mask_path = os.environ.get("PANIC_TEST_MASK", DEFAULT_MASK)
    print(f"[replay] PANIC_TEST_MASK={mask_path}", flush=True)

    if not os.path.isfile(mask_path):
        pytest.fail(
            "Exact searchlight mask not found. Set PANIC_TEST_MASK to the mask "
            f"used by the stalled run. Tried: {mask_path}"
        )

    return {
        "roi_key": "explicit-searchlight-mask",
        "roi_label": os.path.basename(mask_path),
        "hemi_key": "uni",
        "mask_img": nib.load(mask_path),
        "mask_source": mask_path,
    }


def _run_plugin(
    *,
    plugin,
    plugin_kwargs,
    X_center,
    labels,
    folds,
    cfg,
    groups,
    locked_params,
    standardize,
    permute,
    rng=None,
):
    """
    Call the same plugin shape used by searchlight._one_center().

    Searchlight mode disables ROI-level feature selection/grid search and passes
    the locked parameters plus the subject's standardization setting.
    """
    result = plugin(
        X_center,
        labels,
        cfg=cfg,
        folds=folds,
        cols=None,
        groups=groups,
        permute=permute,
        rng=rng,
        return_artifacts=False,
        locked=locked_params,
        searchlight=True,
        standardize=standardize,
        # Avoid writing fold/debug artifacts into the real analysis directory.
        save_dir=None,
        **plugin_kwargs,
    )
    score, _ = core.unpack_plugin_result(result)
    return float(score)


@pytest.mark.integration
@pytest.mark.slow
def test_searchlight_center_58116_perm9(tmp_path):
    """
    Reproduce the exact center/permutation that became the sole outstanding task.

    Expected diagnostic interpretation
    ----------------------------------
    PASS quickly:
        center/permutation is not intrinsically pathological in a fresh process;
        long-lived worker state becomes the leading hypothesis.

    HANG at a printed BEFORE_PERM line:
        the corresponding plugin invocation is reproducibly pathological even
        in a fresh serial process.

    ERROR:
        pytest traceback identifies the failing Python layer directly.
    """
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(
        30,
        repeat=True,
        file=sys.stderr,
    )

    try:
        cfg_file = get_config_path()
        subject = "sub-017"

        print(f"\n[replay] config={cfg_file}", flush=True)
        print(f"[replay] subject={subject}", flush=True)

        # Match the repository's existing searchlight integration-test setup,
        # but initialize only the betas.  Do NOT call decoder._fit(), because
        # that would launch the full searchlight.
        decoder = ClassifySubject(
            subject,
            cfg_file,
            save_imgs=False,
            searchlight=True,
        )
        decoder._init_betas()

        selected = _select_searchlight_mask(decoder)
        mask_img = selected["mask_img"]

        print(
            "[replay] selected "
            f"roi_key={selected['roi_key']!r} "
            f"roi_label={selected['roi_label']!r} "
            f"hemi_key={selected['hemi_key']!r} "
            f"mask_source={selected['mask_source']!r}",
            flush=True,
        )

        cfg = decoder.dec_settings_searchlight
        sl_cfg = cfg.get("searchlight", {})
        radius_mm = float(sl_cfg.get("radius_mm", 6))
        locked_params = sl_cfg.get("locked")
        n_perms = int(cfg.get("n_permutations", 1000))

        if TARGET_PERM_IX < 0 or TARGET_PERM_IX >= n_perms:
            pytest.fail(
                f"TARGET_PERM_IX={TARGET_PERM_IX} is outside configured "
                f"n_permutations={n_perms}"
            )

        # Match permutation_searchlight's handling of groups.
        groups = getattr(decoder, "groups", None)
        if not cfg.get("permute_within_groups", False):
            groups = None

        plugin, plugin_kwargs = core.get_analysis_plugin(
            cfg,
            label_dict=decoder.cfg["label_dict"],
        )

        # This mirrors permutation_searchlight's feature construction.
        mf = data.MaskAndFilterBetas(
            decoder.betas,
            mask_img,
            trial_list=decoder.trial_list,
            label_mapper=decoder.cfg["label_dict"],
            output_file=None,
            zooms=sl_cfg.get("target_zooms"),
        )

        X = mf.X.astype("float32", copy=False)
        y = np.asarray(mf.labels)

        vol_shape = mf.mask_resampled_to_betas.shape[:3]
        col_index_vol = np.full(vol_shape, -1, np.int32)
        coords = np.unravel_index(
            np.asarray(mf.roi_linidx, dtype=np.int64),
            vol_shape,
            order="C",
        )
        col_index_vol[coords] = np.arange(
            len(mf.roi_linidx),
            dtype=np.int32,
        )

        centers = np.column_stack(
            np.where(col_index_vol >= 0)
        )

        print(
            f"[replay] X={X.shape} centers={len(centers)} "
            f"radius_mm={radius_mm} n_perms={n_perms}",
            flush=True,
        )

        if len(centers) != EXPECTED_N_CENTERS:
            pytest.fail(
                "Replay domain does not match the stalled production run: "
                f"expected {EXPECTED_N_CENTERS} centers, reconstructed {len(centers)}. "
                f"CENTER_IX={CENTER_IX} cannot be compared safely until the exact "
                "searchlight mask/domain is reproduced."
            )

        if CENTER_IX < 0 or CENTER_IX >= len(centers):
            pytest.fail(
                f"CENTER_IX={CENTER_IX} is outside this mask's center range "
                f"[0, {len(centers) - 1}]. "
                "If the stalled run used another ROI/hemi, set "
                "PANIC_TEST_ROI_KEY / PANIC_TEST_HEMI_KEY."
            )

        folds = create_outer_folds(
            cfg,
            y,
            groups=groups,
        )

        zooms = mf.mask_resampled_to_betas.header.get_zooms()
        offsets = _neighbors_ball_mm(
            zooms,
            radius_mm,
        )

        center_ijk = tuple(
            map(int, centers[CENTER_IX])
        )
        cols = _cols_for_center(
            center_ijk,
            offsets,
            col_index_vol,
            vol_shape,
        )

        if len(cols) < 2:
            pytest.fail(
                f"center {CENTER_IX} {center_ijk} has only {len(cols)} features"
            )

        # Reconstruct exactly the RNG hierarchy used by permutation_searchlight.
        center_rng = np.random.default_rng(GLOBAL_SEED)
        center_seeds = center_rng.integers(
            0,
            2**32 - 1,
            size=len(centers),
            dtype=np.uint32,
        )
        center_seed = int(center_seeds[CENTER_IX])

        perm_rng = np.random.default_rng(center_seed)
        perm_seeds = perm_rng.integers(
            0,
            2**32 - 1,
            size=n_perms,
            dtype=np.uint32,
        )

        print(
            f"[replay] CENTER ix={CENTER_IX} ijk={center_ijk} "
            f"ncols={len(cols)} center_seed={center_seed}",
            flush=True,
        )
        print(
            f"[replay] permutation seeds 0..{TARGET_PERM_IX}: "
            f"{[int(x) for x in perm_seeds[:TARGET_PERM_IX + 1]]}",
            flush=True,
        )
        print(
            f"[replay] TARGET perm_ix={TARGET_PERM_IX} "
            f"perm_seed={int(perm_seeds[TARGET_PERM_IX])}",
            flush=True,
        )

        # Mirror the real searchlight's dump + load_feature_matrix path rather
        # than slicing X directly.  This preserves one more relevant layer.
        X_path = dump(
            X,
            tmp_path / "Xsl_replay.joblib",
            compress=0,
        )[0]

        print("[replay] BEFORE_LOAD", flush=True)
        t0 = time.perf_counter()
        X_center = load_feature_matrix(
            X_path,
            cols=cols,
        )
        print(
            f"[replay] AFTER_LOAD shape={X_center.shape} "
            f"elapsed={time.perf_counter() - t0:.6f}s",
            flush=True,
        )

        print("[replay] BEFORE_OBS", flush=True)
        t0 = time.perf_counter()
        obs = _run_plugin(
            plugin=plugin,
            plugin_kwargs=plugin_kwargs,
            X_center=X_center,
            labels=y,
            folds=folds,
            cfg=cfg,
            groups=groups,
            locked_params=locked_params,
            standardize=decoder.do_standardization,
            permute=False,
            rng=None,
        )
        print(
            f"[replay] AFTER_OBS score={obs:.8f} "
            f"elapsed={time.perf_counter() - t0:.6f}s",
            flush=True,
        )

        # Run 0..TARGET sequentially, rather than just permutation 9.  The
        # original worker reached permutation 9 only after these preceding
        # plugin calls, so this is the more faithful reproducer.
        scores = []
        for perm_ix in range(TARGET_PERM_IX + 1):
            perm_seed = int(perm_seeds[perm_ix])

            print(
                f"[replay] BEFORE_PERM_{perm_ix} perm_seed={perm_seed}",
                flush=True,
            )

            t0 = time.perf_counter()
            score = _run_plugin(
                plugin=plugin,
                plugin_kwargs=plugin_kwargs,
                X_center=X_center,
                labels=y,
                folds=folds,
                cfg=cfg,
                groups=groups,
                locked_params=locked_params,
                standardize=decoder.do_standardization,
                permute=True,
                rng=np.random.default_rng(perm_seed),
            )
            elapsed = time.perf_counter() - t0

            scores.append(score)
            print(
                f"[replay] AFTER_PERM_{perm_ix} "
                f"perm_seed={perm_seed} score={score:.8f} "
                f"elapsed={elapsed:.6f}s",
                flush=True,
            )

        assert len(scores) == TARGET_PERM_IX + 1
        assert np.all(np.isfinite(scores)), scores

        print(
            f"[replay] PASS center={CENTER_IX} "
            f"through perm={TARGET_PERM_IX}",
            flush=True,
        )

    finally:
        faulthandler.cancel_dump_traceback_later()
