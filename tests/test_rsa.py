
import numpy as np
import pytest
from copy import deepcopy

from panic.decode import ClassifySubject
from panic.utils import get_config_path, load_yaml, dump_yaml


PROJECT_DIR = (
    "/lustre/scratch/data/gheij_hpc-fmriprep/"
    "projects/CogLearn7T"
)

BETA_DIR = f"{PROJECT_DIR}/derivatives/glmsingle/impulse"

CONDITIONS = [
    "US_0%",
    "US_100%_1",
    "US_50%_1_pos",
    "US_50%_1_neg",
    "US_50%_2_pos",
    "US_50%_2_neg",
]

LABEL_DICT = {
    condition: i
    for i, condition in enumerate(CONDITIONS)
}

# Order must match CONDITIONS.
DELIVERED = np.array([0, 1, 1, 0, 2, 0], dtype=float)
EXPECTED = np.array([0, 1, 0.5, 0.5, 1, 1], dtype=float)
SIGNED_PE = DELIVERED - EXPECTED


def make_model_rdm(values):
    """Absolute pairwise differences between condition values."""
    values = np.asarray(values, dtype=float)
    return np.abs(values[:, None] - values[None, :]).tolist()


def update_nested_dict(base, updates):
    """Recursively update nested configuration values."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            update_nested_dict(base[key], value)
        else:
            base[key] = value
    return base


def make_test_config(tmp_path, settings):
    cfg = deepcopy(load_yaml(get_config_path()))
    update_nested_dict(cfg, settings)

    # Replace dictionaries rather than inheriting entries
    # from the default configuration.
    cfg["label_dict"] = settings["label_dict"]

    for key in ("analysis", "outer_cv", "inner_cv"):
        cfg["decoding_settings"][key] = (
            settings["decoding_settings"][key]
        )

    cfg_file = tmp_path / "panic_rsa_config.yml"
    dump_yaml(cfg, cfg_file)
    return str(cfg_file)


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)
def test_panic_rsa_crossnobis(tmp_path):

    settings = {
        "label_dict": LABEL_DICT,

        "general_settings": {
            "project_dir": PROJECT_DIR,
            "beta_dir": BETA_DIR,
            "source": "stglm",
            "method": "GLMsingleTYPED",
            "standardize": None,
            "save_dir": str(tmp_path / "rsa_results"),
            "overwrite": True,
            "filters": [
                "task-aversivechoice_",
                "space-T1w_",
                "desc-masked_",
            ],
        },

        "residual_settings": {
            "path_pattern": (
                "{beta_dir}/{subject}/"
                "task-aversivechoice_space-T1w_desc-masked/"
                "residuals/model-typed/"
                "{subject}_task-aversivechoice_"
                "run-{run:02d}_space-T1w_"
                "model-typed_desc-fullnuisance_residuals.nii.gz"
            ),
            "center_per_run": True,
            "run_numbers": [1, 2, 3, 4],
        },

        "decoding_settings": {
            "n_permutations": 0,
            "permute_within_groups": True,

            "parallel": {
                "n_jobs": 1,
            },

            "analysis": {
                "type": "rsa",
                "name": "RSA_signed_PE_crossnobis",
                "permutations": False,
                "higher_is_better": True,
                "args": {
                    "metric": "crossnobis",
                    "compare": "spearman",
                    "noise_normalization": "shrinkage",
                    "min_partitions": 2,
                    "normalize_by_features": True,
                    "min_rdm_pairs": 3,

                    # Encoded condition IDs, matching label_dict.
                    "conditions": list(range(6)),

                    "target_model": "signed_pe",

                    "model_rdms": {
                        "signed_pe": make_model_rdm(SIGNED_PE),
                        "delivered_magnitude": make_model_rdm(
                            DELIVERED
                        ),
                        "expected_magnitude": make_model_rdm(
                            EXPECTED
                        ),
                    },
                },
            },

            "outer_cv": {
                "mode": "sklearn",
                "name": "LeaveOneGroupOut",
                "args": {},
            },

            "inner_cv": {
                "name": "LeaveOneGroupOut",
                "args": {},
            },
        },
    }

    cfg_file = make_test_config(tmp_path, settings)

    decoder = ClassifySubject(
        "sub-01",
        cfg_file,
        save_imgs=False,
    )

    decoder._fit()

    assert (tmp_path / "rsa_results").exists()

