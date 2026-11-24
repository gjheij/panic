import pytest
from copy import deepcopy

from panic.decode import ClassifySubject
from panic.utils import get_config_path, load_yaml, dump_yaml

@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning"
)
def _update_nested_dict(base, updates):
    """Recursively update nested configuration values."""
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and isinstance(base.get(key), dict)
        ):
            _update_nested_dict(base[key], value)
        else:
            base[key] = value

    return base


def make_test_config(tmp_path, settings=None):
    cfg = deepcopy(load_yaml(get_config_path()))

    if settings:
        _update_nested_dict(cfg, settings)

    cfg_file = tmp_path / "panic_config.yml"
    dump_yaml(cfg, cfg_file)

    return str(cfg_file)


def test_panic(tmp_path):

    settings = {
        "label_dict": {
            "US_50%_1_pos": 0,
            "US_100%_1": 1,
        },
        "general_settings": {
            "project_dir": "/lustre/scratch/data/gheij_hpc-fmriprep/projects/CogLearn7T",
            "beta_dir": "/lustre/scratch/data/gheij_hpc-fmriprep/projects/CogLearn7T/derivatives/glmsingle/impulse",
            "source": "stglm",
            "method": "GLMsingleTYPED",
            "standardize": None,
            "save_dir": "/tmp/test_panic",
            "overwrite": True,
            "filters": None,
        },
        "decoding_settings": {
            "n_permutations": 5,
            "parallel": {
                "n_jobs": 1,
            },
            "analysis": {
                "type": "scikit_svm",
                "name": "positive_surprise",
                "args": {},
            },
            "permute_within_groups": True,
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

    cfg_file = make_test_config(tmp_path, settings=settings)

    # Replace entire dictionaries to remove inherited labels/CV arguments.
    cfg = load_yaml(cfg_file)
    cfg["label_dict"] = settings["label_dict"]

    for key in ("analysis", "outer_cv", "inner_cv"):
        cfg["decoding_settings"][key] = settings["decoding_settings"][key]

    dump_yaml(cfg, cfg_file)

    decoder = ClassifySubject("sub-01", cfg_file, save_imgs=True)
    decoder._fit()
