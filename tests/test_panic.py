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

    cfg_file = make_test_config(
        tmp_path,
        settings={
            "general_settings": {
                "project_dir": "/mnt/d/fMRI/ENIGMA_FC_SingleTrials/site_data/acquisition/site-Amsterdam_pi-Visser_sample-01_desc-2011",
                "source": "halfpipe",
                "save_dir": "/tmp/test_panic",
                "overwrite": True,
                "filters": None
            },
            "decoding_settings": {
                "n_permutations": 5,
                "parallel": {
                    "n_jobs": 1,
                },
            }
        },
    )
    
    decoder = ClassifySubject(
        "sub-sc001",
        cfg_file,
        save_imgs=True,
    )

    decoder._fit()
