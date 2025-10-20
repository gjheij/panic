from panic.utils import get_config_path
from panic.decode import ClassifySubject


def test_searchlight():

    cfg_file = get_config_path()
    subject = "sub-1"
    decoder = ClassifySubject(
        subject,
        cfg_file,
        save_imgs=True,
        searchlight=True
    )

    decoder._fit()