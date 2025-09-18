from panic.utils import get_config_path
from panic.decode import ClassifySubject


def test_panic():

    cfg_file = get_config_path()
    subject = "sub-017"
    _ = ClassifySubject(
        subject,
        cfg_file,
        save_imgs=True
    )