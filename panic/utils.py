# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import yaml
from pathlib import Path
from typing import Any, Dict
from importlib.resources import files, as_file

import panic
from panic.logger import get_logger
logger = get_logger(__name__)

def get_config_path(filename="config.yml"):
    with as_file(files(panic) / filename) as p:
        return p

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)