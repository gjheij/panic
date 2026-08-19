#!/usr/bin/env python3
"""
panic — Run PANIC decoding with CLI-configurable YAML settings.

Examples
--------
# Use default config, run one subject
python panic_cli.py run --subject sub-015

# Override nested settings (write into a temporary config for this run only)
panic run --subject 016 --set general_settings.project_dir=/mnt/d/fMRI/HRA general_settings.save_dir=/mnt/d/fMRI/HRA/derivatives/decoding general_settings.n_jobs=10 general_settings.source=glmsingle general_settings.method=lsa decoding_settings.param_grid.select__='[100, 200, 300, 400]'

# Show current YAML contents
python panic_cli.py show
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import logging
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml  # PyYAML

# Your PANIC imports
from panic.utils import (
    get_config_path,
    load_yaml,
    dump_yaml
)
from panic.logger import init_logging
from panic.decode import ClassifySubject

opj = os.path.join

# YAML helpers
def parse_value(raw: str) -> Any:
    """Parse a string like '1e-3', 'true', '[1,2]', '"str with spaces"' into Python types."""
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


_key_index_re = re.compile(r"^(?P<name>[^\[\]]+)(\[(?P<idx>\d+)\])?$")


def _ensure_list_len(lst: list, idx: int, fill: Any = None):
    while len(lst) <= idx:
        lst.append(deepcopy(fill))


def set_by_path(obj: Any, path: str, value: Any) -> None:
    """
    Set obj[path]=value where path supports dotted dict keys and list indices:

      trainer.epochs=50
      model.layers[0].dropout=0.1
      data.folds[2].seed=123

    If intermediate containers are missing, they are created as dicts or lists as needed.
    """
    parts = path.split(".")
    cur = obj
    for i, token in enumerate(parts):
        m = _key_index_re.match(token)
        if not m:
            raise ValueError(f"Bad key token: {token}")
        name = m.group("name")
        idx = m.group("idx")
        last = i == len(parts) - 1

        # descend/create dict layer
        if not isinstance(cur, dict):
            raise TypeError(f"Cannot set '{token}' under non-dict container at {'.'.join(parts[:i])}")
        if name not in cur or cur[name] is None:
            # create list if token has [idx], else dict
            cur[name] = [] if idx is not None else {}
        node = cur[name]

        if idx is not None:
            # list layer
            idx = int(idx)
            if not isinstance(node, list):
                # convert/replace with list
                cur[name] = []
                node = cur[name]
            if last:
                _ensure_list_len(node, idx)
                node[idx] = value
            else:
                _ensure_list_len(node, idx, {})
                cur = node[idx]
        else:
            if last:
                cur[name] = value
            else:
                if not isinstance(node, dict):
                    cur[name] = {}
                cur = cur[name]


def apply_assignments(
        cfg: Dict[str, Any],
        assignments: Iterable[str]
    ) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply key=value strings to cfg. Returns (updated_cfg, list_of_changes_as_strings).
    """
    cfg = deepcopy(cfg)
    changes = []
    for kv in assignments:
        if "=" not in kv:
            raise ValueError(f"--set expects KEY=VALUE, got: {kv}")
        key, raw = kv.split("=", 1)
        val = parse_value(raw)
        set_by_path(cfg, key, val)
        # pretty for echo
        changes.append(f"{key} = {val!r}")
    return cfg, changes


# CLI actions
def action_show(args):
    cfg_path = Path(args.config or get_config_path())
    data = load_yaml(cfg_path)
    yaml_str = yaml.safe_dump(data, sort_keys=False)
    print(f"# Config: {cfg_path}\n{yaml_str}")


def action_config(args):
    cfg_path = Path(args.config or get_config_path())
    data = load_yaml(cfg_path)
    updated, changes = apply_assignments(data, args.set or [])
    if not args.save:
        print("No --save given; preview of changes only:\n" + "\n".join(f"  - {c}" for c in changes))
        return
    # backup
    backup = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    backup.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    dump_yaml(updated, cfg_path)
    print(f"Saved {len(changes)} change(s) to {cfg_path} (backup at {backup.name}).")


def _write_temp_config(base_cfg: Dict[str, Any]) -> Path:
    fd, temp_path = tempfile.mkstemp(prefix="panic_cfg_", suffix=".yml")
    os.close(fd)
    temp = Path(temp_path)
    dump_yaml(base_cfg, temp)
    return temp


def action_run(args):
    cfg_path = Path(args.config or get_config_path())
    print(f"Using base config: {cfg_path}")
    base_cfg = load_yaml(cfg_path)

    # Run PANIC
    subjects = args.subject or []
    if not subjects:
        print("Please provide at least one --subject")
        sys.exit(2)

    # Apply overrides (if any)
    cfg_for_run = base_cfg
    changes = []
    if args.set:
        cfg_for_run, changes = apply_assignments(base_cfg, args.set)

    # Decide where to write config
    if args.set:
        # ephemeral temp config
        cfg_to_use = _write_temp_config(cfg_for_run)
        print(f"Using temporary config with overrides at: {cfg_to_use}")
    else:
        cfg_to_use = cfg_path
    
    loaded_cfg = load_yaml(cfg_to_use)
    if changes:
        print("Applied overrides:")
        for c in changes:
            print(f" {c}")

    for subj in subjects:

        gen_settings = loaded_cfg["general_settings"]
        dec_settings = loaded_cfg["decoding_settings"]
        analysis = dec_settings["analysis"]

        analysis_name = analysis.get("name") or analysis.get("type")

        save_dir = opj(
            gen_settings["save_dir"],
            analysis_name,
            subj,
        )

        os.makedirs(save_dir, exist_ok=True)

        # Build run-specific log name
        mode = "searchlight" if args.searchlight else "roi"
        source = gen_settings["source"]
        method = gen_settings["method"]

        log_file = os.path.join(
            save_dir,
            f"{subj}_model-{method}_source-{source}_desc-{mode}.log",
        )

        log_level = logging.DEBUG if args.debug else logging.INFO

        init_logging(
            level=log_level,
            logfile=log_file,
            filemode="w",
        )

        decoder = ClassifySubject(
            subj,
            str(cfg_to_use),
            save_imgs=args.save_imgs,
            searchlight=args.searchlight,
        )

        decoder._fit()


# Argparse
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="panic",
        description="Run PANIC decoding and update YAML settings from the command line.",
    )

    p.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable verbose debug-level logging.\n\n"
            "When set, the logger runs at ``logging.DEBUG`` instead of "
            "``logging.INFO``. This produces substantially more detailed diagnostic "
            "output, which is useful for troubleshooting file discovery, conversion "
            "steps, metadata handling, and worker behavior."
        ),
    )

    p.add_argument(
        "-c", "--config",
        help="Path to the YAML config. Defaults to panic.utils.get_config_path().",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # show
    sp = sub.add_parser(
        "show",
        help="Print the current YAML config."
    )

    sp.set_defaults(func=action_show)

    # config (update & save)
    sp = sub.add_parser(
        "config",
        help="Update the YAML (optionally save)."
    )

    sp.add_argument(
        "--set",
        metavar="KEY=VALUE",
        nargs="+",
        help="Override(s), e.g., general_settings.source=glmsingle"
    )

    sp.add_argument(
        "--save",
        action="store_true",
        help="Write the changes back to the YAML."
    )

    sp.set_defaults(func=action_config)

    # run
    sp = sub.add_parser(
        "run",
        help="Run decoding for one or more subjects."
    )

    sp.add_argument(
        "--subject", "-s",
        nargs="+",
        help="Subject ID(s), e.g., sub-015 sub-016", 
        required=False
    )

    sp.add_argument(
        "--save-imgs",
        action='store_true',
        help="Save the resampled masks as new nifti files ('_desc-resampled.nii.gz' is appended)", 
        required=False
    )    

    sp.add_argument(
        "--searchlight",
        action='store_true',
        help="Run a searchlight analysis (see subsection 'searchlight' in config file)", 
        required=False
    )    

    sp.add_argument(
        "--set",
        metavar="KEY=VALUE",
        nargs="+",
        help="Override(s) to apply for this run."
    )

    sp.set_defaults(func=action_run)

    return p


def main(argv: List[str] | None = None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
