# PANIC: Pattern Analysis of Neural Imaging in Conditioning 

Tools and recipes to (1) estimate single-trial betas with HALFpipe’s LSS/LSA, (2) classify CS+ vs CS-, and (3) track the temporal evolution of CS+ responses across the session.

# Features
- Single-trial estimation (LSS/LSA) using HALFpipe-compatible workflows gives a beta per trial—ideal for MVPA and learning curves.

- Classification of CS+ vs CS- tests whether multivariate patterns distinguish conditioned from non-conditioned stimuli.

- Temporal evolution of CS+ betas lets you visualize acquisition/extinction dynamics over runs/trials.

# CS+ vs CS- classification
Train a classifier to predict CS label from single-trial beta patterns.

Recommended recipe

- Features: vectorized beta maps (whole-brain, mask, or ROI).

- Preprocessing: z-score across trials; optionally PCA to ~50–200 comps.

- Classifier: linear SVM or logistic regression (L2).

- CV: leave-run-out (grouped CV) to avoid leakage; stratify by label.

- Metrics: accuracy, ROC-AUC, balanced accuracy; permutation test optional.

# Temporal evolution of CS+ beta time-series
Quantify how CS+ responses change over trials (e.g., acquisition or extinction).

1. Trial-wise trajectory: average CS+ betas within an ROI (or searchlight) and plot vs. trial index.

2. Run-wise summary: mean CS+ beta per run/block; fit a slope or mixed-effects trend.


# Installation

```bash
pip install git+https://github.com/gjheij/panic
```

# Configuration

``panic`` uses a YAML configuration file (defaults to panic.utils.get_config_path()).
Example config.yml:

```yaml
label_dict:
  CS-: 0
  CS+_noUCS: 1

roi_dict:
  lateral: [7001]
  basal: [7003]
  central: [7005]
  medial: [7006]
  total: [7001, 7003, 7005, 7006]

general_settings:
  project_dir: "/mnt/d/fMRI/HRA"
  save_dir: "/mnt/d/fMRI/HRA/derivatives/decoding"
  method: "lss"
  source: "stglm"
  n_jobs: 10

decoding_settings:
  param_grid:
    "select__k": [100, 200, 300, "all"]
    "svc__C": [0.01, 0.1, 1, 10]
  internal_folds: 3
  internal_interval: 3
  n_permutations: 1000
```

# Command-line usage

```bash
usage: panic [-h] [-c CONFIG] {show,config,run} ...
```

| Mode     | Description                                 |
| -------- | ------------------------------------------- |
| `show`   | Print the current YAML config.              |
| `config` | Update and optionally save the config file. |
| `run`    | Run decoding for one or more subjects.      |

Full `panic run` help:

```bash
usage: panic run [-h] [--subject SUBJECT [SUBJECT ...]]
                 [--set KEY=VALUE [KEY=VALUE ...]] [--save-config]

options:
  -h, --help            Show this help message and exit
  --subject SUBJECT [SUBJECT ...], -s SUBJECT [SUBJECT ...]
                        Subject ID(s), e.g., sub-015 sub-016
  --set KEY=VALUE [KEY=VALUE ...]
                        Override(s) to apply for this run.
  --save-config         Persist --set overrides into the YAML before running.
```

Specific settings from the config file can be overwritten from the command line:
```bash
panic run ... --set general_settings.project_dir=/some/other/path
```

Example shell script:

```bash
subjs=("015" "016" "017" "018" "019" "020" "021" "022")
sources=("stglm")  # or: "glmsingle"
methods=("lsa" "lss")

proj_dir="/mnt/d/fMRI/HRA"
work_dir="${DIR_LOGS}"
mkdir -p "${work_dir}" 2>/dev/null
n_cpus=4

# Default keys
set_keys=(
    general_settings.project_dir="${proj_dir}"
    general_settings.save_dir="${proj_dir}/derivatives/decoding"
    general_settings.n_jobs="${n_cpus}"
)

for src in "${sources[@]}"; do
    for method in "${methods[@]}"; do
        for subID in "${subjs[@]}"; do
            job=$(
                decide_job_type \
                "panic" \
                "sub-${subID}_source-${src}_model-${method}_desc-decoding" \
                0 \
                "${work_dir}" \
                "${n_cpus}" \
                "main"
            )

            use_keys=(
                "${set_keys[@]}"
                general_settings.source="${src}"
                general_settings.method="${method}"
                roi_dict="${DIR_DATA_SOURCE}/sub-${subID}/struct/masks/nifti"
            )

            cmd=(
                ${job}
                run
                --subject "sub-${subID}"
                --set "${use_keys[@]}"
            )

            eval "${cmd[@]}"
        done
    done
done
```
