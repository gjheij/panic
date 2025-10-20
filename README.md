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
  fold_interval: 3            # if not LORO/LOSO, iterate over labels
  n_permutations: 10         # permutations for ROI-decoding/searchlight
  early_stop_alpha: 0.05      # enable early stopping
  early_stop_batch: 32        # check after X permutations if significance can be reached
  variance_threshold: 1e-12   # avoid flat time series
  permute_both_sets: false    # false: only permute training labels
  permute_within_groups: true # true: LOSO/LORO set up; see 'outer_cv'

  parallel:
    n_jobs: 1                 # this one actually parallizes over permutations/batches
    batch_size: 16            # batch size Parallel process
    backend: "loky"           # backend for Parallel process
    prefer: "processes"       # preference for Parallel process
    verbose: 0                # verbosity of Parallel process

  # Pipeline settings: DO NOT CHANGE THE HEADER NAMES ('SCALER', 'CV', 'ESTIMATOR', etc)
  # preprocessing | if 'standardize' is null
  scaler:
    name: StandardScaler
    args:
      with_mean: true
      with_std: true

  # outer cv
  outer_cv:
    name: LeaveOneGroupOut
    args: {}

  # select test/training labels
  cv:
    name: StratifiedGroupKFold
    args:
      n_splits: 3
      shuffle: False

  # estimator
  estimator:
    name: SVC
    args:
      C: 1
      kernel: linear
      class_weight: balanced

  # select features (e.g., percentile or SelectKBest)
  feature_selection:
    name: SelectPercentile
    args:
      percentile: 5
      score_func: f_classif

  # grid search
  gridsearch:
    name: GridSearchCV
    args:
      param_grid:
        select__percentile: [10, 20, 40, 100] # see 'feature_selection'
        clf__C: [0.01, 0.1, 1, 10]
      scoring: balanced_accuracy
      n_jobs: 1               # keep this at 1 to avoid nesting

  searchlight:
    alpha: 0.05               # value for FDR correction
    radius_mm: 10             # radius around center
    locked:                   # no gridsearch within searchlight..
      clf__C: 1.0
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
subjs=$(seq 1 6)
sources=("halfpipe") # "glmsingle" "stglm")
methods=("LSA")

# proj_dir="/mnt/d/fMRI/HRA"
proj_dir="/mnt/d/fMRI/Development/Haxby"
work_dir="${proj_dir}/logs"
mkdir -p "${work_dir}" 2>/dev/null
n_cpus=4
searchlight=1
# src="glmsingle"
# method="lsa" # for GLMsingle, 'lsa' denotes model D

# default settings
set_keys=(
    general_settings.project_dir="${proj_dir}"
    general_settings.save_dir="${proj_dir}/derivatives/decoding"
    general_settings.n_jobs="${n_cpus}"
)

for src in ${sources[@]}; do
    for method in ${methods[@]}; do
        for subID in ${subjs[@]}; do
            job=$(
                decide_job_type \
                "panic" \
                "sub-${subID}_source-${src}_model-${method}_desc-decoding" \
                0 \
                "${work_dir}" \
                "${n_cpus}" \
                "main"
            )

            # use 'panic run --help' for more
            use_keys=(
                ${set_keys[@]}
                general_settings.source="${src}"
                general_settings.method="${method}"
            )

            # to use DB's manual masks:
            bold_mask=$(find ${proj_dir}/derivatives/fmriprep/sub-${subID}/anat -type f -name "*brain_mask.nii.gz" -and -not -name "*space-*")
            use_keys+=(
                # roi_dict="${DIR_DATA_SOURCE}/sub-${subID}/struct/masks/nifti"
                roi_dict="${bold_mask}"
            )

            cmd=(
                ${job}
                run
                --subject "sub-${subID}"
                --set "${use_keys[@]}"
                # --save-imgs
            )

            if [[ ${searchlight} -eq 1 ]]; then
                cmd+=(--searchlight)
            fi

            eval ${cmd[@]}
        done
    done
done
```
