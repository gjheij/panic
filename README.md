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

# Configuration of `scikit-learn`-pipelines

Because `scikit-learn` is so consistent with its I/O, we can use the configuration file to flexibly generate decoding/classification pipelines.
However, the metric that is ultimately used for classification is the delta value between the observed and permutation-derived null-distribution by permuting the labels a bunch of times.
This procedure will be performed regardless of method-of-choice.
If you are only interested in the observed accuracy, you could set the permutations to a small number.
Below, I highlight several classifiers that are supported by `panic`:

> [!NOTE]
> The package has been built around ``SVM``. The other implementations have not been thoroughly tested.
> It basically rests on sklearn's I/O consistency.
> Please open an issue in case you encounter problems.

## Linear classifiers

| Classifier                           | [factory](panic/factory.py) name              | Best for                                | Notes / Typical config                                                          |
| ------------------------------------ | ------------------------------ | --------------------------------------- | ------------------------------------------------------------------------------- |
| **[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)**               | `"LogisticRegression"`         | Standard binary/multiclass decoding     | Robust and interpretable; use `penalty='l2'` or `'elasticnet'`, `solver='saga'` |
| **[LinearDiscriminantAnalysis (LDA)](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)** | `"LinearDiscriminantAnalysis"` | ROI & searchlight decoding              | Fast, stable; use `solver='lsqr', shrinkage='auto'` for p≫n                     |
| **[RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)**                  | `"RidgeClassifier"`            | fMRI decoding (multivariate, linear)    | Equivalent to L2-logistic regression but faster                                 |
| **[SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)**                    | `"SGDClassifier"`              | Large datasets or regularization sweeps | Supports `loss='hinge'`, `'log_loss'`, `'modified_huber'`                       |
| **[LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)**                        | `"LinearSVC"`                  | Common in searchlight decoding          | Often used with `C=1`, `dual=False` if n_samples > n_features                   |

## Kernel & non-linear classifiers

| Classifier                               | [factory](panic/factory.py) name             | Best for                      | Notes / Typical config                                               |
| ---------------------------------------- | ----------------------------- | ----------------------------- | -------------------------------------------------------------------- |
| **[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)** (RBF kernel)                     | `"SVC"`                       | Nonlinear decision boundaries | Slow on high-dim fMRI, but fine for ROI-level decoding               |
| **[GaussianProcessClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)**            | `"GaussianProcessClassifier"` | Small datasets                | Probabilistic output; interpretable kernel-based                     |
| **[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)**               | `"RandomForestClassifier"`    | ROI-level decoding            | Handles nonlinearities; robust to noise, but slower for permutations |
| **[GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) / XGBoost** | (you can add)                 | Nonlinear decoding            | Better performance on structured ROI data; slower for searchlight    |
| **[KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)**                 | `"KNeighborsClassifier"`      | Small, low-dimensional ROIs   | Parameter-free, but not great for high-dim voxel patterns            |

## Sparse / feature-selection–oriented

| Classifier                                           | Module   | Benefit                    | Notes                                                         |
| ---------------------------------------------------- | -------- | -------------------------- | ------------------------------------------------------------- |
| **[LogisticRegressionCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)**                             | sklearn  | built-in CV for C tuning   | Helps control sparsity in high-dim ROI                        |
| **Lasso (L1) LogisticRegression**                    | sklearn  | feature sparsity           | Use `penalty='l1'`, `solver='saga'`; yields sparse brain maps |
| **[ElasticNetCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)**                                     | sklearn  | L1+L2 mixed regularization | Good middle ground for interpretability vs. accuracy          |

4. Probabilistic and Bayesian options

| Classifier                                | Source               | Strength                         | Comment                                           |
| ----------------------------------------- | -------------------- | -------------------------------- | ------------------------------------------------- |
| **[NaiveBayes (GaussianNB)](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)**               | sklearn              | very fast baseline               | OK for sanity checks; poor with correlated voxels |
| **[BayesianRidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)**               | sklearn.linear_model | stable under small samples       | slower but provides uncertainty estimates         |
| **LogisticRegressionCV with refit=False** | sklearn              | yields probability distributions | integrates well with permutation inference        |

The following aliases are supported:

```python
_ALIASES = {
    "SVM": "SVC",
    "LinearSVM": "LinearSVC",
    "LR": "LogisticRegression",
    "LDA": "LinearDiscriminantAnalysis",
    "QDA": "QuadraticDiscriminantAnalysis",
    "RF": "RandomForestClassifier",
    "ET": "ExtraTreesClassifier",
    "GPC": "GaussianProcessClassifier",
}
```