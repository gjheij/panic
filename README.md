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
