# Cross-validation across differing trial counts

Place this block inside `decoding_settings`:

```yaml
scoring: balanced_accuracy
outer_cv:
  mode: sklearn
  name: StratifiedKFold
  args: {n_splits: 3, shuffle: true, random_state: 42}
inner_cv:
  name: StratifiedKFold
  args: {n_splits: 3, shuffle: true, random_state: 42}
```

Three folds are a starting point, not an automatic guarantee of adequate data.
Counts refer to included trials after condition selection/filtering. Every class
must have at least `n_splits` trials; PANIC fails explicitly otherwise instead of
accepting sklearn's missing-class warning. Fold counts are never silently reduced.
If grid search runs, the same check is applied to each outer training set before
inner fitting. Inner settings are unused when no search object is constructed.
For 16 CSm and 6 CSpu trials, each outer test set contains two CSpu trials and
training contains four: inner three-fold CV is possible, inner five-fold is not.
Very small samples still produce uncertain estimates despite valid splits.

## Permutation behavior

Built-in plugins use one permuted label vector per null realization. For
`StratifiedKFold` and `RepeatedStratifiedKFold`, the shared scoring loop constructs
splits from the labels being evaluated, for both observed and permuted data.
The splitter algorithm/settings remain the same, while indices change with the
labels. Randomized stratified splitters require an integer `random_state`.
This RNG is separate from the permutation RNG. No permutation is discarded or
retried to obtain a valid result. The common implementation covers ROI and
searchlight decoding; custom plugins must use the shared helper or implement
this procedure explicitly.

Other designs, including `nth_trial` and group-based splitters, retain supplied
folds and their existing validation. This update does not solve invalid null
folds for those designs. Within-group permutation follows `permute_within_groups`;
use genuine experimental exchangeability groups. Stratification does not make
nearby fMRI trials independent. Prefer held-out genuine runs where the design
supports that scientific question. This update does not infer temporal blocks,
run identities, or automatic subject exclusions.

ROI CSVs and searchlight metadata record `cv_permutation_policy` as
`rebuild_stratified_v1` or `fixed`. Searchlight completion checks reject older
stratified outputs lacking the new policy marker. Existing ROI results require
explicit overwrite or a new result directory; high-level PANIC analysis IDs do
not encode all CV settings.
