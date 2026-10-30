"""Feature selectors with explicit behavior for small decoding masks."""
from numbers import Integral

import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif


class SelectPercentileMinFeatures(SelectPercentile):
    """Keep at least min_features scored features for a positive percentile.

    Ordinary percentile selection is unchanged whenever it already retains the
    requested minimum. Otherwise, add highest-scoring non-NaN features, breaking
    score ties by feature order. Scoring and selection use only fit's training
    data. A zero percentile explicitly selects nothing; min_features=0 restores
    sklearn's behavior. Empty inputs and upstream variance filtering are unchanged.
    """

    def __init__(self, score_func=f_classif, *, percentile=10, min_features=1):
        super().__init__(score_func=score_func, percentile=percentile)
        self.min_features = min_features

    def fit(self, X, y=None):
        if isinstance(self.min_features, (bool, np.bool_)) or not isinstance(self.min_features, Integral) or self.min_features < 0:
            raise ValueError("min_features must be a nonnegative integer")
        return super().fit(X, y)

    def _get_support_mask(self):
        mask = super()._get_support_mask()
        if self.percentile == 0 or self.min_features == 0:
            return mask
        minimum = min(self.min_features, len(mask))
        if int(mask.sum()) >= minimum:
            return mask
        scores = np.asarray(self.scores_, dtype=float)
        valid = np.flatnonzero(~np.isnan(scores))
        ranking = valid[np.argsort(-scores[valid], kind="stable")]
        additions = ranking[~mask[ranking]][:minimum - int(mask.sum())]
        mask[additions] = True
        return mask
