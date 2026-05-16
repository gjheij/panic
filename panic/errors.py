class EmptyMaskError(RuntimeError):
    """ROI mask ended up empty after resampling/support filtering."""
    pass

class NoTrialsFoundError(RuntimeError):
    """Trial filtering removed all trials."""
    pass

class NoFeaturesSelectedError(RuntimeError):
    """Feature selection removed all features."""
    pass
