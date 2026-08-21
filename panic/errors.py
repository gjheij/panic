class PanicAnalysisError(RuntimeError):
    """Expected analysis failure that may be handled without aborting the full run."""
    pass


class EmptyMaskError(PanicAnalysisError):
    """ROI mask ended up empty after resampling/support filtering."""
    pass


class NoTrialsFoundError(PanicAnalysisError):
    """Trial filtering removed all trials."""
    pass


class NoFeaturesSelectedError(PanicAnalysisError):
    """Feature selection removed all features."""
    pass


__all__ = [
    "PanicAnalysisError",
    "EmptyMaskError",
    "NoTrialsFoundError",
    "NoFeaturesSelectedError",
]
