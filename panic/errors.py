class PanicAnalysisError(RuntimeError):
    """Expected analysis failure that may be handled without aborting the full run."""
    pass


class EmptyMaskError(PanicAnalysisError):
    """ROI mask ended up empty; retain its QC evidence for a skipped row."""

    def __init__(self, message, *, reason_code=None, roi_metrics=None):
        super().__init__(message)
        self.reason_code = reason_code
        self.roi_metrics = dict(roi_metrics or {})


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
