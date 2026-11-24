"""Ephemeral subject-scoped caches; input images must remain immutable."""
import logging
from time import perf_counter
import numpy as np
import nibabel as nib
from panic.masking import beta_support_mask
logger = logging.getLogger(__name__)

class SubjectCache:
    def __init__(self):
        self.discovery = {}
        self.segmentations = {}
        self.beta_image = None
        self.supports = {}

    def support(self, image, data, threshold):
        if image is not self.beta_image:
            self.beta_image = image
            self.supports.clear()
        key = float(threshold)
        if key not in self.supports:
            start = perf_counter()
            self.supports[key] = beta_support_mask(data, variance_threshold=key)
            logger.info("Beta support cache built: threshold=%s elapsed=%.3fs", key, perf_counter()-start)
        else:
            logger.debug("Beta support cache hit: threshold=%s", key)
        return self.supports[key]

    def segmentation(self, path, sanitize):
        if path not in self.segmentations:
            start = perf_counter()
            img = nib.load(path)
            values = np.rint(sanitize(img, fill=0.0)).astype(np.int32, copy=False)
            self.segmentations[path] = (values, img.affine.copy())
            logger.info("Segmentation cache built: %s elapsed=%.3fs", path, perf_counter()-start)
        return self.segmentations[path]
