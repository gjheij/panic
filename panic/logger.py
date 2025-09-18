# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import sys
import joblib
import logging
import tqdm as tqmod
from contextlib import contextmanager

@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


def get_logger(name="PANIC", level=logging.INFO, use_tqdm=True, logfile=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = TqdmSafeHandler() if use_tqdm else logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

        if logfile:
            fh = logging.FileHandler(logfile)
            fh.setFormatter(fmt)
            fh.setLevel(level)
            logger.addHandler(fh)

    return logger

class TqdmSafeHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        write = getattr(tqmod, "write", None)
        if callable(write):
            write(msg)               # keeps progress bars intact
        else:
            sys.stderr.write(msg + "\n")  # fallback if write is missing
            sys.stderr.flush()