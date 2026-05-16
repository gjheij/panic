# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import sys
import joblib
import logging
import tqdm as tqmod
import contextvars
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Union


current_subject = contextvars.ContextVar("current_subject", default="-")


class AnsiColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }

    def __init__(self, fmt=None, datefmt=None, use_color=True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record):
        msg = super().format(record)
        if not self.use_color:
            return msg

        color = self.COLORS.get(record.levelno)
        if not color:
            return msg

        return f"{color}{msg}{self.RESET}"


class SubjectFilter(logging.Filter):
    def filter(self, record):
        record.subject = current_subject.get("-")
        return True


class TqdmSafeHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        write = getattr(tqmod, "write", None)

        if callable(write):
            write(msg)
        else:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()


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


def init_logging(
    level: int = logging.INFO,
    logfile: Optional[Union[str, Path]] = None,
    use_tqdm: bool = True,
    use_color: bool = True,
    filemode: str = "w",
) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    subject_filter = SubjectFilter()

    base_fmt = (
        "[%(asctime)s.%(msecs)03d] "
        "[%(subject)s] "
        "[%(levelname)s] "
        "%(name)s - %(message)s"
    )
    date_fmt = "%Y-%m-%d %H:%M:%S"

    handler = TqdmSafeHandler() if use_tqdm else logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.addFilter(subject_filter)
    handler.setFormatter(
        AnsiColorFormatter(base_fmt, datefmt=date_fmt, use_color=use_color)
    )
    root.addHandler(handler)

    if logfile:
        fh = logging.FileHandler(logfile, mode=filemode, encoding="utf-8")
        fh.setLevel(level)
        fh.addFilter(subject_filter)
        fh.setFormatter(logging.Formatter(base_fmt, datefmt=date_fmt))
        root.addHandler(fh)

    logging.captureWarnings(True)
    return root


def get_logger(
    name: str = "PANIC",
    level: int = logging.INFO,
    use_tqdm: bool = True,
    logfile: Optional[Union[str, Path]] = None,
    use_color: bool = True,
    filemode: str = "w",
) -> logging.Logger:
    """
    Backwards-compatible logger setup.

    Configures root logging with color, tqdm-safe output, subject context,
    and optional file logging, then returns a named logger.
    """
    init_logging(
        level=level,
        logfile=logfile,
        use_tqdm=use_tqdm,
        use_color=use_color,
        filemode=filemode,
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def worker_init(
    log_dir: Union[str, Path],
    log_level: int,
) -> None:
    pid = os.getpid()
    worker_log = os.path.join(log_dir, f"log.worker.{pid}.log")

    init_logging(
        level=log_level,
        logfile=worker_log,
        use_tqdm=False,
        use_color=True,
        filemode="a",
    )

    logger = logging.getLogger("calinet.worker")
    logger.info(f"Worker {pid} logging to {worker_log}")