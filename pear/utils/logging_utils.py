import logging
import os
import sys


def get_logger(name: str, default_level: str = "WARNING") -> logging.Logger:
    """A logger with its own stderr handler, at PEAR_LOGGING_LEVEL or, without the variable, at default_level."""
    # its own handler: the vLLM server module sets the root handler to ERROR in every worker
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s:%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(os.getenv("PEAR_LOGGING_LEVEL", default_level))
        logger.propagate = False

    return logger
