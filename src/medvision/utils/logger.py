"""Structured logging setup for MedVision-AI."""

import logging
import sys


def get_logger(name: str = "medvision", log_level: str = "INFO") -> logging.Logger:
    """Configures and returns a structured logger instance.

    Args:
        name: Name of the logger.
        log_level: Desired log level ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        Configured Logger object.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
