"""Shared logger: console output plus an optional log file."""
import logging
import sys

LOGGER_NAME = "llmcpu"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"


def get_logger():
    return logging.getLogger(LOGGER_NAME)


def setup_logging(log_file=None, level=logging.INFO):
    """Attach a stdout handler (once) and, if requested, a UTF-8 file handler."""
    logger = get_logger()
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    if not any(getattr(h, "_llmcpu_console", False) for h in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console._llmcpu_console = True
        console.setFormatter(formatter)
        logger.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
