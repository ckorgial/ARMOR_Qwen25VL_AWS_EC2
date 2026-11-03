import logging
import sys

def setup_logging(level=logging.INFO):
    logger = logging.getLogger("agents")
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger