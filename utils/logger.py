from loguru import logger
import os
import sys


def init_logger(work_dir: str, level="INFO"):
    assert level in ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
    os.makedirs(work_dir, exist_ok=True)
    logger.remove()  # remove default logger
    logger.add(os.path.join(work_dir, "train.log"), level=level)
    logger.add(sys.stderr, level=level)
    logger.info(f"Logger initialized, work_dir: {os.path.abspath(work_dir)}")
