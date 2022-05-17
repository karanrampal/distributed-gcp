"""Utility functions for distributed computing"""

import logging
from typing import Optional


def set_logger(log_path: Optional[str] = None) -> None:
    """Set the logger to log info in terminal and file at log_path.
    Args:
        log_path: Location of log file, optional
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        if log_path:
            file_handler = logging.FileHandler(log_path, mode="w")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
            )
            logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(stream_handler)
