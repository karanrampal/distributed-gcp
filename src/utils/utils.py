"""Utility functions for distributed computing"""

import logging
from typing import Dict, Optional, Union

import yaml


class Params:
    """Class to load hyperparameters from a yaml file."""

    def __init__(self, inp: Union[Dict, str]) -> None:
        self.update(inp)

    def save(self, yaml_path: str) -> None:
        """Save parameters to yaml file at yaml_path"""
        with open(yaml_path, "w", encoding="utf-8") as fptr:
            yaml.safe_dump(self.__dict__, fptr)

    def update(self, inp: Union[Dict, str]) -> None:
        """Loads parameters from yaml file or dict"""
        if isinstance(inp, dict):
            self.__dict__.update(inp)
        elif isinstance(inp, str):
            with open(inp, encoding="utf-8") as fptr:
                params = yaml.safe_load(fptr)
                self.__dict__.update(params)
        else:
            raise TypeError(
                "Input should either be a dictionary or a string path to a config file!"
            )


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
