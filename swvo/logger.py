# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0


import logging
import sys
from pathlib import Path
from typing import Optional

# Get the package logger
logger = logging.getLogger("swvo")
logger.addHandler(logging.NullHandler())


class _ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record):
        msg = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{msg}{self.RESET}"


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None, file_mode: str = "w"):
    """Setup logging for the swvo package and root logger.

    Parameters
    ----------
    level : str, optional
        Logging level, by default is INFO
    log_file : Path, optional
        Path to log file. If None, only console logging is enabled.If provided, logs will be written to both console and file., by default None
    """
    try:
        level = getattr(logging, level.upper())
    except AttributeError:
        raise ValueError(f"Invalid logging level: {level}. Use one of DEBUG, INFO, WARNING, ERROR, CRITICAL.")

    # Configure root logger so all loggers inherit the formatting
    root_logger = logging.getLogger()

    # Check if already configured
    if any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        return

    # Console handler with colors
    log_format = "[%(levelname)-8s] %(asctime)s - %(name)s:%(lineno)d - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_ColorFormatter(log_format, datefmt=datefmt))

    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=datefmt))
        root_logger.addHandler(file_handler)
