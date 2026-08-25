# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0


import logging
import sys
from pathlib import Path
from typing import ClassVar, Optional

from rich.highlighter import NullHighlighter
from rich.logging import RichHandler

# Get the package logger
logger = logging.getLogger(__package__)
logger.addHandler(logging.NullHandler())


class _RichMarkupFormatter(logging.Formatter):
    COLORS: ClassVar = {
        logging.DEBUG: "cyan",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "bold red",
    }

    def format(self, record) -> str:  # noqa: ANN001
        msg = super().format(record)
        style = self.COLORS.get(record.levelno, "")
        return f"[{style}]{msg}[/{style}]" if style else msg


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


def _running_in_ipython() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def setup_logging(level: str | int = "INFO", log_file: Optional[Path] = None, file_mode: str = "w") -> None:
    """Setup logging for the swvo package and root logger.

    Parameters
    ----------
    level : str | int, optional
        Logging level, by default is INFO
    log_file : Path, optional
        Path to log file. If None, only console logging is enabled.If provided, logs will be written to both console and file., by default None
    """
    try:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        elif not isinstance(level, int):
            raise ValueError(f"Invalid logging level: {level}. Expected a string or integer logging level.")
    except AttributeError:
        msg = f"Invalid logging level: {level}. Use one of CRITICAL, FATAL, ERROR, WARNING, WARN, INFO, DEBUG, NOTSET, or their corresponding integer values."
        raise ValueError(msg)  # noqa: B904

    # Configure root logger so all loggers inherit the formatting
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    log_format = "[%(levelname)-8s] %(asctime)s - %(name)s:%(lineno)d - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root_logger.handlers
    )

    if not has_console_handler:
        formatter = _RichMarkupFormatter(
            log_format,
            datefmt=datefmt,
        )

        if _running_in_ipython():
            formatter = _ColorFormatter(
                log_format,
                datefmt=datefmt,
            )
            console_handler = logging.StreamHandler(sys.stdout)
        else:
            console_handler = RichHandler(
                show_time=False,
                show_level=False,
                show_path=False,
                markup=True,
                rich_tracebacks=False,
                highlighter=NullHighlighter(),
            )

        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(
            log_file,
            mode=file_mode,
        )
        file_handler.setFormatter(
            logging.Formatter(
                log_format,
                datefmt=datefmt,
            )
        )
        root_logger.addHandler(file_handler)
