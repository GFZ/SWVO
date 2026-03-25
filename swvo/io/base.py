# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for all IO modules.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BaseIO(ABC):
    """Abstract base class for all IO classes.

    This base class defines the common interface for external data I/O operations,
    including initialization, reading, and downloading/processing data.

    Subclasses can implement flexible signatures for `read()` and `download_and_process()`
    methods to accommodate different data sources and requirements.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for storing downloaded/processed data.
        If not provided, it will be read from the environment variable
        defined by the subclass's `ENV_VAR_NAME`.

    Raises
    ------
    ValueError
        Raises `ValueError` if necessary environment variable is not set
        and `data_dir` is not provided.
    """

    ENV_VAR_NAME: str = ""  # Must be set by subclasses
    LABEL: str = ""  # Must be set by subclasses

    def __init__(self, data_dir: Optional[Path] = None, prefer_env_var: bool = False) -> None:
        """Initialize the BaseIO class.

        Parameters
        ----------
        data_dir : Path | None
            Data directory for storing data. If not provided, it will be read
            from the environment variable defined by `ENV_VAR_NAME`.
        prefer_env_var : bool, optional
            If True, the environment variable takes precedence over the passed `data_dir` argument.
            If False (default), the passed `data_dir` is used if provided, otherwise the environment variable is used.

        Raises
        ------
            ValueError
            If `data_dir` is None and `ENV_VAR_NAME` is not set in environment,
            or if `prefer_env_var` is True and `ENV_VAR_NAME` is not set.
        """
        if prefer_env_var and self.ENV_VAR_NAME in os.environ:
            data_dir = Path(os.environ[self.ENV_VAR_NAME])
        elif data_dir is None:
            if not self.ENV_VAR_NAME or self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")
            data_dir = Path(os.environ[self.ENV_VAR_NAME])

        self.data_dir: Path = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"{self.__class__.__name__} data directory: {self.data_dir}")

    @abstractmethod
    def read(self, *args, **kwargs) -> pd.DataFrame | list[pd.DataFrame]:
        """Read data.

        Subclasses should implement this method with their specific signature.
        Common parameters include:
        - start_time: datetime
            Start time of the data to read. Must be timezone-aware.
        - end_time: datetime
            End time of the data to read. Must be timezone-aware.
        - download: bool, optional
            Download data on the go if not available locally.
        - Additional parameters specific to each data source.

        Returns
        -------
        pd.DataFrame or list[pd.DataFrame]
            Data for the specified parameters.
        """
        pass

    @abstractmethod
    def download_and_process(self, *args, **kwargs) -> None:
        """Download and process data.

        Subclasses should implement this method with their specific signature.
        Common parameters include:
        - start_time: datetime
            Start time of the data to download. Must be timezone-aware.
        - end_time: datetime
            End time of the data to download. Must be timezone-aware.
        - target_date: datetime
            Target date for data (for single-day sources).
        - request_time: datetime
            Request time for data (for streaming sources).
        - reprocess_files: bool, optional
            If True, re-download and re-process existing files.
        - Additional parameters specific to each data source.

        Returns
        -------
        None
        """
        pass
