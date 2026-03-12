# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PlasmasphereCombinedInputsReader:
    """Reads one of the available combined inputs for plasmasphere density prediction.

    Parameters
    ----------
    folder : str
        The folder where the combined inputs files are stored.

    Raises
    ------
    FileNotFoundError
        If the data folder does not exist.
    RuntimeError
        If the source of data requested is not among the available ones.
    """

    ENV_VAR_NAME = "PLASMASPHERE_COMBINED_INPUTS_DIR"
    LABEL = "plasmsphere_combined_inputs"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)  # ty: ignore[invalid-assignment]

        self.data_dir: Path = Path(data_dir)  # ty:ignore[invalid-argument-type]

        logger.info(f"Plasmasphere combined inputs directory: {self.data_dir}")

        if not self.data_dir.exists():
            msg = f"Plasmasphere combined inputs directory {self.data_dir} does not exist! Impossible to retrive data!"
            logger.error(msg)
            raise FileNotFoundError(msg)

    def _read_single_file(self, date: datetime, source: str) -> pd.DataFrame | None:
        """Read a single file from the specified folder for the given date and source.

        Parameters
        ----------
        date : datetime
            date of the plasmasphere prediction we want to read
        source : str
            source of the combined input we want to read. Available "kp" and "solar_wind"

        Returns
        -------
        pd.DataFrame or None
            pandas.DataFrame with the data read from the file, or None if the file does not exist.
        """

        file_name = f"combined_inputs/{source}/{source}_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}T{str(date.hour).zfill(2)}00.csv"
        file_path = os.path.join(self.data_dir, file_name)

        logger.info(f"Looking for file {file_path} for source {source} and date {date}")

        if not os.path.isfile(file_path):
            msg = f"No suitable files found in the folder {self.data_dir} for the requested date {date}"
            logger.warning(msg)
            return None

        if source == "solar_wind":
            data = pd.read_csv(file_path, parse_dates=["date"])
            data["t"] = data["date"]
            data.drop(labels=["date"], axis=1, inplace=True)
        if source == "kp":
            data = pd.read_csv(file_path, parse_dates=["t"])

        return data

    def read(self, source: str, requested_date: datetime | None = None) -> pd.DataFrame | None:
        """Read one of the available combined inputs for plasmasphere density prediction.

        Parameters
        ----------
        source : str
            The source of combined input requested. Available "kp" and "solar_wind".
        requested_date : datetime | None, optional
            Date of combined input thar we want to read up to hour precision, by default None which means current date.

        Returns
        -------
        pd.DataFrame|None
            pandas.DataFrame with the data read from the file, or None if the file does not exist.

        Raises
        ------
        RuntimeError
            If the source of data requested is not among the available ones.
        """
        if requested_date is None:
            requested_date = datetime.now(timezone.utc).replace(microsecond=0, minute=0, second=0)

        if source == "kp":
            requested_date = requested_date.replace(minute=0, second=0, microsecond=0)
            return self._read_single_file(requested_date, "kp")
        elif source == "solar_wind":
            requested_date = requested_date.replace(minute=0, second=0, microsecond=0)
            return self._read_single_file(requested_date, "solar_wind")
        else:
            msg = f"Combined input {source} requested not available..."
            logger.error(msg)
            raise RuntimeError(msg)
