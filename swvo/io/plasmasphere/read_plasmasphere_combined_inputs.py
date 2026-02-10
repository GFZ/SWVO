# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from datetime import datetime, timezone

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

    def __init__(self, folder: str):
        self.data_folder = folder
        self._check_data_folder()

    def _check_data_folder(self) -> None:
        """Checks if the data folder exists.

        Raises
        ------
        FileNotFoundError
            If the data folder does not exist.
        """
        if not os.path.exists(self.data_folder):
            msg = f"Data folder {self.data_folder} for WP3 plasma output not found...impossible to retrieve data."
            logger.error(msg)
            raise FileNotFoundError(msg)

    @staticmethod
    def _read_single_file(folder: str, date: datetime, source: str) -> pd.DataFrame | None:
        """Read a single file from the specified folder for the given date and source.

        Parameters
        ----------
        folder : str
            folder where we look for the plasmasphere prediction
        date : datetime
            date of the plasmasphere prediction we want to read
        source : str
            source of the combined input we want to read. Available "kp" and "solar_wind"

        Returns
        -------
        pd.DataFrame or None
            pandas.DataFrame with the data read from the file, or None if the file does not exist.
        """

        file_name = ""

        if source == "kp":
            file_name = (
                f"kp_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}T{str(date.hour).zfill(2)}00.csv"
            )
        if source == "solar_wind":
            file_name = f"solar_wind_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}T{str(date.hour).zfill(2)}00.csv"

        file_path = os.path.join(folder, file_name)

        logger.info(f"Looking for file {file_path} for source {source} and date {date}")

        if not os.path.isfile(file_path):
            msg = f"No suitable files found in the folder {folder} for the requested date {date}"
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
            return self._read_single_file(os.path.join(self.data_folder, "combined_inputs/kp"), requested_date, "kp")
        elif source == "solar_wind":
            requested_date = requested_date.replace(minute=0, second=0, microsecond=0)
            return self._read_single_file(
                os.path.join(self.data_folder, "combined_inputs/solar_wind"), requested_date, "solar_wind"
            )
        else:
            msg = f"Combined input {source} requested not available..."
            logger.error(msg)
            raise RuntimeError(msg)
