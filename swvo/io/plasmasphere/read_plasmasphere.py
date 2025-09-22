# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from datetime import datetime, timezone

import pandas as pd


class PlasmaspherePredictionReader:
    """Reads one of the available PAGER plasmasphere density prediction.

    Parameters
    ----------
    folder : str
        The folder where the plasmasphere prediction files are stored.

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
        self.file = None
        self.requested_date = None

    def _check_data_folder(self) -> None:
        """Checks if the data folder exists.

        Raises
        ------
        FileNotFoundError
            If the data folder does not exist.
        """
        if not os.path.exists(self.data_folder):
            msg = f"Data folder {self.data_folder} for WP3 plasma output not found...impossible to retrieve data."
            logging.error(msg)
            raise FileNotFoundError(msg)

    @staticmethod
    def _read_single_file(folder: str, date: datetime) -> pd.DataFrame | None:
        """Read a single file from the specified folder for the given date.

        Parameters
        ----------
        folder : str
            folder where we look for the plasmasphere prediction
        date : datetime
            date of the plasmasphere prediction we want to read

        Returns
        -------
        pd.DataFrame | None
            pandas.DataFrame with the data read from the file, or None if the file does not exist.
        """
        file_name = f"plasmasphere_density_{date.year}-{str(date.month).zfill(2)}-{str(date.day).zfill(2)}-{str(date.hour).zfill(2)}-{str(date.minute).zfill(2)}.csv"

        file_path = os.path.join(folder, file_name)

        if not os.path.isfile(file_path):
            msg = f"No suitable files found in the folder {folder} for the requested date {date}"
            logging.warning(msg)
            return None

        data = pd.read_csv(file_path, parse_dates=["date"])
        data["t"] = data["date"]
        data.drop(labels=["date"], axis=1, inplace=True)
        return data

    def read(self, source: str, requested_date: datetime | None = None) -> pd.DataFrame | None:
        """
        Reads one of the available PAGER plasmasphere density prediction.

        Parameters
        ----------
        source : str
            The source of plasmasphere density product requested. Available only "gfz_plasma".
        requested_date : datetime.datetime or None
            Date of plasma density prediction thar we want to read up to hour precision.

        Raises
        ------
        RuntimeError
            if the sources of data requested is not among the available ones.

        Returns
        -------
        pd.DataFrame or None
            pandas.DataFrame with L, MLT, density and date as columns
        """

        if requested_date is None:
            requested_date = datetime.now(timezone.utc).replace(microsecond=0, minute=0, second=0)

        if source == "gfz_plasma":
            requested_date = requested_date.replace(minute=0, second=0, microsecond=0)
            return self._read_single_file(self.data_folder, requested_date)
        else:
            msg = f"Source {source} requested for reading plasmasphere prediction not available..."
            logging.error(msg)
            raise RuntimeError(msg)
