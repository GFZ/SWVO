# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling SWIFT solar wind ensemble data.
"""

import datetime as dt
import json
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from swvo.io.utils import sw_mag_propagation

logging.captureWarnings(True)


class SWSWIFTEnsemble:
    """
    This is a class for SWIFT ensemble data.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for the SWIFT Ensemble data. If not provided, it will be read from the environment variable

    Methods
    -------
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.
    FileNotFoundError
        Returns `FileNotFoundError` if the data directory does not exist.
    """

    PROTON_MASS = 1.67262192369e-27

    ENV_VAR_NAME = "SWIFT_ENSEMBLE_OUTPUT_DIR"
    LABEL = "swift"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir: Path = Path(data_dir)

        logging.info(f"SWIFT ensemble data directory: {self.data_dir}")

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist! Impossible to retrieve data!")

    def read(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        propagation: bool = False,
        truncate: bool = True,
    ) -> list[pd.DataFrame]:
        # It does not make sense to read SWIFT ensemble files from different dates

        """
        Read SWIFT ensemble data for the requested period.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read. Must be timezone-aware.
        end_time : datetime
            End time of the data to read. Must be timezone-aware.
            If not provided, it defaults to 3 days after the start time.
            If `propagation` is True, it defaults to 2 days after the start time.
            If `propagation` is False, it defaults to 3 days after the start time.
        propagation : bool, optional
            Propagate the data from L1 to near-Earth, defaults to False.
        truncate : bool, optional
            If True, truncate the data to the requested period, defaults to True.

        Returns
        -------
        list[:class:`pandas.DataFrame`]
            A list of data frames containing ensemble data for the requested period.
        """

        if start_time and not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time and not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        if start_time is None:
            start_time = datetime.now(timezone.utc).replace(microsecond=0, minute=0, second=0)

        if end_time is None:
            end_time = start_time.replace(tzinfo=timezone.utc) + timedelta(days=3)

        if propagation:
            logging.info("Shifting start day by -1 day to account for propagation")
            start_time = start_time - timedelta(days=1)

        str_date = start_time.strftime("%Y%m%dt0000")

        ensemble_folders = sorted(
            list((self.data_dir / str_date).glob("*task*")),
            key=lambda x: int(x.stem.split("task")[-1]),
        )

        logging.info(f"Found {len(ensemble_folders)} SWIFT tasks folders...")
        gsm_s = []

        if len(ensemble_folders) == 0:
            msg = f"SWIFT ensemble folder for date {str_date} not found...impossible to read, returning DataFrame with NaNs"
            warnings.warn(msg)
            data_out = self._nan_dataframe(start_time, end_time)
            gsm_s.append(data_out)

        for ensemble_folder in ensemble_folders:
            try:
                file = list((ensemble_folder / "SWIFT").glob("gsm_*"))[0]
                data_gsm = self._read_single_file(file)
                if truncate:
                    data_gsm = data_gsm.truncate(
                        before=start_time - timedelta(minutes=10),
                        after=end_time + timedelta(minutes=10),
                    )

                if propagation:
                    data_gsm = sw_mag_propagation(data_gsm)
                    data_gsm["file_name"] = data_gsm.apply(self._update_filename, axis=1)

                gsm_s.append(data_gsm)
            except IndexError:
                msg = f"GSM SWIFT output file for date {str_date} and task {ensemble_folder} not found...impossible to read"
                warnings.warn(msg)

        return gsm_s

    # def read_single_output(self, target_time: datetime):
    #     pass

    def _read_single_file(self, file_name, use_old_column_names=False) -> pd.DataFrame:
        """
        This function reads one of the two available JSON files of SWIFT output and extracts relevant variables, combining them into a pandas DataFrame.

        Parameters
        ----------
        file_name : str
            The path of the file to read.
        fields : list, optional
            List of fields to extract from the DataFrame. The list needs to contain a subset of available fields. If None, all the fields available are retrieved.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the requested variables.
        """

        with open(file_name) as f:
            data = json.load(f)

        time = list(
            map(
                lambda x: dt.datetime.fromtimestamp(int(x), tz=dt.timezone.utc),
                data["arrays"]["Unix time"]["data"],
            )
        )

        ux = np.array(data["arrays"]["Vx"]["data"]) / 1000.0
        uy = np.array(data["arrays"]["Vy"]["data"]) / 1000.0
        uz = np.array(data["arrays"]["Vz"]["data"]) / 1000.0

        bx = np.array(data["arrays"]["Bx"]["data"]) * 1.0e9
        by = np.array(data["arrays"]["By"]["data"]) * 1.0e9
        bz = np.array(data["arrays"]["Bz"]["data"]) * 1.0e9

        temperature = np.array(data["arrays"]["Temperature_ion"]["data"])

        speed = np.sqrt(ux**2 + uy**2 + uz**2)
        b = np.sqrt(bx**2 + by**2 + bz**2)

        n = np.array(data["arrays"]["Rho"]["data"]) / self.PROTON_MASS * 1.0e-6
        pdyn = 2e-6 * n * speed**2

        if use_old_column_names:
            df = pd.DataFrame(
                {
                    "proton_density": n,
                    "speed": speed,
                    "b": b,
                    "temperature": temperature,
                    "bx": bx,
                    "by": by,
                    "bz": bz,
                    "ux": ux,
                    "uy": uy,
                    "uz": uz,
                    "pdyn": pdyn,
                },
                index=time,
            )
        else:
            df = pd.DataFrame(
                {
                    "proton_density": n,
                    "speed": speed,
                    "bavg": b,
                    "temperature": temperature,
                    "bx_gsm": bx,
                    "by_gsm": by,
                    "bz_gsm": bz,
                    "pdyn": pdyn,
                },
                index=time,
            )

        df["file_name"] = file_name

        return df

    def _update_filename(self, row: pd.Series) -> str:
        """Update the filename in the row.

        Parameters
        ----------
        row : pd.Series

        Returns
        -------
        str
            Updated filename.
        """

        if pd.isna(row["file_name"]):
            return row["file_name"]

        file_date_str = Path(row["file_name"]).stem.split("_")[-1]
        file_date = pd.to_datetime(file_date_str, format="%Y-%m-%dt0000").date()
        index_date = row.name.date()
        return "propagated from previous SWIFT FORECAST file" if file_date != index_date else row["file_name"]

    def _nan_dataframe(self, start_time, end_time):
        t = pd.date_range(start_time, end_time, freq="5min", tz=timezone.utc)
        data_out = pd.DataFrame(
            {
                "proton_density": [np.nan] * len(t),
                "speed": [np.nan] * len(t),
                "bavg": [np.nan] * len(t),
                "temperature": [np.nan] * len(t),
                "bx_gsm": [np.nan] * len(t),
                "by_gsm": [np.nan] * len(t),
                "bz_gsm": [np.nan] * len(t),
                "pdyn": [np.nan] * len(t),
                "file_name": [np.nan] * len(t),
            },
            index=t,
        )
        return data_out
