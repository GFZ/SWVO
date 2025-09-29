# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling SWIFT Kp ensemble data.
"""

import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logging.captureWarnings(True)


class KpEnsemble:
    """This is a class for Kp ensemble data.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for the Hp data. If not provided, it will be read from the environment variable

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

    ENV_VAR_NAME = "KP_ENSEMBLE_OUTPUT_DIR"
    LABEL = "ensemble"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)

        logging.info(f"Kp Ensemble data directory: {self.data_dir}")

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist! Impossible to retrive data!")

    def read(self, start_time: datetime, end_time: datetime) -> list:
        """Read Kp ensemble data for the requested period.

        Parameters
        ----------
        start_time : datetime
            Start time of the period for which to read the data.
        end_time : datetime
            End time of the period for which to read the data.

        Returns
        -------
        list[:class:`pandas.DataFrame`]
            A list of data frames containing ensemble data for the requested period.

        Raises
        ------
        FileNotFoundError
            Raises `FileNotFoundError` if no ensemble files are found for the requested date.
        """
        # It does not make sense to read KpEnsemble files from different dates
        if start_time is not None and not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time is not None and not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        if start_time is None:
            start_time = datetime.now(timezone.utc)

        if end_time is None:
            end_time = start_time.replace(tzinfo=timezone.utc) + timedelta(days=3)

        start_time = start_time.replace(microsecond=0, minute=0, second=0)
        str_date = start_time.strftime("%Y%m%dT%H0000")

        file_list = self.ensemble_file_list(str_date)

        data = []

        if len(file_list) == 0:
            msg = f"No ensemble files found for requested date {str_date}"
            warnings.warn(f"{msg}! Returning NaNs dataframe.")

            # initialize data frame with NaNs
            t = pd.date_range(
                datetime(start_time.year, start_time.month, start_time.day),
                datetime(end_time.year, end_time.month, end_time.day, 23, 59, 59),
                freq=timedelta(hours=3),
            )
            data_out = pd.DataFrame(index=t)
            data_out.index = data_out.index.tz_localize(timezone.utc)
            data_out["kp"] = np.array([np.nan] * len(t))
            data_out = data_out.truncate(
                before=start_time - timedelta(hours=2.9999),
                after=end_time + timedelta(hours=2.9999),
            )

            data.append(data_out)
            return data

        else:
            for file in file_list:
                df = pd.read_csv(file, names=["t", "kp"])

                df["t"] = pd.to_datetime(df["t"])
                df.index = df["t"]
                df.drop(labels=["t"], axis=1, inplace=True)

                df["file_name"] = file
                df.loc[df["kp"].isna(), "file_name"] = None

                df.index = df.index.tz_localize("UTC")

                df = df.truncate(
                    before=start_time - timedelta(hours=2.9999),
                    after=end_time + timedelta(hours=2.9999),
                )

                data.append(df)

            return data

    def ensemble_file_list(self, str_date: str) -> list[Path]:
        """Check for the existence of ensemble files for a given date.

        Parameters
        ----------
        str_date : str
            Date string in the format YYYYMMDDTHH0000.

        Returns
        -------
        list[Path]
            A list of file paths for the ensemble files.

        Warnings
        --------
        DeprecationWarning
            Warns if deprecated file naming convention is used.
        """

        file_list_old_name = sorted(
            self.data_dir.glob(f"FORECAST_PAGER_SWIFT_swift_{str_date}_ensemble_*.csv"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )

        file_list_new_name = sorted(
            self.data_dir.glob(f"FORECAST_Kp_{str_date}_ensemble_*.csv"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )

        file_list: list[Path]

        if len(file_list_new_name) == 0 and len(file_list_old_name) == 0:
            file_list = []
        elif len(file_list_new_name) > 0:
            file_list = file_list_new_name
        elif len(file_list_old_name) > 0:
            warnings.warn(
                "The use of FORECAST_PAGER_SWIFT_swift_* files is deprecated. However since we still have these files from the PAGER project with this prefix, this will be supported",
                DeprecationWarning,
                stacklevel=2,
            )
            file_list = file_list_old_name
        return file_list

    def read_with_horizon(self, start_time: datetime, end_time: datetime, horizon: int) -> list[pd.DataFrame]:
        """Read Ensemble Kp forecast data for a given time range and forecast horizon.

        Parameters
        ----------
        start_time : datetime
            Start time of the period for which to read the data.
        end_time : datetime
            End time of the period for which to read the data.
        horizon : int
            Forecast horizon (in hours).

        Returns
        -------
        list[:class:`pandas.DataFrame`]
            A list of data frames containing ensemble data for the requested period.
        """
        if start_time is not None and not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time is not None and not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        align_start_to_3hr = start_time.replace(hour=(start_time.hour // 3) * 3, minute=0, second=0, microsecond=0)
        align_end_to_3hr = end_time.replace(hour=(end_time.hour // 3) * 3, minute=0, second=0, microsecond=0)

        full_time_range = pd.date_range(align_start_to_3hr, align_end_to_3hr, freq="3h", tz=timezone.utc)

        file_offsets, time_indices = self._get_file_offsets_and_time_indices(full_time_range, horizon)

        max_ensembles = 30  # Maximum number of ensemble files to check
        ensemble_dfs = [pd.DataFrame(index=full_time_range) for _ in range(max_ensembles)]

        for file_time, file_offset, iloc in zip(full_time_range, file_offsets, time_indices):
            str_date = (file_time + timedelta(hours=file_offset)).strftime("%Y%m%dT%H0000")
            file_list_new_name = sorted(
                self.data_dir.glob(f"FORECAST_Kp_{str_date}_ensemble_*.csv"),
                key=lambda x: int(x.stem.split("_")[-1]),
            )

            for ensemble_idx in range(max_ensembles):
                df = ensemble_dfs[ensemble_idx]
                if ensemble_idx < len(file_list_new_name):
                    data = pd.read_csv(
                        file_list_new_name[ensemble_idx],
                        names=["Forecast Time", "kp"],
                        parse_dates=["Forecast Time"],
                    ).iloc[iloc]
                    data["source"] = str_date
                    data["Forecast Time"] = data["Forecast Time"].tz_localize("UTC")
                    df.loc[file_time, "Forecast Time"] = data["Forecast Time"]
                    df.loc[file_time, "kp"] = data["kp"]
                    df.loc[file_time, "source"] = file_list_new_name[ensemble_idx].stem
                else:
                    df.loc[file_time, "kp"] = np.nan

        for df in ensemble_dfs:
            df["horizon"] = horizon
            df.index.name = "Time"
        ensemble_dfs = [df for df in ensemble_dfs if not df["kp"].isna().all()]

        return ensemble_dfs

    def _get_file_offsets_and_time_indices(
        self, file_time_range: Iterable, forecast_horizon: int
    ) -> tuple[list[int], list[int]]:
        """
        Compute file offsets and time indices for a given forecast horizon.

        Parameters
        ----------
        file_time_range : iterable
            Available file time steps.
        forecast_horizon : int
            Forecast horizon (in hours or chosen unit).

        Returns
        -------
        file_offsets : list[int]
            Offsets from the given time steps indicating which files to read.
        time_indices : list[int]
            Time indices to use for each file.
        """
        file_offsets = []
        time_indices = []

        for _ in file_time_range:
            file_offset = -(forecast_horizon % 3)
            file_offsets.append(file_offset)

            time_index = (forecast_horizon + 2) // 3
            time_indices.append(time_index)

        return file_offsets, time_indices
