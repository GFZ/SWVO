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
from typing import Optional

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

        file_list_old_name = sorted(
            self.data_dir.glob(f"FORECAST_PAGER_SWIFT_swift_{str_date}_ensemble_*.csv"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )

        file_list_new_name = sorted(
            self.data_dir.glob(f"FORECAST_Kp_{str_date}_ensemble_*.csv"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        data = []

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
