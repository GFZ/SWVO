# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling OMNI Dst data.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from swvo.io.omni import OMNILowRes

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


class DSTOMNI(OMNILowRes):
    """
    Class for reading F10.7 data from OMNI DST files.
    Inherits the `download_and_process`, other private methods and attributes from OMNILowRes.
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        """
        Initialize a DSTOMNI object.

        Parameters
        ----------
        data_dir : Path | None
            Data directory for the Dst OMNI data. If not provided, it will be read from the environment variable
        """
        super().__init__(data_dir=data_dir)

    # data is downloaded along with OMNI data, check file name in parent class
    def read(self, start_time: datetime, end_time: datetime, download: bool = False) -> pd.DataFrame:
        """
        Read OMNI DST data for the given time range.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read. Must be timezone-aware.
        end_time : datetime
            End time of the data to read. Must be timezone-aware.
        download : bool, optional
            Download data on the go, defaults to False.

        Returns
        -------
        :class:`pandas.DataFrame`
            OMNI DST data.
        """
        if not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)

        if not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)
        assert start_time < end_time

        file_paths, _ = self._get_processed_file_list(start_time, end_time)

        t = pd.date_range(
            datetime(start_time.year, start_time.month, start_time.day),
            datetime(end_time.year, end_time.month, end_time.day, 23, 00, 00),
            freq=timedelta(hours=1),
            tz=timezone.utc,
        )
        data_out = pd.DataFrame(index=t)
        data_out["dst"] = np.array([np.nan] * len(t))
        data_out["file_name"] = np.array([None] * len(t))

        for file_path in file_paths:
            if not file_path.exists():
                if download:
                    self.download_and_process(start_time, end_time)
                else:
                    warnings.warn(f"File {file_path} not found")
                    continue

            df_one_file = self._read_single_file(file_path)
            data_out = df_one_file.combine_first(data_out)

        data_out = data_out.truncate(
            before=start_time - timedelta(hours=0.9999),
            after=end_time + timedelta(hours=0.9999),
        )
        if all(data_out["dst"].isna()):
            return data_out

        data_out.drop(columns=["timestamp", "t"], inplace=True)

        return data_out

    def _read_single_file(self, file_path: Path) -> pd.DataFrame:
        """Read yearly OMNI DST file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from yearly OMNI DST file.
        """
        df = pd.read_csv(file_path)
        df.drop(columns=["kp", "f107"], inplace=True)

        df["t"] = pd.to_datetime(df["timestamp"], utc=True)
        df.index = df["t"]

        df["file_name"] = file_path
        df.loc[df["dst"].isna(), "file_name"] = None

        return df
