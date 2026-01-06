# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling OMNI SYM-H data.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from swvo.io.omni import OMNIHighRes

logging.captureWarnings(True)


class SymhOMNI(OMNIHighRes):
    """
    Class for reading SYM-H data from OMNI High Resolution files.
    Inherits the `download_and_process`, other private methods and attributes from OMNIHighRes.
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        """
        Initialize a SymhOMNI object.

        Parameters
        ----------
        data_dir : Path | None
            Data directory for the SYM-H OMNI data. If not provided, it will be read from the environment variable
        """
        super().__init__(data_dir=data_dir)

    def read(
        self,
        start_time: datetime,
        end_time: datetime,
        cadence_min: float = 1,
        download: bool = True,
    ) -> pd.DataFrame:
        """
        Read OMNI SYM-H data for the given time range.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read. Must be timezone-aware.
        end_time : datetime
            End time of the data to read. Must be timezone-aware.
        cadence_min : float, optional
            Cadence of the data in minutes, defaults to 1
        download : bool, optional
            Download data on the go, defaults to True.

        Returns
        -------
        :class:`pandas.DataFrame`
            OMNI SYM-H data.
        """
        assert cadence_min == 1 or cadence_min == 5, (
            "Only 1 or 5 minute cadence can be chosen for high resolution omni data."
        )

        if not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)

        if not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        assert start_time < end_time

        file_paths, _ = self._get_processed_file_list(start_time, end_time, cadence_min)

        t = pd.date_range(
            start=start_time,
            end=end_time,
            freq=timedelta(minutes=cadence_min),
            tz=timezone.utc,
        )
        data_out = pd.DataFrame(index=t)
        data_out["sym-h"] = np.array([np.nan] * len(t))
        data_out["file_name"] = np.array([None] * len(t))

        for file_path in file_paths:
            if not file_path.exists():
                if download:
                    self.download_and_process(start_time, end_time, cadence_min=cadence_min)
                else:
                    warnings.warn(f"File {file_path} not found")
                    continue

            df_one_file = self._read_single_file(file_path)
            data_out = df_one_file.combine_first(data_out)

        data_out = data_out.truncate(
            before=start_time - timedelta(minutes=cadence_min - 0.0000001),
            after=end_time + timedelta(minutes=cadence_min + 0.0000001),
        )

        if all(data_out["sym-h"].isna()):
            return data_out


        return data_out

    def _read_single_file(self, file_path: Path) -> pd.DataFrame:
        """Read yearly OMNI High Resolution file and extract SYM-H data."""

        df = pd.read_csv(file_path)

        df = df[["timestamp", "sym-h"]].copy()


        df["t"] = pd.to_datetime(df["timestamp"], utc=True)
        df.index = df["t"]

        df.drop(columns=["timestamp", "t"], inplace=True)

        df["file_name"] = file_path
        df.loc[df["sym-h"].isna(), "file_name"] = None

        return df
