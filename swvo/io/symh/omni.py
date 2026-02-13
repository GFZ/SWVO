# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling OMNI SYM-H data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from swvo.io.omni import OMNIHighRes
from swvo.io.utils import enforce_utc_timezone

logger = logging.getLogger(__name__)

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
        cadence_min: int = 1,
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
        cadence_min : int, optional
            Cadence of the data in minutes, defaults to 1
        download : bool, optional
            Download data on the go, defaults to True.

        Returns
        -------
        :class:`pandas.DataFrame`
            OMNI SYM-H data.
        """
        data_out = super().read(start_time, end_time, cadence_min=cadence_min, download=download)

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        symh_df = pd.DataFrame(index=data_out.index)

        symh_df["sym-h"] = data_out["sym-h"]
        symh_df["file_name"] = data_out["file_name"]

        symh_df = symh_df.truncate(
            before=start_time - timedelta(minutes=cadence_min - 0.0000001),
            after=end_time + timedelta(minutes=cadence_min + 0.0000001),
        )

        return symh_df
