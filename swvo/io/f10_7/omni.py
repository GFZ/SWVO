# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling F10.7 data from OMNI low resolution files.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Union

import numpy as np
import pandas as pd

from swvo.io.omni import OMNILowRes


class F107OMNI(OMNILowRes):
    """
    Class for reading F10.7 data from OMNI low resolution files.
    Inherits the :func:`download_and_process`, other private methods and attributes from :class:`OMNILowRes`.
    """

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize a F107OMNI object.

        Parameters
        ----------
        data_dir : str | None, optional
            Data directory for the OMNI Kp data. If not provided, it will be read from the environment variable
        """
        super().__init__(data_dir=data_dir)

    # data is downloaded along with OMNI data, check file name in parent class
    def read(
        self, start_time: datetime, end_time: datetime, download: bool = False
    ) -> pd.DataFrame:
        """
        Extract F10.7 data from OMNI Low Resolution files.

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
            F10.7 from OMNI Low Resolution data.
        """

        data_out = super().read(start_time, end_time, download=download)

        if not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        f107_df = pd.DataFrame(index=data_out.index)

        f107_df["f107"] = data_out["f107"]
        f107_df["file_name"] = data_out["file_name"]

        # we return it just every 24 hours
        f107_df = f107_df.drop(f107_df[data_out.index.hour % 24 != 0].index, axis=0)
        f107_df = f107_df.replace(999.9, np.nan)
        f107_df = f107_df.truncate(
            before=start_time - timedelta(hours=23.9999),
            after=end_time + timedelta(hours=23.9999),
        )

        return f107_df  # noqa: RET504
