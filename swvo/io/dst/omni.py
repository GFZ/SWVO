# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling OMNI Dst data.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from swvo.io.omni import OMNILowRes


class DSTOMNI(OMNILowRes):
    """
    Class for reading Dst data from OMNI hourly files.
    Inherits the `download_and_process`, other private methods and attributes from OMNILowRes.
    """

    _READ_TIME_PADDING = timedelta(hours=0.9999)

    # data is downloaded along with OMNI data, check file name in parent class
    def read(  # ty: ignore[invalid-method-override]
        self, start_time: datetime, end_time: datetime, download: bool = False
    ) -> pd.DataFrame:
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
        data_out = super().read(start_time, end_time, download=download, variables="dst")
        data_out.index.name = "t"

        return data_out
