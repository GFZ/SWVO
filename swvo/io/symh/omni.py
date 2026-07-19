# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Simon Mischel
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling OMNI SYM-H data.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from swvo.io.omni import OMNIHighRes


class SymhOMNI(OMNIHighRes):
    """
    Class for reading SYM-H data from OMNI High Resolution files.
    Inherits the `download_and_process`, other private methods and attributes from OMNIHighRes.
    """

    def read(  # ty: ignore[invalid-method-override]
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

        data_out = super().read(
            start_time,
            end_time,
            cadence_min=cadence_min,
            download=download,
            variables="sym-h",
        )

        symh_df = pd.DataFrame(index=data_out.index)

        symh_df["sym-h"] = data_out["sym-h"]
        symh_df["file_name"] = data_out["file_name"]

        return symh_df
