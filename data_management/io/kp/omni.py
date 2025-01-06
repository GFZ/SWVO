"""Module holding the reader for reading Kp data from OMNI files."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from data_management.io.omni import OMNILowRes
from data_management.io.decorators import (
    add_time_docs,
    add_attributes_to_class_docstring,
    add_methods_to_class_docstring,
)


@add_attributes_to_class_docstring
@add_methods_to_class_docstring
class KpOMNI(OMNILowRes):
    """
    Class for reading Kp data from OMNI low resolution files.
    Inherits the `download_and_process`, other private methods and attributes from OMNILowRes.
    """

    def __init__(self, data_dir: str | None = None) -> None:
        """
        Initialize a KpOMNI object.

        Parameters
        ----------
        data_dir : str | None, optional
            Data directory for the OMNI Kp data. If not provided, it will be read from the environment variable
        """
        super().__init__(data_dir=data_dir)

    @add_time_docs("read")
    def read(
        self, start_time: datetime, end_time: datetime, *, download: bool = False
    ) -> pd.DataFrame:
        """
        Extract Kp data from OMNI Low Resolution files.

        Parameters
        ----------
        download : bool, optional
            Download data on the go, defaults to False.

        Returns
        -------
        pd.DataFrame
            Kp data from OMNI Low Resolution data.
        """
        data_out = super().read(start_time, end_time, download=download)
        kp_df = pd.DataFrame(index=data_out.index)

        if not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        kp_df["kp"] = data_out["kp"]
        kp_df["file_name"] = data_out["file_name"]
        # we return it just every 3 hours
        kp_df = kp_df.drop(kp_df[data_out.index.hour % 3 != 0].index, axis=0)
        kp_df = kp_df.truncate(
            before=start_time - timedelta(hours=2.9999),
            after=end_time + timedelta(hours=2.9999),
        )

        return kp_df  # noqa: RET504
