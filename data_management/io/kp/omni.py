"""Module holding the reader for reading Kp data from OMNI files."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from data_management.io.omni import OMNILowRes


class KpOMNI(OMNILowRes):
    """Class for reading Kp data from OMNI low resolution files.

    Inherits the download_and_process function from OMNILowRes.
    """

    def __init__(self, data_dir:str|None=None) -> None:
        """Initialize a KpOMNI object.

        :param data_dir: Directory path of the OMNI low resolution data.
        If not set, the data will be read from the directory specified by an environmental variable.
        :type data_dir: str | None, optional
        """
        super().__init__(data_dir=data_dir)

    def read(self, start_time: datetime, end_time: datetime, *, download: bool = False) -> pd.DataFrame:
        """Read Kp OMNI data for a specific time.

        Read Kp data from the OMNI low resultion data product. If the requested years have not been downloaded
        before, the required files are downloaded if download is set to True. The cadence of the data is always
        3 hours.

        :param start_time: Start time of the data request.
        :type start_time: datetime
        :param end_time: End time of the data request.
        :type end_time: datetime
        :param download: Flag whether new data should be downloaded, defaults to False
        :type download: bool, optional
        :return: A data frame holding Kp data for the requested period.
        :rtype: pd.DataFrame
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
        kp_df = kp_df.truncate(before=start_time - timedelta(hours=2.9999), after=end_time + timedelta(hours=2.9999))

        return kp_df  # noqa: RET504
