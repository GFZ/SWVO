from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from data_management.io.omni import OMNILowRes


class F107OMNI(OMNILowRes):
    def __init__(self, data_dir:str|None=None):
        super().__init__(data_dir=data_dir)

    # data is downloaded along with OMNI data, check file name in parent class

    def read(self, start_time: datetime, end_time: datetime, *, download: bool = False) -> pd.DataFrame:
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
        f107_df = f107_df.truncate(before=start_time - timedelta(hours=23.9999), after=end_time + timedelta(hours=23.9999))

        return f107_df  # noqa: RET504
