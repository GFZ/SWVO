import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from data_management.io.omni import OMNILowRes


class KpOMNI(OMNILowRes):

    def __init__(self, data_dir=None):
        super().__init__(data_dir=data_dir)

    def read(self, start_time: datetime, end_time: datetime, download: bool = False) -> pd.DataFrame:

        data_out = super().read(start_time, end_time, download=download)
        df = pd.DataFrame(index=data_out.index)

        df["kp"] = data_out["kp"]

        # we return it just every 3 hours
        df.drop(df[data_out.index.hour % 3 != 0].index, axis=0, inplace=True)
        df.index = df.index.tz_localize("UTC")
        df = df.truncate(before=start_time - timedelta(hours=2.9999), after=end_time + timedelta(hours=2.9999))

        return df
