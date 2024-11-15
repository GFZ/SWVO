from datetime import datetime, timedelta

import pandas as pd

from data_management.io.omni import OMNILowRes


class F107OMNI(OMNILowRes):

    def __init__(self, data_dir=None):
        super().__init__(data_dir=data_dir)

    def read(self, start_time: datetime, end_time: datetime, download: bool = False) -> pd.DataFrame:

        data_out = super().read(start_time, end_time, download=download)

        df = pd.DataFrame(index=data_out.index)

        df["f107"] = data_out["f107"]

        # we return it just every 24 hours
        df.drop(df[data_out.index.hour % 24 != 0].index, axis=0, inplace=True)
        df.index = df.index.tz_localize("UTC")
        df = df.truncate(before=start_time - timedelta(hours=23.9999), after=end_time + timedelta(hours=23.9999))

        return df
