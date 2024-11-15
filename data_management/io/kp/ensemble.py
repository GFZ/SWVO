import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import pandas as pd


class KpEnsemble(object):

    ENV_VAR_NAME = "KP_ENSEMBLE_OUTPUT_DIR"

    def __init__(self, data_dir: str | Path = None):
        if data_dir is None:

            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist! Impossible to retrive data!")

    def read(self, start_time: datetime, end_time: datetime) -> list:

        if start_time is None:
            start_time = datetime.utcnow().replace(microsecond=0, minute=0, second=0)

        if end_time is None:
            end_time = start_time + timedelta(days=3)

        str_date = start_time.strftime("%Y%m%dT%H0000")

        file_list = sorted(self.data_dir.glob(f"FORECAST_PAGER_SWIFT_swift_{str_date}_ensemble_*.csv"))

        data = []
        for file in file_list:
            df = pd.read_csv(file, names=["t", "kp"])

            df["t"] = pd.to_datetime(df["t"])
            df.index = df["t"]
            df.drop(labels=["t"], axis=1, inplace=True)

            df["file_name"] = file
            df.loc[df["kp"].isna(), "file_name"] = None

            df = df.truncate(before=start_time - timedelta(hours=2.9999), after=end_time + timedelta(hours=2.9999))

            data.append(df)

        return data
