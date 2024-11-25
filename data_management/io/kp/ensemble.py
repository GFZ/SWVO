import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

import pandas as pd
import logging

class KpEnsemble(object):

    ENV_VAR_NAME = "KP_ENSEMBLE_OUTPUT_DIR"

    def __init__(self, data_dir: str | Path = None):
        if data_dir is None:

            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)

        logging.info(f"Kp Ensemble data directory: {self.data_dir}")

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist! Impossible to retrive data!")

    def read(self, start_time: datetime, end_time: datetime) -> list:

        if start_time is not None and not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time is not None and not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        if start_time is None:
            start_time = datetime.now(timezone.utc)

        if end_time is None:
            end_time = start_time.replace(tzinfo=timezone.utc) + timedelta(days=3)

        start_time = start_time.replace(microsecond=0, minute=0, second=0)
        str_date = start_time.strftime("%Y%m%dT%H0000")

        file_list = sorted(self.data_dir.glob(f"FORECAST_PAGER_SWIFT_swift_{str_date}_ensemble_*.csv"), key=lambda x: int(x.stem.split('_')[-1]))

        if len(file_list) == 0:
            msg = f"No ensemble files found for requested date {str_date}"
            logging.error(msg)
            raise FileNotFoundError(msg)
        
        data = []
        for file in file_list:
            df = pd.read_csv(file, names=["t", "kp"])

            df["t"] = pd.to_datetime(df["t"])
            df.index = df["t"]
            df.drop(labels=["t"], axis=1, inplace=True)

            df["file_name"] = file
            df.loc[df["kp"].isna(), "file_name"] = None

            df.index = df.index.tz_localize("UTC")

            df = df.truncate(before=start_time - timedelta(hours=2.9999), after=end_time + timedelta(hours=2.9999))

            data.append(df)

        return data
