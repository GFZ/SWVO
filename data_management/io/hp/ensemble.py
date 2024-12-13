from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


class HpEnsemble:
    ENV_VAR_NAME = "PLACEHOLDER; SEE DERIVED CLASSES BELOW"
    LABEL = "ensemble"

    def __init__(self, index:str, data_dir: str|Path|None = None):
        self.index = index
        if self.index not in ("hp30", "hp60"):
            msg = "Encountered invalid index: {self.index}. Possible options are: hp30, hp60!"
            raise ValueError(msg)

        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                msg = f"Necessary environment variable {self.ENV_VAR_NAME} not set!"
                raise ValueError(msg)

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)

        logging.info(f"{self.index.upper()} Ensemble data directory: {self.data_dir}")

        if not self.data_dir.exists():
            msg = f"Data directory {self.data_dir} does not exist! Impossible to retrive data!"
            raise FileNotFoundError(msg)

        self.index_number = index[2:]

    def read(self, start_time: datetime, end_time: datetime) -> list:
        if start_time is not None and not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time is not None and not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        if start_time is None:
            start_time = datetime.now(timezone.utc)

        if end_time is None:
            end_time = start_time.replace(tzinfo=timezone.utc) + timedelta(days=3)

        start_date = start_time.replace(microsecond=0, minute=0, second=0)
        str_date = start_date.strftime("%Y%m%dT%H0000")
        file_list = sorted(
            self.data_dir.glob(f"FORECAST_{self.index.upper()}_SWIFT_DRIVEN_swift_{str_date}_ensemble_*.csv")
        )

        data = []

        if len(file_list) == 0:
            msg = f"No {self.index} ensemble file found for requested date {start_date}"
            logging.error(msg)
            raise FileNotFoundError(msg)

        for file in file_list:
            hp_df = pd.read_csv(file, names=["t", self.index])

            hp_df["t"] = pd.to_datetime(hp_df["t"], utc=True)
            hp_df.index = hp_df["t"]
            hp_df = hp_df.drop(labels=["t"], axis=1)

            hp_df = hp_df.truncate(
                before=start_time - timedelta(minutes=int(self.index_number) - 0.01),
                after=end_time + timedelta(minutes=int(self.index_number) + 0.01),
            )

            data.append(hp_df)

        return data


class Hp30Ensemble(HpEnsemble):
    ENV_VAR_NAME = "HP30_ENSEMBLE_FORECAST_DIR"

    def __init__(self, data_dir: str|Path|None = None):
        super().__init__("hp30", data_dir)


class Hp60Ensemble(HpEnsemble):
    ENV_VAR_NAME = "HP60_ENSEMBLE_FORECAST_DIR"

    def __init__(self, data_dir: str|Path|None = None):
        super().__init__("hp60", data_dir)
