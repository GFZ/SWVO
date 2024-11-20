import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import List, Tuple
import logging
import numpy as np
import pandas as pd
import wget


class KpSWPC(object):

    ENV_VAR_NAME = "RT_KP_SWPC_STREAM_DIR"

    URL = "https://services.swpc.noaa.gov/text/"
    NAME = "3-day-geomag-forecast.txt"

    def __init__(self, data_dir: str | Path = None):

        if data_dir is None:

            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_and_process(self, target_date: datetime, reprocess_files: bool = False, verbose: bool = False):

        if target_date.date() < datetime.now(timezone.utc).date():
            raise ValueError("We can only download and progress a Kp SWPC file for the current day!")

        file_path = self.data_dir / f"SWPC_KP_FORECAST_{target_date.strftime('%Y%m%d')}.csv"

        if file_path.exists():
            if reprocess_files:
                file_path.unlink()
            else:
                # nothing to do
                return

        temporary_dir = Path("./temp_kp_swpc_wget")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        try:
            if verbose:
                logging.info(f"Downloading file {self.URL + self.NAME} ...")

            wget.download(self.URL + self.NAME, str(temporary_dir))

            if verbose:
                logging.info(f"Processing file ...")
            processed_df = self._process_single_file(temporary_dir)

            processed_df.to_csv(file_path, index=False, header=False)

            if verbose:
                logging.info(f"Saving processed file {file_path}")

        finally:
            rmtree(temporary_dir)

    def read(self, start_time: datetime, end_time: datetime = None, download: bool = False) -> pd.DataFrame:

        if not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time is not None and not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)


        if end_time is None:
            end_time = start_time + timedelta(days=3)

        if (end_time - start_time).days > 3:
            raise ValueError("We can only read 3 days at a time of Kp SWPC!")

        file_path = self.data_dir / f"SWPC_KP_FORECAST_{start_time.strftime('%Y%m%d')}.csv"

        if not file_path.exists():
            if download:
                self.download_and_process(start_time)
            else:
                logging.warning(f"File {file_path} not found")

        data_out = self._read_single_file(file_path)
        data_out.index = data_out.index.tz_localize("UTC")
        data_out = data_out.truncate(before=start_time - timedelta(hours=2.9999), after=end_time + timedelta(hours=2.9999))

        return data_out

    def _read_single_file(self, file_path) -> pd.DataFrame:

        df = pd.read_csv(file_path, names=["t", "kp"])

        df["t"] = pd.to_datetime(df["t"])
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)

        df["file_name"] = file_path
        df.loc[df["kp"].isna(), "file_name"] = None

        return df

    def _process_single_file(self, temporary_dir: Path) -> pd.DataFrame:

        first_line = None
        dates = None
        year = None

        with open(temporary_dir / self.NAME) as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                if i == 1:
                    year = line.split(" ")[1].lstrip().rstrip()
                if "Kp index" in line:
                    first_line = i
                    l_ = lines[i + 1].lstrip().rstrip()
                    dates = l_.split("   ")
                    dates = [year + " " + d.lstrip().rstrip() for d in dates]
                    dates = [datetime.strptime(d, "%Y %b %d") for d in dates]
                    break

        data = pd.read_csv(temporary_dir / self.NAME, skiprows=first_line + 2, sep=r"\s+", names=["0", "1", "2", "3"])
        kp = []
        timestamp = []

        for i in range(1, 4):
            timestamp += [dates[i - 1] + timedelta(hours=3 * j) for j in range(8)]
            kp += list(data[str(i)].values)

        time_in = [timestamp[0]] * 24
        df = pd.DataFrame({"t_forecast": timestamp}, index=time_in)
        df["kp"] = kp

        # change rounded numbers to be equal to 1/3 or 2/3 to be consistent with other Kp products
        df.loc[round(df["kp"] % 1, 2) == 0.67, "kp"] = round(df.loc[round(df["kp"] % 1, 2) == 0.67, "kp"]) + 2 / 3
        df.loc[round(df["kp"] % 1, 2) == 0.33, "kp"] = round(df.loc[round(df["kp"] % 1, 2) == 0.33, "kp"]) + 1 / 3

        df.index.rename("t", inplace=True)

        return df
