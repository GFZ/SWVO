import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import List, Tuple

import numpy as np
import pandas as pd
import wget


class HpGFZ:
    ENV_VAR_NAME = "RT_HP_GFZ_STREAM_DIR"

    START_YEAR = 1985
    URL = "ftp://ftp.gfz-potsdam.de/pub/home/obs/Hpo/"

    def __init__(self, index: str, data_dir: str | Path = None):
        self.index = index
        assert (
            self.index == "hp30" or self.index == "hp60"
        ), "Enountered invalid index: {self.index}. Possible options are: hp30, hp60!"

        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_number = index[2:]

        logging.info(f"{self.index.upper()} GFZ data directory: {self.data_dir}")

        (self.data_dir / str(self.index)).mkdir(exist_ok=True)

    def download_and_process(
        self, start_time: datetime, end_time: datetime, reprocess_files: bool = False, verbose: bool = False
    ):
        temporary_dir = Path("./temp_hp_wget")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        try:
            file_paths, time_intervals = self._get_processed_file_list(start_time, end_time)

            for file_path, time_interval in zip(file_paths, time_intervals):
                filenames_download = [
                    f"Hp{self.index_number}/Hp{self.index_number}_ap{self.index_number}_{time_interval[0].year!s}.txt"
                ]

                # there is a separate nowcast file
                if time_interval[0].year == datetime.now(timezone.utc).year:
                    filenames_download.append(
                        f"Hp{self.index_number}/Hp{self.index_number}_ap{self.index_number}_nowcast.txt"
                    )

                for filename_download in filenames_download:
                    if verbose:
                        logging.info(f"Downloading file {self.URL + filename_download} ...")

                    wget.download(self.URL + filename_download, str(temporary_dir))

                    if verbose:
                        logging.info(f"Processing file ...")

                    if file_path.exists():
                        if reprocess_files:
                            file_path.unlink()
                        else:
                            continue

                filenames_download = [x[5:] for x in filenames_download]  # strip of folder of filename

                processed_df = self._process_single_file(temporary_dir, filenames_download)
                processed_df.to_csv(file_path, index=True, header=False)

        finally:
            rmtree(temporary_dir)

    def read(self, start_time: datetime, end_time: datetime, download: bool = False) -> pd.DataFrame:
        if not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        if start_time < datetime(self.START_YEAR, 1, 1, tzinfo=timezone.utc):
            logging.warning(
                "Start date chosen falls behind the mission starting year. Moving start date to first"
                " available mission files..."
            )
            start_time = datetime(self.START_YEAR, 1, 1, tzinfo=timezone.utc)

        assert start_time < end_time

        file_paths, time_intervals = self._get_processed_file_list(start_time, end_time)

        # initialize data frame with NaNs
        t = pd.date_range(
            datetime(start_time.year, start_time.month, start_time.day),
            datetime(end_time.year, end_time.month, end_time.day, 23, 59, 59),
            freq=timedelta(minutes=int(self.index_number)),
        )

        data_out = pd.DataFrame(index=t)
        data_out[self.index] = np.array([np.nan] * len(t))

        for file_path, time_interval in zip(file_paths, time_intervals):
            logging.info(f"Processing file {file_path} ...")

            if not file_path.expanduser().exists():
                if download:
                    self.download_and_process(start_time, end_time)

            # if we request a date in the future, the file will still not be found here
            if not file_path.expanduser().exists():
                logging.warning(f"File {file_path} not found, filling with NaNs")
                continue
            df_one_file = self._read_single_file(file_path)

            # combine the new file with the old ones, replace all values present in df_one_file in data_out
            data_out = df_one_file.combine_first(data_out)

        data_out.index = data_out.index.tz_localize("UTC")
        data_out = data_out.truncate(
            before=start_time - timedelta(minutes=int(self.index_number) - 0.01),
            after=end_time + timedelta(minutes=int(self.index_number) + 0.01),
        )

        return data_out

    def _get_processed_file_list(self, start_time: datetime, end_time: datetime) -> Tuple[List, List]:
        file_paths = []
        time_intervals = []

        current_time = datetime(start_time.year, 1, 1, 0, 0, 0)
        end_time = datetime(end_time.year, 12, 31, 23, 59, 59)

        while current_time < end_time:
            file_path = self.data_dir / self.index / f"Hp{self.index_number}_GFZ_{current_time.strftime('%Y')}.csv"
            file_paths.append(file_path)

            interval_start = current_time
            interval_end = datetime(current_time.year, 12, 31, 23, 59, 59)

            time_intervals.append((interval_start, interval_end))
            current_time = datetime(current_time.year + 1, 1, 1, 0, 0, 0)

        return file_paths, time_intervals

    def _process_single_file(self, temp_dir, filenames) -> pd.DataFrame:
        data_total = pd.DataFrame()

        # combine nowcast and yearly file
        for filename in filenames:
            data = {self.index: [], "timestamp": []}

            with open(temp_dir / filename) as f:
                for line in f.readlines():
                    if line[0] == "#":
                        continue
                    line = line.split(" ")
                    line = [x for x in line if x != ""]

                    year = line[0]
                    month = line[1]
                    day = line[2]
                    hour = line[3][0:2]

                    if int(line[3][3:4]) == 0:
                        minute = 0
                    elif int(line[3][3:4]) == 5:
                        minute = 30
                    else:
                        raise ValueError("value for minute not expected")
                    data["timestamp"] += [datetime(int(year), int(month), int(day), int(hour), minute)]
                    data[self.index] += [float(line[7])]

            data = pd.DataFrame(data)
            data.index = data["timestamp"]
            data.drop(labels=["timestamp"], axis=1, inplace=True)
            data.loc[data[self.index] == -1, self.index] = np.nan

            data_total = data_total.combine_first(data)

        return data_total

    def _read_single_file(self, file_path) -> pd.DataFrame:
        df = pd.read_csv(file_path, names=["t", str(self.index)])

        df["t"] = pd.to_datetime(df["t"])
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)

        return df


class Hp30GFZ(HpGFZ):
    def __init__(self, data_dir: str | Path = None):
        super().__init__("hp30", data_dir)


class Hp60GFZ(HpGFZ):
    def __init__(self, data_dir: str | Path = None):
        super().__init__("hp60", data_dir)
