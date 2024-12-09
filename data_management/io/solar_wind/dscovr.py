import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import List, Tuple

import numpy as np
import pandas as pd
import wget


class DSCOVR:
    ENV_VAR_NAME = "SW_DSCOVR_STREAM_DIR"

    URL = "https://services.swpc.noaa.gov/products/solar-wind/"
    NAME_MAG = "mag-1-day.json"
    NAME_SWEPAM = "plasma-1-day.json"

    SWEPAM_FIELDS = ["speed", "proton_density", "temperature"]
    MAG_FIELDS = ["bx_gsm", "by_gsm", "bz_gsm", "bavg"]

    def __init__(self, data_dir: str | Path = None):
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(
                    f"Necessary environment variable {self.ENV_VAR_NAME} not set!"
                )

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"DSCOVR data directory: {self.data_dir}")

    def download_and_process(self, request_time: datetime, verbose: bool = False):
        """
        Download and process DSCOVR data, splitting data across midnight into appropriate day files.
        """
        current_time = datetime.now(timezone.utc)

        if current_time - request_time > timedelta(hours=24):
            if verbose:
                logging.info(
                    "We can only download DSCOVR data for the last 23 hours and a hour in past!"
                )
            return

        temporary_dir = Path("./temp_sw_dscovr_wget")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        try:
            if verbose:
                logging.info(f"Downloading file {self.URL + self.NAME_MAG} ...")

            wget.download(self.URL + self.NAME_MAG, str(temporary_dir))

            if os.stat(str(temporary_dir / self.NAME_MAG)).st_size == 0:
                raise FileNotFoundError(
                    f"Error while downloading file: {self.URL + self.NAME_MAG}!"
                )

            if verbose:
                logging.info(f"Downloading file {self.URL + self.NAME_SWEPAM} ...")

            wget.download(self.URL + self.NAME_SWEPAM, str(temporary_dir))

            if os.stat(str(temporary_dir / self.NAME_SWEPAM)).st_size == 0:
                raise FileNotFoundError(
                    f"Error while downloading file: {self.URL + self.NAME_SWEPAM}!"
                )

            if verbose:
                logging.info(f"Processing file ...")
            processed_df = self._process_single_file(temporary_dir)

            unique_dates = np.unique(processed_df.index.date)

            for date in unique_dates:
                file_path = (
                    self.data_dir / f"DSCOVR_SW_NOWCAST_{date.strftime('%Y%m%d')}.csv"
                )

                day_start = datetime.combine(date, datetime.min.time()).replace(
                    tzinfo=timezone.utc
                )
                day_end = datetime.combine(date, datetime.max.time()).replace(
                    tzinfo=timezone.utc
                )

                day_data = processed_df[
                    (processed_df.index >= day_start) & (processed_df.index <= day_end)
                ]

                if file_path.exists():
                    if verbose:
                        logging.info(
                            f"Found previous file for {date}. Loading and combining ..."
                        )
                    previous_df = self._read_single_file(file_path)

                    previous_df.drop("file_name", axis=1, inplace=True)
                    day_data = day_data.combine_first(previous_df)

                if verbose:
                    logging.info(f"Saving processed file for {date}")
                day_data.to_csv(file_path, index=True, header=True)

        finally:
            rmtree(temporary_dir)

    def read(
        self, start_time: datetime, end_time: datetime, download: bool = False
    ) -> pd.DataFrame:
        """
        Read DSCOVR data for the specified time range.
        """

        assert start_time < end_time, "Start time must be before end time!"
        file_paths, _ = self._get_processed_file_list(start_time, end_time)

        t = pd.date_range(
            datetime(start_time.year, start_time.month, start_time.day),
            datetime(end_time.year, end_time.month, end_time.day, 23, 59, 59),
            freq=timedelta(minutes=1),
            tz="UTC",
        )
        nan_data = [np.nan] * len(t)
        data_out = pd.DataFrame(
            index=t,
            data={
                "bavg": nan_data,
                "bx_gsm": nan_data,
                "by_gsm": nan_data,
                "bz_gsm": nan_data,
                "proton_density": nan_data,
                "speed": nan_data,
                "temperature": nan_data,
            },
        )

        for file_path in file_paths:
            if not file_path.exists() or download:
                file_date = datetime.strptime(
                    file_path.stem.split("_")[-1], "%Y%m%d"
                ).replace(tzinfo=timezone.utc)
                self.download_and_process(file_date, verbose=False)

            if not file_path.exists():
                logging.warning(f"File {file_path} not found")
                continue

            df_one_day = self._read_single_file(file_path)
            data_out = df_one_day.combine_first(data_out)

        data_out = data_out.truncate(
            before=start_time - timedelta(minutes=0.999999),
            after=end_time + timedelta(minutes=0.999999),
        )

        return data_out

    def _get_processed_file_list(
        self, start_time: datetime, end_time: datetime
    ) -> Tuple[List, List]:
        file_paths = []
        time_intervals = []

        current_time = datetime(
            start_time.year, start_time.month, start_time.day, 0, 0, 0
        )
        end_time = datetime(
            end_time.year, end_time.month, end_time.day, 0, 0, 0
        )  # + timedelta(days=1)

        while current_time <= end_time:
            file_path = (
                self.data_dir
                / f"DSCOVR_SW_NOWCAST_{current_time.strftime('%Y%m%d')}.csv"
            )
            file_paths.append(file_path)

            interval_start = current_time
            interval_end = datetime(
                current_time.year, current_time.month, current_time.day, 23, 59, 59
            )

            time_intervals.append((interval_start, interval_end))
            current_time += timedelta(days=1)

        return file_paths, time_intervals

    def _read_single_file(self, file_path) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        df["t"] = pd.to_datetime(df["t"], utc=True)
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)

        df["file_name"] = file_path
        df.loc[df["bavg"].isna() & df["temperature"].isna(), "file_name"] = None

        return df

    def _process_single_file(self, temporary_dir: Path) -> pd.DataFrame:
        data_mag = self._process_mag_file(temporary_dir)
        data_swepam = self._process_swepam_file(temporary_dir)

        data = pd.concat([data_swepam, data_mag], axis=1)

        start_time = data.index.min()
        end_time = data.index.max()
        complete_range = pd.date_range(
            start=start_time, end=end_time, freq="1min", tz="UTC"
        )

        data = data.reindex(complete_range)
        data.index.name = "t"

        return data

    def _process_mag_file(self, temporary_dir: Path) -> pd.DataFrame:
        """
        Reads magnetic instrument last available real time DSCOVR data.

        :return: A pandas.DataFrame with magnetic field components and timestamp sampled every minute.
        """

        data_mag = pd.read_json(temporary_dir / self.NAME_MAG)
        data_mag.columns = data_mag.iloc[0]
        data_mag = data_mag.iloc[1:].reset_index(drop=True)
        data_mag["t"] = pd.to_datetime(data_mag["time_tag"])
        data_mag.index = data_mag["t"]
        data_mag.index = data_mag.index.tz_localize("UTC")
        data_mag.drop(
            ["lon_gsm", "lat_gsm", "time_tag", "t"],
            axis=1,
            inplace=True,
        )

        data_mag.rename(columns={"bt": "bavg"}, inplace=True)

        return data_mag

    def _process_swepam_file(self, temporary_dir: Path) -> pd.DataFrame:
        """
        This method reads faraday cup SWEPAM instrument daily file from DSCOVR original data.

        :return: A pandas.DataFrame with solar wind speed, proton density, temperature and timestamp,
                 sampled every minute.
        """

        data_plasma = pd.read_json(temporary_dir / self.NAME_SWEPAM)
        data_plasma.columns = data_plasma.iloc[0]
        data_plasma = data_plasma.iloc[1:].reset_index(drop=True)
        data_plasma["t"] = data_plasma["time_tag"]
        data_plasma.index = pd.to_datetime(data_plasma["t"])
        data_plasma.index = data_plasma.index.tz_localize("UTC")
        data_plasma.drop(
            ["time_tag", "t"],
            axis=1,
            inplace=True,
        )

        data_plasma.rename(
            columns={"bt": "bavg", "density": "proton_density"}, inplace=True
        )

        return data_plasma
