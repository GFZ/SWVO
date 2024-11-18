import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import List, Tuple

import logging

import numpy as np
import pandas as pd
import wget


class SWACE(object):

    ENV_VAR_NAME = "RT_SW_ACE_STREAM_DIR"

    URL = "https://services.swpc.noaa.gov/text/"
    NAME_MAG = "ace-magnetometer.txt"
    NAME_SWEPAM = "ace-swepam.txt"

    SWEPAM_FIELDS = ["speed", "proton_density", "temperature"]
    MAG_FIELDS = ["bx_gsm", "by_gsm", "bz_gsm", "bavg"]

    def __init__(self, data_dir: str | Path = None):

        if data_dir is None:

            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_and_process(self, request_time: datetime, verbose: bool = False):
        """
        Download and process ACE data, splitting data across midnight into appropriate day files.
        """
        current_time = datetime.now(timezone.utc)

        if current_time - request_time > timedelta(hours=2):
            if verbose:
                print("We can only download and process ACE RT data for the last two hours!")
            return

        temporary_dir = Path("./temp_sw_ace_wget")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        try:
            if verbose:
                print(f"Downloading file {self.URL + self.NAME_MAG} ...")

            wget.download(self.URL + self.NAME_MAG, str(temporary_dir))
            print("")

            if os.stat(str(temporary_dir / self.NAME_MAG)).st_size == 0:
                raise FileNotFoundError(f"Error while downloading file: {self.URL + self.NAME_MAG}!")

            if verbose:
                print(f"Downloading file {self.URL + self.NAME_SWEPAM} ...")

            wget.download(self.URL + self.NAME_SWEPAM, str(temporary_dir))
            print("")

            if os.stat(str(temporary_dir / self.NAME_SWEPAM)).st_size == 0:
                raise FileNotFoundError(f"Error while downloading file: {self.URL + self.NAME_SWEPAM}!")

            if verbose:
                print(f"Processing file ...")
            processed_df = self._process_single_file(temporary_dir)

            unique_dates = np.unique(processed_df.index.date)

            for date in unique_dates:
                file_path = self.data_dir / f"ACE_SW_NOWCAST_{date.strftime('%Y%m%d')}.csv"

                day_start = datetime.combine(date, datetime.min.time()).replace(tzinfo=timezone.utc)
                day_end = datetime.combine(date, datetime.max.time()).replace(tzinfo=timezone.utc)

                day_data = processed_df[(processed_df.index >= day_start) & (processed_df.index <= day_end)]

                if file_path.exists():
                    if verbose:
                        print(f"Found previous file for {date}. Loading and combining ...")
                    previous_df = self._read_single_file(file_path)

                    previous_df.drop("file_name", axis=1, inplace=True)
                    day_data = day_data.combine_first(previous_df)

                if verbose:
                    print(f"Saving processed file for {date}")
                day_data.to_csv(file_path, index=True, header=True)

        finally:
            rmtree(temporary_dir)

    def read(self, start_time: datetime, end_time: datetime, download: bool = False) -> pd.DataFrame:
        """
        Read ACE data for the specified time range.
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
            if not file_path.exists() and download:
                file_date = datetime.strptime(file_path.stem.split("_")[-1], "%Y%m%d").replace(tzinfo=timezone.utc)
                self.download_and_process(file_date, verbose=False)

            if not file_path.exists():
                logging.warning(f"File {file_path} not found")
                continue

            df_one_day = self._read_single_file(file_path)
            data_out = df_one_day.combine_first(data_out)

        data_out = data_out.truncate(
            before=start_time - timedelta(minutes=0.999999), after=end_time + timedelta(minutes=0.999999)
        )

        return data_out

    def _get_processed_file_list(self, start_time: datetime, end_time: datetime) -> Tuple[List, List]:

        file_paths = []
        time_intervals = []

        current_time = datetime(start_time.year, start_time.month, start_time.day, 0, 0, 0)
        end_time = datetime(end_time.year, end_time.month, end_time.day, 0, 0, 0) # + timedelta(days=1)

        while current_time <= end_time:

            file_path = self.data_dir / f"ACE_SW_NOWCAST_{current_time.strftime('%Y%m%d')}.csv"
            file_paths.append(file_path)

            interval_start = current_time
            interval_end = datetime(current_time.year, current_time.month, current_time.day, 23, 59, 59)

            time_intervals.append((interval_start, interval_end))
            current_time += timedelta(days=1)

        return file_paths, time_intervals

    def _read_single_file(self, file_path) -> pd.DataFrame:

        df = pd.read_csv(file_path, header="infer")

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

        return data

    def _process_mag_file(self, temporary_dir: Path) -> pd.DataFrame:
        """
        Reads magnetic instrument last available real time ACE data.

        :return: A pandas.DataFrame with magnetic field components and timestamp sampled every minute.
        """

        header_mag = [
            "year",
            "month",
            "day",
            "time",
            "Discard1",
            "Discard2",
            "status_mag",
            "bx_gsm",
            "by_gsm",
            "bz_gsm",
            "bavg",
            "lat",
            "lon",
        ]

        data_mag = pd.read_csv(
            temporary_dir / self.NAME_MAG, comment="#", skiprows=2, sep=r"\s+", names=header_mag, dtype={"time": str}
        )

        data_mag["t"] = data_mag.apply(lambda x: self._to_date(x), 1)
        data_mag.index = data_mag["t"]
        data_mag.drop(
            ["Discard1", "Discard2", "year", "month", "day", "time", "t", "status_mag", "lat", "lon"], axis=1, inplace=True
        )
        for k in ["bx_gsm", "by_gsm", "bz_gsm", "bavg"]:
            mask = data_mag[k] < -999.0
            data_mag.loc[mask, k] = np.nan

        return data_mag

    def _process_swepam_file(self, temporary_dir: Path) -> pd.DataFrame:
        """
        This method reads faraday cup SWEPAM instrument daily file from ACE original data.

        :return: A pandas.DataFrame with solar wind speed, proton density, temperature and timestamp,
                 sampled every minute.
        """
        header_sw = [
            "year",
            "month",
            "day",
            "time",
            "Discard1",
            "Discard2",
            "status_sw",
            "proton_density",
            "speed",
            "temperature",
        ]

        data_sw = pd.read_csv(
            temporary_dir / self.NAME_SWEPAM, comment="#", skiprows=2, sep=r"\s+", names=header_sw, dtype={"time": str}
        )

        data_sw["t"] = data_sw.apply(lambda x: self._to_date(x), 1)
        data_sw.index = data_sw["t"]
        data_sw.drop(["Discard1", "Discard2", "year", "month", "day", "time", "t", "status_sw"], axis=1, inplace=True)

        for k in ["proton_density", "speed"]:
            mask = data_sw[k] < -9999.0
            data_sw.loc[mask, k] = np.nan

        mask = data_sw["temperature"] < -99999.0
        data_sw.loc[mask, "temperature"] = np.nan

        return data_sw

    def _to_date(self, x) -> datetime:
        """
        Converts into a proper datetime format

        :param x: A row from the dataframe containing keys: year, month, day and time
        :type x: pandas.Dataframe Row
        :return: The converted datetime
        """
        year = int(x["year"])
        month = int(x["month"])
        day = int(x["day"])
        hour = int(str(x["time"])[0:2])
        minute = int(str(x["time"])[2:4])
        return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)

    # def fix_historical_files(self):
    #     """
    #     Fixes historical ACE data files by ensuring data is stored in the correct day's file.
    #     Redistributes any data that was incorrectly stored in adjacent day files.
    #     """
    #     ace_files = sorted(list(self.data_dir.glob("ACE_SW_NOWCAST_*.csv")))

    #     if not ace_files:
    #         print("No historical files found to fix.")
    #         return

    #     date_data = {}

    #     for file_path in ace_files:
    #         try:
    #             df = self._read_single_file(file_path)
    #             if "file_name" in df.columns:
    #                 df.drop("file_name", axis=1, inplace=True)

    #             for date, group in df.groupby(df.index.date):
    #                 if date not in date_data:
    #                     date_data[date] = group
    #                 else:
    #                     date_data[date] = group.combine_first(date_data[date])

    #         except Exception as e:
    #             print(f"Error processing {file_path}: {str(e)}")
    #             continue

    #     files_modified = 0
    #     for date, data in date_data.items():
    #         try:
    #             file_path = self.data_dir / f"ACE_SW_NOWCAST_{date.strftime('%Y%m%d')}.csv"

    #             data = data.sort_index()

    #             data.to_csv(file_path, index=True, header=True)
    #             files_modified += 1

    #         except Exception as e:
    #             print(f"Error writing file for date {date}: {str(e)}")
    #             continue

    #     print(f"Successfully processed {len(ace_files)} files and corrected {files_modified} date files.")

    # def fix_and_validate_data(self, start_date: datetime, end_date: datetime):
    #     """
    #     Fixes historical data and validates the results for a specified date range.

    #     Parameters:
    #     -----------
    #     start_date : datetime
    #         Start date for validation
    #     end_date : datetime
    #         End date for validation
    #     """
    #     print("Starting historical data fix...")
    #     self.fix_historical_files()

    #     print("\nValidating fixed files...")

    #     current_date = start_date
    #     while current_date <= end_date:
    #         file_path = self.data_dir / f"ACE_SW_NOWCAST_{current_date.strftime('%Y%m%d')}.csv"

    #         if file_path.exists():
    #             try:
    #                 df = self._read_single_file(file_path)
    #                 if "file_name" in df.columns:
    #                     df.drop("file_name", axis=1, inplace=True)

    #                 incorrect_dates = df.index.date != current_date.date()
    #                 if incorrect_dates.any():
    #                     print(f"Warning: Found {incorrect_dates.sum()} incorrect date entries in {file_path}")
    #                 else:
    #                     print(f"Validated {file_path}: All entries belong to correct date")

    #                 time_gaps = df.index.to_series().diff() > pd.Timedelta(minutes=2)
    #                 if time_gaps.any():
    #                     gap_count = time_gaps.sum()
    #                     print(f"Note: Found {gap_count} time gaps > 2 minutes in {file_path}")

    #             except Exception as e:
    #                 print(f"Error validating {file_path}: {str(e)}")
    #         else:
    #             print(f"No data file exists for {current_date.date()}")

    #         current_date += timedelta(days=1)
