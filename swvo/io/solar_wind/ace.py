# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling ACE Solar Wind data.
"""

import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import wget

from swvo.io.utils import sw_mag_propagation

logging.captureWarnings(True)


class SWACE:
    """This is a class for the ACE Solar Wind data.

    Parameters
    ----------
    data_dir : str | Path, optional
        Data directory for the ACE Solar Wind data. If not provided, it will be read from the environment variable

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.


    """

    ENV_VAR_NAME = "RT_SW_ACE_STREAM_DIR"

    URL = "https://services.swpc.noaa.gov/text/"
    NAME_MAG = "ace-magnetometer.txt"
    NAME_SWEPAM = "ace-swepam.txt"

    SWEPAM_FIELDS = ["speed", "proton_density", "temperature"]
    MAG_FIELDS = ["bx_gsm", "by_gsm", "bz_gsm", "bavg"]

    LABEL = "ace"

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"ACE data directory: {self.data_dir}")

    def download_and_process(self, request_time: datetime) -> None:
        """
        Download and process ACE data, splitting data across midnight into appropriate day files.

        Parameters
        ----------
        request_time : datetime
            The time for which the data is requested. Must be in the past and within the last two hours.

        Raises
        ------
        AssertionError
            If the request_time is in the future.
        FileNotFoundError
            If the downloaded files are empty.

        Returns
        -------
        None
        """

        current_time = datetime.now(timezone.utc)

        assert request_time < current_time, "Request time cannot be in the future!"
        # assert request_time < (datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes = 121)), "Request time cannot be in the past!"

        if current_time - request_time > timedelta(hours=2):
            logging.debug("We can only download and process ACE RT data for the last two hours!")
            return

        temporary_dir = Path("./temp_sw_ace_wget")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        try:
            self._download_file(temporary_dir, self.NAME_MAG)
            self._download_file(temporary_dir, self.NAME_SWEPAM)
            logging.debug("Processing file ...")
            processed_df = self._process_single_file(temporary_dir)

            unique_dates = np.unique(processed_df.index.date)

            for date in unique_dates:
                file_path = self.data_dir / f"ACE_SW_NOWCAST_{date.strftime('%Y%m%d')}.csv"

                day_start = datetime.combine(date, datetime.min.time()).replace(tzinfo=timezone.utc)
                day_end = datetime.combine(date, datetime.max.time()).replace(tzinfo=timezone.utc)

                day_data = processed_df[(processed_df.index >= day_start) & (processed_df.index <= day_end)]

                if file_path.exists():
                    logging.debug(f"Found previous file for {date}. Loading and combining ...")
                    previous_df = self._read_single_file(file_path)

                    previous_df.drop("file_name", axis=1, inplace=True)
                    day_data = day_data.combine_first(previous_df)

                logging.debug(f"Saving processed file for {date}")
                day_data.to_csv(file_path, index=True, header=True)

        finally:
            rmtree(temporary_dir)

    def _download_file(self, temporary_dir: Path, file_name: str) -> None:
        logging.debug(f"Downloading file {self.URL + file_name} ...")
        wget.download(self.URL + file_name, str(temporary_dir))

        if os.stat(str(temporary_dir / file_name)).st_size == 0:
            raise FileNotFoundError(f"Error while downloading file: {self.URL + file_name}!")

    def read(
        self,
        start_time: datetime,
        end_time: datetime,
        download: bool = False,
        propagation: bool = False,
    ) -> pd.DataFrame:
        """
        Read ACE data for the specified time range.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read.
        end_time : datetime
            End time of the data to read.
        download : bool, optional
            Download data on the go, defaults to False.
        propagation : bool, optional
            Propagate the data from L1 to near-Earth, defaults to False.

        Returns
        -------
        :class:`pandas.DataFrame`
            ACE data

        Raises
        ------
        AssertionError
            Raises `AssertionError` if the start time is before the end time.
        """

        assert start_time < end_time, "Start time must be before end time!"

        if propagation:
            logging.info("Shiting start day by -1 day to account for propagation")
            start_time = start_time - timedelta(days=1)

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
                hour_now = datetime.now(timezone.utc).hour
                file_date = file_date.replace(hour=hour_now, minute=0, second=0, microsecond=0)
                self.download_and_process(file_date)

            if not file_path.exists():
                warnings.warn(f"File {file_path} not found")
                continue

            df_one_day = self._read_single_file(file_path)
            data_out = df_one_day.combine_first(data_out)

        data_out = data_out.truncate(
            before=start_time - timedelta(minutes=0.999999),
            after=end_time + timedelta(minutes=0.999999),
        )

        if propagation:
            data_out = sw_mag_propagation(data_out)
            data_out["file_name"] = data_out.apply(self._update_filename, axis=1)

        return data_out

    def _get_processed_file_list(self, start_time: datetime, end_time: datetime) -> Tuple[List, List]:
        """Get list of file paths and their corresponding time intervals.

        Parameters
        ----------
        cadence_min : float
            Cadence of the data in minutes.

        Returns
        -------
        Tuple[List, List]
            List of file paths and time intervals.
        """
        file_paths = []
        time_intervals = []

        current_time = datetime(start_time.year, start_time.month, start_time.day, 0, 0, 0)
        end_time = datetime(end_time.year, end_time.month, end_time.day, 0, 0, 0)  # + timedelta(days=1)

        while current_time <= end_time:
            file_path = self.data_dir / f"ACE_SW_NOWCAST_{current_time.strftime('%Y%m%d')}.csv"
            file_paths.append(file_path)

            interval_start = current_time
            interval_end = datetime(current_time.year, current_time.month, current_time.day, 23, 59, 59)

            time_intervals.append((interval_start, interval_end))
            current_time += timedelta(days=1)

        return file_paths, time_intervals

    def _update_filename(self, row: pd.Series) -> str:
        """Update the filename in the row.

        Parameters
        ----------
        row : pd.Series

        Returns
        -------
        str
            Updated filename.
        """
        if pd.isna(row["file_name"]):
            return row["file_name"]

        file_date_str = Path(row["file_name"]).stem.split("_")[-1]
        file_date = pd.to_datetime(file_date_str, format="%Y%m%d").date()
        index_date = row.name.date()
        return "propagated from previous ACE NOWCAST file" if file_date != index_date else row["file_name"]

    def _read_single_file(self, file_path) -> pd.DataFrame:
        """Read ACE file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from ACE file.
        """
        df = pd.read_csv(file_path, header="infer")

        df["t"] = pd.to_datetime(df["t"], utc=True)
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)

        df["file_name"] = file_path
        df.loc[df["bavg"].isna() & df["temperature"].isna(), "file_name"] = None

        return df

    def _process_single_file(self, temporary_dir: Path) -> pd.DataFrame:
        """Process mag and swepam ACE file to a DataFrame.

        Returns
        -------
        pd.DataFrame
            ACE data.
        """
        data_mag = self._process_mag_file(temporary_dir)
        data_swepam = self._process_swepam_file(temporary_dir)

        data = pd.concat([data_swepam, data_mag], axis=1)

        return data

    def _process_mag_file(self, temporary_dir: Path) -> pd.DataFrame:
        """
        Reads magnetic instrument last available real time ACE data.

        Returns
        -------

        pd.DataFrame
            Dataframe with magnetic field components and timestamp sampled every minute.
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
            temporary_dir / self.NAME_MAG,
            comment="#",
            skiprows=2,
            sep=r"\s+",
            names=header_mag,
            dtype={"time": str},
        )

        data_mag["t"] = data_mag.apply(lambda x: self._to_date(x), 1)
        data_mag.index = data_mag["t"]
        data_mag.drop(
            [
                "Discard1",
                "Discard2",
                "year",
                "month",
                "day",
                "time",
                "t",
                "status_mag",
                "lat",
                "lon",
            ],
            axis=1,
            inplace=True,
        )
        for k in ["bx_gsm", "by_gsm", "bz_gsm", "bavg"]:
            mask = data_mag[k] < -999.0
            data_mag.loc[mask, k] = np.nan

        return data_mag

    def _process_swepam_file(self, temporary_dir: Path) -> pd.DataFrame:
        """
        This method reads faraday cup SWEPAM instrument daily file from ACE original data.


        Returns
        -------

        pd.DataFrame
            Dataframe  with solar wind speed, proton density, temperature and timestamp, sampled every minute.
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
            temporary_dir / self.NAME_SWEPAM,
            comment="#",
            skiprows=2,
            sep=r"\s+",
            names=header_sw,
            dtype={"time": str},
        )

        data_sw["t"] = data_sw.apply(lambda x: self._to_date(x), 1)
        data_sw.index = data_sw["t"]
        data_sw.drop(
            ["Discard1", "Discard2", "year", "month", "day", "time", "t", "status_sw"],
            axis=1,
            inplace=True,
        )

        for k in ["proton_density", "speed"]:
            mask = data_sw[k] < -9999.0
            data_sw.loc[mask, k] = np.nan

        mask = data_sw["temperature"] < -99999.0
        data_sw.loc[mask, "temperature"] = np.nan
        data_sw["pdyn"] = 2e-6 * data_sw["proton_density"].values * data_sw["speed"].values ** 2

        return data_sw

    def _to_date(self, x) -> datetime:
        """
        Converts into a proper datetime format.

        Parameters
        ----------
        x : pandas.Series
            A row from the dataframe containing keys: year, month, day, and time.

        Returns
        -------
        datetime
            The converted datetime.
        """

        year = int(x["year"])
        month = int(x["month"])
        day = int(x["day"])
        hour = int(str(x["time"])[0:2])
        minute = int(str(x["time"])[2:4])
        return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
