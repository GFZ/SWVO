# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling SWPC Kp data.
"""

import logging
import os
import re
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import Optional, Union

import numpy as np
import pandas as pd
import wget

logging.captureWarnings(True)


class KpSWPC:
    """
    A class for handling SWPC Kp data.
    In SWPC data, the file for current day always contains the forecast for the next 3 days. Keep this in mind when using the `read` and `download_and_process` methods.

    Parameters
    ----------
    data_dir : str | Path, optional
        Data directory for the SWPC Kp data. If not provided, it will be read from the environment variable

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set
    """

    ENV_VAR_NAME = "RT_KP_SWPC_STREAM_DIR"

    URL = "https://services.swpc.noaa.gov/text/"
    NAME = "3-day-forecast.txt"

    LABEL = "swpc"

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Kp SWPC  data directory: {self.data_dir}")

    def download_and_process(self, target_date: datetime, reprocess_files: bool = False):
        """
        Download and process SWPC Kp data file.

        Parameters
        ----------
        target_date : datetime
            Target date for the Kp data,
        reprocess_files : bool, optional
                        Downloads and processes the files again, defaults to False, by default False

        Raises
        ------
        ValueError
            Raises `ValueError` if the target date is in the past.
        """
        if target_date.date() < datetime.now(timezone.utc).date():
            raise ValueError("We can only download and progress a Kp SWPC file for the current day!")

        file_path = self.data_dir / f"SWPC_KP_FORECAST_{target_date.strftime('%Y%m%d')}.csv"

        if file_path.exists():
            if reprocess_files:
                file_path.unlink()
            else:
                return

        temporary_dir = Path("./temp_kp_swpc_wget")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        try:
            logging.debug(f"Downloading file {self.URL + self.NAME} ...")

            wget.download(self.URL + self.NAME, str(temporary_dir))

            logging.debug("Processing file ...")
            processed_df = self._process_single_file(temporary_dir)

            processed_df.to_csv(file_path, index=False, header=False)

            logging.debug(f"Saving processed file {file_path}")

        finally:
            rmtree(temporary_dir)

    def read(self, start_time: datetime, end_time: datetime = None, download: bool = False) -> pd.DataFrame:
        """
        Read Kp data for the specified time range.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read.
        end_time : datetime, optional
            End time of the data to read. If not provided, it will be set to 3 days after `start_time`.
        download : bool, optional
            Download data on the go, defaults to False.

        Returns
        -------
        pd.DataFrame
            SWPC Kp dataframe.

        Raises
        ------
        ValueError
            Raises `ValueError` if the time range is more than 3 days.
        """
        if not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time is not None and not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        if end_time is None:
            end_time = start_time + timedelta(days=3)

        if (end_time - start_time).days > 3:
            msg = "The difference between `end_time` and `start_time` should be less than 3 days. We can only read 3 days at a time of Kp SWPC!"
            logging.error(msg)
            raise ValueError(msg)

        t = pd.date_range(
            datetime(start_time.year, start_time.month, start_time.day),
            datetime(end_time.year, end_time.month, end_time.day, 23, 59, 59),
            freq=timedelta(hours=3),
        )
        data_out = pd.DataFrame(index=t)
        data_out.index = data_out.index.tz_localize(timezone.utc)
        data_out["kp"] = np.array([np.nan] * len(t))
        data_out["file_name"] = np.array([np.nan] * len(t))

        file_name_time = start_time
        file_path = self.data_dir / f"SWPC_KP_FORECAST_{file_name_time.strftime('%Y%m%d')}.csv"
        if not file_path.exists() and download:
            self.download_and_process(start_time)
        if file_path.exists():
            df_one_file = self._read_single_file(file_path)
            data_out = df_one_file.combine_first(data_out)
        else:
            warnings.warn(f"File {file_path} not found")

        data_out = data_out.truncate(
            before=start_time - timedelta(hours=2.9999),
            after=end_time + timedelta(hours=2.9999),
        )

        return data_out

    def _read_single_file(self, file_path: Path) -> pd.DataFrame:
        """Read Kp file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from Kp file.
        """
        df = pd.read_csv(file_path, names=["t", "kp"])

        df["t"] = pd.to_datetime(df["t"], utc=True)
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)

        df["file_name"] = file_path
        df.loc[df["kp"].isna(), "file_name"] = None

        return df

    def _process_single_file(self, temporary_dir: Path) -> pd.DataFrame:
        """Process Kp file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Kp data.
        """
        first_line = None
        dates = []
        year = None
        kp_data = []

        with open(temporary_dir / self.NAME) as f:
            lines = f.readlines()
            for line in lines:
                if ":Issued:" in line:
                    year = int(re.search(r"(\d{4})", line).group(1))
                    break

            for i, line in enumerate(lines):
                if "NOAA Kp index breakdown" in line:
                    first_line = i + 2
                    break

            headers = lines[first_line].split()
            headers = [headers[i] + " " + headers[i + 1] for i in range(0, len(headers), 2)]
            for d in headers:
                try:
                    if any("Dec" in month for month in headers) and "Jan" in d:
                        parsed_date = self._parse_date(d, year + 1)
                    else:
                        parsed_date = self._parse_date(d, year)
                    dates.append(parsed_date)
                except ValueError:
                    raise

            for line in lines[first_line + 1 : first_line + 9]:
                values = [float(val) for val in line.split()[1:] if re.match(r"^\d+\.\d+$", val)]

                kp_data.append(values)

        kp = []
        timestamp = []
        for i, day in enumerate(dates):
            for j in range(8):
                timestamp.append(day + timedelta(hours=3 * j))
                kp.append(kp_data[j][i])

        time_in = [timestamp[0]] * 24
        df = pd.DataFrame({"t_forecast": timestamp}, index=time_in)
        df["kp"] = kp

        df.loc[round(df["kp"] % 1, 2) == 0.67, "kp"] = round(df.loc[round(df["kp"] % 1, 2) == 0.67, "kp"]) + 2 / 3
        df.loc[round(df["kp"] % 1, 2) == 0.33, "kp"] = round(df.loc[round(df["kp"] % 1, 2) == 0.33, "kp"]) + 1 / 3

        df.index.rename("t", inplace=True)

        return df

    def _parse_date(self, date_str, year):
        return datetime.strptime(f"{year} {date_str}", "%Y %b %d")
