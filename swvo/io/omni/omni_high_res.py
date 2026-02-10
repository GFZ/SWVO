# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling OMNI high resolution data.
"""

import calendar
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests


class OMNIHighRes:
    """This is a class for the OMNI High Resolution data.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for the OMNI High Resolution data. If not provided, it will be read from the environment variable

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.
    """

    ENV_VAR_NAME = "OMNI_HIGH_RES_STREAM_DIR"

    URL = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

    START_YEAR = 1981
    LABEL = "omni"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir: Path = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"OMNI high resolution data directory: {self.data_dir}")

    def download_and_process(
        self,
        start_time: datetime,
        end_time: datetime,
        cadence_min: float = 1,
        reprocess_files: bool = False,
    ) -> None:
        """Download and process OMNI High Resolution data files.

        Parameters
        ----------
        start_time : datetime
            Start time for data download.
        end_time : datetime
            End time for data download.
        cadence_min : float, optional
            Cadence of the data in minutes, defaults to 1
        reprocess_files : bool, optional
            Downloads and processes the files again, defaults to False, by default False

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            Raises `AssertionError` if the cadence is not 1 or 5 minutes.
        """

        assert cadence_min == 1 or cadence_min == 5, (
            "Only 1 or 5 minute cadence can be chosen for high resolution omni data."
        )

        file_paths, time_intervals = self._get_processed_file_list(start_time, end_time, cadence_min)

        for file_path, time_interval in zip(file_paths, time_intervals):
            if file_path.exists() and not reprocess_files:
                continue

            # Create directory structure if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")

            try:
                data = self._get_data_from_omni(
                    start=time_interval[0],
                    end=time_interval[1],
                    cadence=cadence_min,
                )

                logging.debug("Processing file ...")

                processed_df = self._process_single_month(data)
                processed_df.to_csv(tmp_path, index=True, header=True)
                tmp_path.replace(file_path)

            except Exception as e:
                logging.error(f"Failed to process {file_path}: {e}")
                if tmp_path.exists():
                    tmp_path.unlink()
                    pass
                continue

    def read(
        self,
        start_time: datetime,
        end_time: datetime,
        cadence_min: float = 1,
        download: bool = False,
    ) -> pd.DataFrame:
        """
        Read OMNI High Resolution data for the given time range.

        Parameters
        ----------
        start_time : datetime
            Start time for reading data.
        end_time : datetime
            End time for reading data.
        cadence_min : float, optional
            Cadence of the data in minutes, defaults to 1
        download : bool, optional
            Download data on the go, defaults to False.

        Returns
        -------
        :class:`pandas.DataFrame`
            OMNI High Resolution data.

        Raises
        ------
        AssertionError
            Raises `AssertionError` if the cadence is not 1 or 5 minutes.
        """
        assert cadence_min == 1 or cadence_min == 5, (
            "Only 1 or 5 minute cadence can be chosen for high resolution omni data."
        )

        if not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)

        if not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        if start_time < datetime(self.START_YEAR, 1, 1, tzinfo=timezone.utc):
            logging.warning(
                "Start date chosen falls behind the existing data. Moving start date to first"
                " available mission files..."
            )
            start_time = datetime(self.START_YEAR, 1, 1, tzinfo=timezone.utc)

        assert start_time < end_time

        file_paths, _ = self._get_processed_file_list(start_time, end_time, cadence_min)

        dfs = []

        for file_path in file_paths:
            if not file_path.exists():
                if download:
                    self.download_and_process(start_time, end_time, cadence_min=cadence_min)
                else:
                    logging.warning(f"File {file_path} not found")
                    continue

            dfs.append(self._read_single_file(file_path))

        data_out = pd.concat(dfs, ignore_index=False)

        if not data_out.empty:
            if not data_out.index.tzinfo:
                data_out.index = data_out.index.tz_localize("UTC")

        data_out = data_out.truncate(
            before=start_time - timedelta(minutes=cadence_min - 0.0000001),
            after=end_time + timedelta(minutes=cadence_min + 0.0000001),
        )

        return data_out

    def _get_processed_file_list(
        self, start_time: datetime, end_time: datetime, cadence_min: float
    ) -> Tuple[List, List]:
        """Get list of file paths and their corresponding time intervals.

        Parameters
        ----------
        start_time : datetime
            Start time for the data.
        end_time : datetime
            End time for the data.
        cadence_min : float
            Cadence of the data in minutes.

        Returns
        -------
        Tuple[List, List]
            List of file paths and time intervals.
        """

        file_paths = []
        time_intervals = []

        # Start from the first day of the start_time month
        current_date = start_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Check if end_time is within cadence_min of the next month boundary
        # This ensures we include the next month's file if needed
        end_year = end_time.year
        end_month = end_time.month

        # Calculate next month start
        if end_month == 12:
            next_month_start = datetime(end_year + 1, 1, 1, 0, 0, 0, tzinfo=end_time.tzinfo)
        else:
            next_month_start = datetime(end_year, end_month + 1, 1, 0, 0, 0, tzinfo=end_time.tzinfo)

        time_diff_to_next_month = (next_month_start - end_time).total_seconds() / 3600

        # If end_time is within `cadence_min` of next month, include the next month
        cadence_hours = cadence_min / 60
        include_next_month = time_diff_to_next_month <= cadence_hours

        while current_date <= end_time or (include_next_month and current_date == next_month_start):
            year = current_date.year
            month = current_date.month

            # directory: YYYY/
            year_dir = self.data_dir / f"{year:04d}"

            # Create file path
            file_path = year_dir / f"OMNI_HIGH_RES_{cadence_min}min_{year:04d}{month:02d}.csv"
            file_paths.append(file_path)

            # Create time interval for current month
            interval_start = datetime(year, month, 1, 0, 0, 0, tzinfo=current_date.tzinfo)

            # Get last day of the month
            last_day = calendar.monthrange(year, month)[1]
            interval_end = datetime(year, month, last_day, 23, 59, 59, tzinfo=current_date.tzinfo)

            time_intervals.append((interval_start, interval_end))

            # Move to next month
            if month == 12:
                current_date = current_date.replace(year=year + 1, month=1)
            else:
                current_date = current_date.replace(month=month + 1)

            # Break condition to avoid infinite loop
            if include_next_month and current_date > next_month_start:
                break

        return file_paths, time_intervals

    def _process_single_month(self, data: list[str]) -> pd.DataFrame:
        """Process monthly OMNI High Resolution data to a DataFrame.

        Parameters
        ----------
        data : list[str]
            Raw data lines from the OMNI service.

        Returns
        -------
        pd.DataFrame
            Monthly OMNI High Resolution data.
        """
        header_line = next(line for line in data if line.strip().startswith("YYYY"))
        columns = header_line.split()

        data_lines = [line for line in data if line.strip().startswith(("19", "20"))]

        if not data_lines:
            msg = "DataFrame is empty."
            logging.error(msg)
            raise ValueError(msg)

        df = pd.DataFrame([line.split() for line in data_lines], columns=columns)
        df = df.apply(pd.to_numeric)

        df["timestamp"] = df["YYYY"].map(str).apply(lambda x: x + "-01-01 ") + df["HR"].map(str).apply(
            lambda x: x.zfill(2)
        )
        df["timestamp"] += df["MN"].map(str).apply(lambda x: ":" + x.zfill(2) + ":00")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["timestamp"] = df["timestamp"] + df["DOY"].apply(lambda x: timedelta(days=int(x) - 1))

        df.drop(columns=["YYYY", "HR", "MN", "DOY"], inplace=True)
        df.set_index("timestamp", inplace=True)

        maxes = {
            "bavg": 9999.9,
            "bx_gsm": 9999.9,
            "by_gsm": 9999.9,
            "bz_gsm": 9999.9,
            "speed": 99999.8,
            "proton_density": 999.8,
            "temperature": 9999998.0,
            "pdyn": 99.0,
            "sym-h": 99999.0,
        }

        df.columns = maxes.keys()
        for col, max_val in maxes.items():
            df[col] = df[col].where(df[col] < max_val, other=pd.NA)

        if df.empty:
            msg = "DataFrame is empty after processing the month."
            logging.error(msg)
            raise ValueError(msg)

        return df

    def _read_single_file(self, file_path) -> pd.DataFrame:
        """Read monthly OMNI High Resolution file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from monthly High Resolution file.
        """
        df = pd.read_csv(file_path)

        df["t"] = pd.to_datetime(df["timestamp"], utc=True)
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)
        df.drop(labels=["timestamp"], axis=1, inplace=True)

        nan_mask = df.isna().all(axis=1)
        df["file_name"] = file_path
        df.loc[nan_mask, "file_name"] = None

        return df

    def _get_data_from_omni(self, start: datetime, end: datetime, cadence: int = 1) -> list:
        """
        Fetches data from NASA's OMNIWeb service.

        If an invalid date range error is returned, it automatically finds the
        suggested valid end date and retries the request.
        """

        payload = {
            "activity": "retrieve",
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
        }
        common_vars = {"vars": ["13", "14", "17", "18", "21", "25", "26", "27", "41"]}
        if cadence == 1:
            params = {"res": "min", "spacecraft": "omni_min"}
            payload.update(params)
            payload.update(common_vars)
        elif cadence == 5:
            params = {"res": "5min", "spacecraft": "omni_5min"}
            payload.update(params)
            payload.update(common_vars)

        else:
            msg = f"Invalid cadence: {cadence}. Only 1 or 5 minutes are supported."
            logging.error(msg)
            raise ValueError(msg)
        logging.debug(f"Fetching data from {self.URL} with payload: {payload}")
        response = requests.post(self.URL, data=payload)
        response.raise_for_status()
        data = response.text.splitlines()

        if data and "Error" in data[0]:
            logging.warning("Received an error response from the server.")

            for line in data:
                if "correct range" in line:
                    # Use regex to find the valid date range (e.g., YYYYMMDD - YYYYMMDD)
                    match = re.search(r"correct range: \d{8} - (\d{8})", line)
                    if match:
                        new_end_date_str = match.group(1)
                        new_end_date = datetime.strptime(new_end_date_str, "%Y%m%d")

                        logging.warning(
                            f"Invalid date range detected. Found suggested end date: {new_end_date.strftime('%Y-%m-%d')}"
                        )

                        # Recursively call the function with the original start date and the new end date
                        return self._get_data_from_omni(start=start, end=new_end_date, cadence=cadence)
            msg = f"An unspecified error occurred: {data}"
            logging.error(msg)
            raise ValueError(msg)
        return data
