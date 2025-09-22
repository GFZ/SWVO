# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling WDC Dst data.
"""

import logging
import os
import re
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import wget

logging.captureWarnings(True)


class DSTWDC:
    """This is a class for the WDC Dst data.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for the WDC Dst data. If not provided, it will be read from the environment variable

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Raises `ValueError` if necessary environment variable is not set.
    """

    ENV_VAR_NAME = "WDC_STREAM_DIR"

    URL = "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/YYYYMM/"
    LABEL = "wdc"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"WDC Dst data directory: {self.data_dir}")

    def download_and_process(self, start_time: datetime, end_time: datetime, reprocess_files: bool = False) -> None:
        """Download and process WDC Dst data files.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to download. Must be timezone-aware.
        end_time : datetime
            End time of the data to download. Must be timezone-aware.
        reprocess_files : bool, optional
            Downloads and processes the files again, defaults to False, by default False

        Returns
        -------
        None
        """

        assert start_time < end_time, "Start time must be before end time"

        temporary_dir = Path("./temp_wdc")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        try:
            file_paths, time_intervals = self._get_processed_file_list(start_time, end_time)

            for file_path, time_interval in zip(file_paths, time_intervals):
                filename = "index.html"

                URL = self.URL.replace("YYYYMM", time_interval.strftime("%Y%m"))

                if file_path.exists():
                    if reprocess_files:
                        file_path.unlink()
                    else:
                        continue

                logging.debug(f"Downloading file {URL + filename} ...")

                wget.download(URL + filename, str(temporary_dir))

                logging.debug("Processing file ...")

                processed_df = self._process_single_file(
                    temporary_dir / filename,
                    year=time_interval.year,
                    month=time_interval.month,
                )
                processed_df.to_csv(file_path, index=True, header=True)

        finally:
            rmtree(temporary_dir, ignore_errors=True)

    def _get_processed_file_list(self, start_time: datetime, end_time: datetime) -> Tuple[List, List]:
        """Get list of file paths and their corresponding time intervals.

        Returns
        -------
        Tuple[List, List]
            List of file paths and time intervals.
        """

        file_paths = []
        time = []

        current_time = datetime(start_time.year, start_time.month, 1)
        if end_time.month == 12 and end_time.month == 12:
            end_time = datetime(end_time.year + 1, 1, 1, 0, 0, 0)
        else:
            end_time = datetime(end_time.year, end_time.month + 1, 1)

        while current_time < end_time:
            file_path = self.data_dir / f"WDC_DST_{current_time.strftime('%Y%m')}.csv"
            file_paths.append(file_path)

            file_time = current_time

            time.append(file_time)
            # Increment the month
            if current_time.month == 12 and current_time.month == 12:
                current_time = datetime(current_time.year + 1, 1, 1, 0, 0, 0)
            else:
                current_time = datetime(current_time.year, current_time.month + 1, 1, 0, 0, 0)

        return file_paths, time

    def _process_single_file(self, file_path: Path, year, month) -> pd.DataFrame:
        """Process yearly WDC Dst file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            YearlyWDC Dst data.

        """
        with open(file_path, "r") as file:
            text = file.read()

        data = text.split("DAY")[-1]
        data = data.split("<!-- vvvvv S yyyymm_part3.html vvvvv -->", 1)[0]
        lines = data.strip().splitlines()

        records = []

        # Skip header and any non-data lines
        for line in lines:
            numbers = re.findall(r"[-+]?\d+", line)
            if not numbers:
                continue
            day = int(numbers[0])
            dst_values = numbers[1:]

            for hour, val in enumerate(dst_values):
                if val.startswith("9999"):
                    continue
                if len(val) > 4:
                    val = val[:4] if not val.startswith("9999") else None
                try:
                    dst = float(val)
                except:  # noqa: E722
                    continue
                dt = datetime(year, month, day, hour)
                records.append({"timestamp": dt, "dst": dst})

        df = pd.DataFrame(records)
        df.reset_index(drop=True, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True) + pd.Timedelta(hours=1)
        df.index = df["timestamp"]
        df.drop(columns=["timestamp"], inplace=True)

        file_path.unlink()

        return df

    def read(self, start_time: datetime, end_time: datetime, download: bool = False) -> pd.DataFrame:
        """
        Read WDC Dst data for the given time range. it always returns the data until the last day of the month or incase of current month, until the current day.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read. Must be timezone-aware.
        end_time : datetime
            End time of the data to read. Must be timezone-aware.
        download : bool, optional
            Download data on the go, defaults to False.

        Returns
        -------
        :class:`pandas.DataFrame`
           WDC Dst data.
        """

        if not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)

        if not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        assert start_time < end_time, "Start time must be before end time!"

        file_paths, _ = self._get_processed_file_list(start_time, end_time)
        t = pd.date_range(
            datetime(start_time.year, start_time.month, start_time.day),
            datetime(end_time.year, end_time.month, end_time.day, 23, 00, 00),
            freq=timedelta(hours=1),
            tz=timezone.utc,
        )
        data_out = pd.DataFrame(index=t)
        data_out["dst"] = np.array([np.nan] * len(t))
        data_out["file_name"] = np.array([None] * len(t))

        for file_path in file_paths:
            if not file_path.exists():
                if download:
                    self.download_and_process(start_time, end_time)
                else:
                    warnings.warn(f"File {file_path} not found")
                    continue

            df_one_file = self._read_single_file(file_path)
            data_out = df_one_file.combine_first(data_out)

        data_out = data_out.truncate(
            before=start_time - timedelta(hours=0.9999),
            after=end_time + timedelta(hours=0.9999),
        )

        return data_out

    def _read_single_file(self, file_path: Path) -> pd.DataFrame:
        """Read yearlyWDC Dst file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from yearly WDC Dst file.
        """
        df = pd.read_csv(file_path)

        df.index = pd.to_datetime(df["timestamp"], utc=True)

        df["file_name"] = file_path
        df.loc[df["dst"].isna(), "file_name"] = None

        return df
