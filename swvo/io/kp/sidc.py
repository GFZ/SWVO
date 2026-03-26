# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling SIDC Kp data.
"""

import logging
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from swvo.io.base import BaseIO
from swvo.io.utils import enforce_utc_timezone

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


class KpSIDC(BaseIO):
    """A class to handle SIDC Kp data.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for the SIDC Kp data. If not provided, it will be read from the environment variable

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.
    """

    ENV_VAR_NAME = "RT_KP_SIDC_STREAM_DIR"

    URL = "https://ssa.sidc.be/prod/API/index.php?component=latest&pc=G158&psc=a"
    NAME = "kp.json"

    LABEL = "sidc"

    def download_and_process(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, reprocess_files: bool = False
    ) -> None:
        """Download and process SIDC Kp data file.

        Parameters
        ----------
        start_time : Optional[datetime]
            Start time of the data to download and process.
        end_time : Optional[datetime]
            End time of the data to download and process.
        reprocess_files : bool, optional
            Downloads and processes the files again, defaults to False, by default False

        Raises
        ------
        FileNotFoundError
            Raise `FileNotFoundError` if the file is not downloaded successfully.
        """

        if start_time is None:
            start_time = datetime.now(timezone.utc)

        if end_time is None:
            end_time = datetime.now(timezone.utc) + timedelta(days=2)

        if start_time >= end_time:
            msg = "start_time must be before end_time"
            logger.error(msg)
            raise ValueError(msg)

        temporary_dir = Path("./temp_kp_sidc_wget")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        logger.debug(f"Downloading file {self.URL} ...")

        file_paths, time_intervals = self._get_processed_file_list(start_time, end_time)
        for num, (file_path, time_interval) in enumerate(zip(file_paths, time_intervals)):
            if file_path.exists() and not reprocess_files:
                continue
            tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
            try:
                # check if download was successfull
                json_file = temporary_dir / f"{self.NAME}"
                self._download(json_file, time_interval[0], time_interval[1])
                if not json_file.exists() or json_file.stat().st_size == 0:
                    raise FileNotFoundError(f"Error while downloading file: {self.URL}!")

                logger.debug("Processing file ...")
                processed_df = self._process_single_file(json_file)
                data_single_file = processed_df[
                    (processed_df.index >= time_interval[0]) & (processed_df.index <= time_interval[1])
                ]

                if len(data_single_file.index) == 0:
                    continue

                file_path.parent.mkdir(parents=True, exist_ok=True)
                data_single_file.to_csv(tmp_path, index=True, header=False)
                tmp_path.replace(file_path)

                logger.debug(f"Saving processed file {file_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                if tmp_path.exists():
                    tmp_path.unlink()
                continue

        rmtree(temporary_dir, ignore_errors=True)

    def _download(self, temporary_dir, start_time: datetime, end_time: datetime) -> None:
        response = requests.get(
            self.URL
            + f"&dts_start={start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}&dts_end={(end_time + timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        response.raise_for_status()

        with open(temporary_dir, "w") as f:
            f.write(response.text)

    def read(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, download: bool = False
    ) -> pd.DataFrame:
        """Read SIDC Kp data for the specified time range.

        Parameters
        ----------
        start_time : Optional[datetime]
            Start time of the data to read.
        end_time : Optional[datetime]
            End time of the data to read.
        download : bool, optional
            Download data on the go, defaults to False.

        Returns
        -------
        :class:`pandas.DataFrame`
            SIDC Kp dataframe.
        """

        if start_time is None:
            start_time = datetime.now(timezone.utc)

        if end_time is None:
            end_time = datetime.now(timezone.utc) + timedelta(days=2)

        if start_time >= end_time:
            msg = "start_time must be before end_time"
            logger.error(msg)
            raise ValueError(msg)

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        file_paths, time_intervals = self._get_processed_file_list(start_time, end_time)

        # initialize data frame with NaNs
        t = pd.date_range(
            datetime(start_time.year, start_time.month, start_time.day),
            datetime(end_time.year, end_time.month, end_time.day, 23, 59, 59),
            freq=timedelta(hours=3),
        )
        data_out = pd.DataFrame(index=t)
        data_out.index = enforce_utc_timezone(data_out.index)  # ty: ignore[no-matching-overload]
        data_out["kp"] = np.array([np.nan] * len(t))
        data_out["file_name"] = np.array([None] * len(t))

        for file_path, time_interval in zip(file_paths, time_intervals):
            if not file_path.exists():
                if download:
                    self.download_and_process(start_time, end_time)

            # if we request a date in the future, the file will still not be found here
            if not file_path.exists():
                warnings.warn(f"File {file_path} not found")
                continue
            df_one_file = self._read_single_file(file_path)

            # combine the new file with the old ones, replace all values present in df_one_file in data_out
            data_out = df_one_file.combine_first(data_out)

        data_out = data_out.truncate(
            before=start_time - timedelta(hours=2.9999),
            after=end_time + timedelta(hours=2.9999),
        )

        return data_out

    def _get_processed_file_list(self, start_time: datetime, end_time: datetime) -> Tuple[List, List]:
        """Get list of file paths and their corresponding time intervals for monthly files.

        Returns
        -------
        Tuple[List, List]
            List of file paths and tuples containing (start_time, end_time) for each month.
        """
        file_paths = []
        time_intervals = []

        # Start from the first day of the month containing start_time
        current_time = datetime(
            start_time.year,
            start_time.month,
            1,
            0,
            0,
            0,
            tzinfo=timezone.utc,
        )

        # End at the last day of the month containing end_time
        if end_time.month == 12:
            end_of_period = datetime(end_time.year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        else:
            end_of_period = datetime(end_time.year, end_time.month + 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        while current_time < end_of_period:
            file_path = (
                self.data_dir / current_time.strftime("%Y") / f"SIDC_KP_FORECAST_{current_time.strftime('%Y%m')}.csv"
            )
            file_paths.append(file_path)

            # Create interval covering the entire month
            month_start = current_time
            if current_time.month == 12:
                month_end = datetime(
                    current_time.year + 1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    tzinfo=timezone.utc,
                )
            else:
                month_end = datetime(
                    current_time.year,
                    current_time.month + 1,
                    1,
                    0,
                    0,
                    0,
                    tzinfo=timezone.utc,
                )
            month_end -= timedelta(seconds=1)  # Set to the last second of the month
            time_intervals.append((month_start, month_end))

            # Move to next month
            if current_time.month == 12:
                current_time = datetime(current_time.year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            else:
                current_time = datetime(current_time.year, current_time.month + 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        return file_paths, time_intervals

    def _read_single_file(self, file_path) -> pd.DataFrame:
        """Read SIDC Kp file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from SIDC Kp file.
        """
        df = pd.read_csv(file_path, names=["t", "kp"])

        df["t"] = pd.to_datetime(df["t"])
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)
        df.index = enforce_utc_timezone(df.index)  # ty: ignore[no-matching-overload]

        df["file_name"] = file_path
        df.loc[df["kp"].isna(), "file_name"] = None

        return df

    def _process_single_file(self, temporary_dir: Path) -> pd.DataFrame:
        """Process SIDC Kp file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            SIDC Kp data.
        """

        data = pd.read_json(temporary_dir)
        data = pd.json_normalize(data["data"])

        data = data.apply(lambda x: pd.to_datetime(x) if x.name != "value" else x)

        data.sort_values(["issue_time", "start_time"], inplace=True)

        data.drop(columns=["end_time"], inplace=True)

        data = data.loc[data.groupby("start_time")["issue_time"].idxmax(), ["start_time", "value"]]
        data.rename(columns={"start_time": "t", "value": "Kp"}, inplace=True)
        data.index = data["t"]
        data.index = enforce_utc_timezone(data.index)
        data.drop(columns=["t"], inplace=True)

        return data
