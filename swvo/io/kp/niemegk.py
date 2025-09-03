# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling Niemegk Kp data.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import List, Tuple, Optional, Union
import warnings

import numpy as np
import pandas as pd
import wget


logging.captureWarnings(True)


class KpNiemegk:
    """A class to handle Niemegk Kp data.

    Parameters
    ----------
    data_dir : str | Path, optional
        Data directory for the Niemegk Kp data. If not provided, it will be read from the environment variable

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.
    """

    ENV_VAR_NAME = "RT_KP_NIEMEGK_STREAM_DIR"

    URL = "https://kp.gfz-potsdam.de/app/files/"
    NAME = "Kp_ap_nowcast.txt"

    DAYS_TO_SAVE_EACH_FILE = 3
    LABEL = "niemegk"

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(
                    f"Necessary environment variable {self.ENV_VAR_NAME} not set!"
                )

            data_dir = os.environ.get(self.ENV_VAR_NAME)

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Kp Niemegk  data directory: {self.data_dir}")

    def download_and_process(
        self, start_time: datetime, end_time: datetime, reprocess_files: bool = False
    ) -> None:
        """Download and process Niemegk Kp data file.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to download and process.
        end_time : datetime
            End time of the data to download and process.
        reprocess_files : bool, optional
            Downloads and processes the files again, defaults to False, by default False

        Raises
        ------
        FileNotFoundError
            Raise `FileNotFoundError` if the file is not downloaded successfully.
        """
        if start_time < datetime.now(timezone.utc) - timedelta(days=30):
            logging.info(
                "We can only download and process a Kp Niemegk file from the last 30 days!"
            )
            return

        temporary_dir = Path("./temp_kp_niemegk_wget")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        try:
            logging.debug(f"Downloading file {self.URL + self.NAME} ...")

            wget.download(self.URL + self.NAME, str(temporary_dir))

            # check if download was successfull
            if os.stat(str(temporary_dir / self.NAME)).st_size == 0:
                raise FileNotFoundError(
                    f"Error while downloading file: {self.URL + self.NAME}!"
                )

            logging.debug("Processing file ...")
            processed_df = self._process_single_file(temporary_dir)

            file_paths, time_intervals = self._get_processed_file_list(
                start_time, end_time
            )

            for file_path, time_interval in zip(file_paths, time_intervals):
                if file_path.exists():
                    if reprocess_files:
                        file_path.unlink()
                    else:
                        continue

                data_single_file = processed_df[
                    (processed_df.index >= time_interval[0])
                    & (processed_df.index <= time_interval[1])
                ]

                if len(data_single_file.index) == 0:
                    continue

                data_single_file.to_csv(file_path, index=True, header=False)

                logging.debug(f"Saving processed file {file_path}")

        finally:
            rmtree(temporary_dir)

    def read(
        self, start_time: datetime, end_time: datetime, download: bool = False
    ) -> pd.DataFrame:
        """Read Niemegk Kp data for the specified time range.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read.
        end_time : datetime
            End time of the data to read.
        download : bool, optional
            Download data on the go, defaults to False.

        Returns
        -------
        :class:`pandas.DataFrame`
            Niemegk Kp dataframe.
        """

        if not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)

        if not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        file_paths, time_intervals = self._get_processed_file_list(start_time, end_time)

        # initialize data frame with NaNs
        t = pd.date_range(
            datetime(start_time.year, start_time.month, start_time.day),
            datetime(end_time.year, end_time.month, end_time.day, 23, 59, 59),
            freq=timedelta(hours=3),
        )
        data_out = pd.DataFrame(index=t)
        data_out.index = data_out.index.tz_localize(timezone.utc)
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

    def _get_processed_file_list(
        self, start_time: datetime, end_time: datetime
    ) -> Tuple[List, List]:
        """Get list of file paths and their corresponding time intervals.

        Returns
        -------
        Tuple[List, List]
            List of file paths and time intervals.
        """
        file_paths = []
        time_intervals = []

        current_time = datetime(
            start_time.year,
            start_time.month,
            start_time.day,
            0,
            0,
            0,
            tzinfo=timezone.utc,
        )
        end_time = datetime(
            end_time.year, end_time.month, end_time.day, 0, 0, 0, tzinfo=timezone.utc
        ) + timedelta(days=1)

        while current_time <= end_time:
            file_path = (
                self.data_dir
                / f"NIEMEGK_KP_NOWCAST_{current_time.strftime('%Y%m%d')}.csv"
            )
            file_paths.append(file_path)

            interval_start = current_time - timedelta(
                days=self.DAYS_TO_SAVE_EACH_FILE - 1
            )
            interval_end = datetime(
                current_time.year,
                current_time.month,
                current_time.day,
                23,
                59,
                59,
                tzinfo=timezone.utc,
            )

            time_intervals.append((interval_start, interval_end))
            current_time += timedelta(days=1)

        return file_paths, time_intervals

    def _read_single_file(self, file_path) -> pd.DataFrame:
        """Read Nimegk Kp file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from Nimegk Kp file.
        """
        df = pd.read_csv(file_path, names=["t", "kp"])

        df["t"] = pd.to_datetime(df["t"])
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)
        if not df.index.tzinfo:
            df.index = df.index.tz_localize(timezone.utc)

        df["file_name"] = file_path
        df.loc[df["kp"].isna(), "file_name"] = None

        return df

    def _process_single_file(self, temporary_dir: Path) -> pd.DataFrame:
        """Process Nimegk Kp file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Nimegk Kp data.
        """

        header = [
            "#YYY",
            "MM",
            "DD",
            "hh.h",
            "hh._m",
            "days",
            "days_m",
            "Kp",
            "ap",
            "D",
        ]

        data = pd.read_csv(
            temporary_dir / self.NAME, names=header, sep=r"\s+", comment="#"
        )
        data["t"] = pd.to_datetime(
            data[["#YYY", "MM", "DD", "hh.h"]].astype(str).agg("-".join, axis=1),
            format="%Y-%m-%d-%H.%f",
        )
        data["kp"] = data["Kp"]
        data.drop(
            labels=header,
            axis=1,
            inplace=True,
        )
        data.index.rename("t", inplace=True)
        data.index = data["t"]
        data.index = data.index.tz_localize(timezone.utc)
        data.drop(labels=["t"], axis=1, inplace=True)
        data.dropna(inplace=True)
        data = data[data["kp"] != -1.0]

        return data
