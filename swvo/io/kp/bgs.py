# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling BGS Kp data.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from swvo.io.base import BaseIO
from swvo.io.utils import enforce_utc_timezone

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


class KpBGS(BaseIO):
    """A class to handle BGS Kp data.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for the BGS Kp data. If not provided, it will be read from the environment variable

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.
    """

    ENV_VAR_NAME = "RT_KP_BGS_STREAM_DIR"

    URL = "https://geomag.bgs.ac.uk/cgi-bin/solar"
    NAME = "kp.html"

    LABEL = "bgs"

    def download_and_process(self, request_time: Optional[datetime] = None, reprocess_files: bool = False) -> None:
        """Download and process BGS Kp data file for a specific month.

        Parameters
        ----------
        request_time : Optional[datetime]
            Time for which to download and process data (month and year are extracted).
        reprocess_files : bool, optional
            Downloads and processes the files again, defaults to False, by default False

        Raises
        ------
        FileNotFoundError
            Raise `FileNotFoundError` if the file is not downloaded successfully.
        """

        if request_time is None:
            request_time = datetime.now(timezone.utc)

        request_time = enforce_utc_timezone(request_time)

        temporary_dir = Path("./temp_kp_bgs_wget")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        logger.debug(f"Downloading file {self.URL} ...")

        file_path = self.data_dir / request_time.strftime("%Y") / f"BGS_KP_FORECAST_{request_time.strftime('%Y%m')}.csv"

        if file_path.exists() and not reprocess_files:
            return

        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        try:
            logger.info(f"Downloading file for {request_time.strftime('%Y-%m')} from {self.URL}")
            html_file = temporary_dir / f"{self.NAME}"
            self._download(html_file, request_time)
            if not html_file.exists() or html_file.stat().st_size == 0:
                raise FileNotFoundError(f"Error while downloading file: {self.URL}!")

            logger.debug("Processing file ...")
            processed_df = self._process_single_file(html_file)

            if len(processed_df.index) == 0:
                return

            file_path.parent.mkdir(parents=True, exist_ok=True)
            processed_df.to_csv(tmp_path, index=True, header=False)
            tmp_path.replace(file_path)

            logger.debug(f"Saving processed file {file_path}")
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            return

        rmtree(temporary_dir, ignore_errors=True)

    def _download(self, temporary_dir: Path, request_time: datetime) -> None:
        """Download BGS Kp data for a specific month.

        Parameters
        ----------
        temporary_dir : Path
            Path to save the temporary downloaded file.
        request_time : datetime
            Time for which to download data (month and year are extracted).
        """

        payload = {
            "name": "not given",
            "solar_geo": "1",
            "month": str(request_time.month),
            "year": str(request_time.year),
        }

        response = requests.post(self.URL, data=payload)
        response.raise_for_status()

        with open(temporary_dir, "w") as f:
            f.write(response.text)

    def read(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, download: bool = False
    ) -> pd.DataFrame:
        """Read BGS Kp data for the specified time range.

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
            BGS Kp dataframe.
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

        logger.info(f"Reading data from {start_time} to {end_time}")

        file_paths, time_intervals = self._get_processed_file_list(start_time, end_time)

        # Download data for every month if download is True
        if download:
            for time_interval in time_intervals:
                self.download_and_process(time_interval[0])

        # initialize data frame with NaNs
        t = pd.date_range(
            datetime(start_time.year, start_time.month, start_time.day),
            datetime(end_time.year, end_time.month, end_time.day, 23, 59, 59),
            freq=timedelta(hours=3),
        )
        data_out = pd.DataFrame(index=t)
        data_out.index = enforce_utc_timezone(data_out.index)
        data_out["kp"] = np.array([np.nan] * len(t))
        data_out["file_name"] = np.array([None] * len(t))

        for file_path, time_interval in zip(file_paths, time_intervals):
            if not file_path.exists():
                logger.warning(f"File {file_path} not found")
                logger.warning(f"Data not available from {time_interval[0]} to {time_interval[1]}")
                continue
            logger.info(f"Reading data from {file_path}")
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

        current_time = datetime(
            start_time.year,
            start_time.month,
            1,
            0,
            0,
            0,
            tzinfo=timezone.utc,
        )

        if end_time.month == 12:
            end_of_period = datetime(end_time.year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        else:
            end_of_period = datetime(end_time.year, end_time.month + 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        while current_time < end_of_period:
            file_path = (
                self.data_dir / current_time.strftime("%Y") / f"BGS_KP_FORECAST_{current_time.strftime('%Y%m')}.csv"
            )
            file_paths.append(file_path)

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
            month_end -= timedelta(seconds=1)
            time_intervals.append((month_start, month_end))

            if current_time.month == 12:
                current_time = datetime(current_time.year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            else:
                current_time = datetime(current_time.year, current_time.month + 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        return file_paths, time_intervals

    def _read_single_file(self, file_path) -> pd.DataFrame:
        """Read BGS Kp file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from BGS Kp file.
        """
        df = pd.read_csv(file_path, names=["t", "kp"])

        df["t"] = pd.to_datetime(df["t"])
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)
        df.index = enforce_utc_timezone(df.index)

        df["file_name"] = file_path
        df.loc[df["kp"].isna(), "file_name"] = None

        return df

    def _process_single_file(self, temporary_dir: Path) -> pd.DataFrame:
        """Process BGS Kp file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            BGS Kp data.
        """
        with open(temporary_dir, "r") as f:
            content = f.read()
        soup = BeautifulSoup(content, "html.parser")
        table = soup.find("table")

        if not table:
            msg = f"No table found in response from {self.URL}"
            logger.error(msg)
            raise ValueError(msg)

        rows = table.find_all("tr")
        records = []

        for row in rows[1:]:  # skip header row
            cols = row.find_all("td")
            if len(cols) >= 9:
                date = pd.to_datetime(cols[0].text.strip(), format="%d-%m-%y")

                for i, hour in enumerate(range(0, 24, 3)):
                    kp = self._snap_to_kp_scale(int(cols[i + 1].text.strip()) / 10)
                    timestamp = date + pd.Timedelta(hours=hour)
                    records.append({"t": timestamp, "Kp": kp})

        df = pd.DataFrame(records).set_index("t")
        return df

    def _snap_to_kp_scale(self, value: float) -> float:
        KP_SCALE = np.linspace(0, 9, 28)

        idx = np.argmin(np.abs(KP_SCALE - value))
        return round(float(KP_SCALE[idx]), 3)
