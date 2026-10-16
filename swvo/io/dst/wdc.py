# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling WDC Dst data.
"""

import io
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests

from swvo.io.base import BaseIO
from swvo.io.utils import enforce_utc_timezone, start_of_next_month

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


class DSTWDC(BaseIO):
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

    URL = "https://wdc.kugi.kyoto-u.ac.jp/hapi/"
    # "Best Available" serves, for each period, the highest processing level: final > provisional > real-time
    DATASET = "hour_dst"
    FILL_VALUE = 99999
    LABEL = "wdc"

    def download_and_process(self, start_time: datetime, end_time: datetime, reprocess_files: bool = False) -> None:
        """Download and process WDC Dst data files.

        The data is requested from the HAPI server of the WDC Kyoto (https://wdc.kugi.kyoto-u.ac.jp/hapi/). Its
        "Best Available" dataset serves, for each period, the final Dst if available, otherwise the provisional
        and otherwise the real-time one. A file which was written while only the real-time or provisional values
        existed is not updated automatically, use `reprocess_files=True` to replace it.

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

        Examples
        --------
        >>> reader = DSTWDC(data_dir="/path/to/wdc")
        >>> reader.download_and_process(start_time, end_time)
        """

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        assert start_time < end_time, "Start time must be before end time!"

        self._resolved_urls = []
        file_paths, time_intervals = self._get_processed_file_list(start_time, end_time)

        files = []
        for file_path, time_interval in zip(file_paths, time_intervals):
            if file_path.exists() and not reprocess_files:
                continue
            files.append((file_path, time_interval))

        if not files:
            return

        try:
            processed_df = self._process_hapi_response(self._download(files[0][1], start_of_next_month(files[-1][1])))
        except Exception as e:
            logger.error(f"Failed to download or process WDC Dst from {self.URL}: {e}")
            return

        for file_path, time_interval in files:
            month_end = start_of_next_month(time_interval)
            data_single_file = processed_df[(processed_df.index >= time_interval) & (processed_df.index < month_end)]

            if len(data_single_file.index) == 0:
                logger.warning(f"WDC Dst data not found for {time_interval.strftime('%Y%m')}")
                continue

            tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
            try:
                data_single_file.to_csv(tmp_path, index=True, header=True)
                tmp_path.replace(file_path)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                if tmp_path.exists():
                    tmp_path.unlink()
                continue

    def _download(self, start_time: datetime, end_time: datetime) -> str:
        """Download the WDC Dst of the given range from the HAPI server as CSV.

        The range is clamped to the period covered by the server, since it rejects requests beyond it.
        """
        info_url = f"{self.URL}info?id={self.DATASET}"
        self._record_url(info_url)
        info = requests.get(info_url, timeout=10)
        info.raise_for_status()
        info = info.json()

        available_start = datetime.fromisoformat(info["startDate"].replace("Z", "+00:00"))
        available_end = datetime.fromisoformat(info["stopDate"].replace("Z", "+00:00"))

        start_time = max(start_time, available_start)
        end_time = min(end_time, available_end)
        if start_time >= end_time:
            return ""

        url = (
            f"{self.URL}data?id={self.DATASET}&format=csv"
            f"&start={start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}&stop={end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )
        self._record_url(url)
        logger.debug(f"Downloading data from {url} ...")

        response = requests.get(url, timeout=120)
        response.raise_for_status()

        return response.text

    def _get_processed_file_list(self, start_time: datetime, end_time: datetime) -> Tuple[List, List]:
        """Get list of file paths and their corresponding time intervals.

        Returns
        -------
        Tuple[List, List]
            List of file paths and time intervals.
        """

        file_paths = []
        time = []

        start_time = enforce_utc_timezone(start_time)
        end_time = start_of_next_month(enforce_utc_timezone(end_time))

        current_time = datetime(start_time.year, start_time.month, 1, tzinfo=timezone.utc)

        while current_time < end_time:
            # Create yearly subdirectory
            year_dir = self.data_dir / str(current_time.year)
            year_dir.mkdir(parents=True, exist_ok=True)

            file_path = year_dir / f"WDC_DST_{current_time.strftime('%Y%m')}.csv"
            file_paths.append(file_path)

            file_time = current_time

            time.append(file_time)
            current_time = start_of_next_month(current_time)

        return file_paths, time

    def _process_hapi_response(self, text: str) -> pd.DataFrame:
        """Process the CSV returned by the HAPI server to a DataFrame.

        Parameters
        ----------
        text : str
            CSV with the columns time, Dst and version code. The time is the center of the hour.

        Returns
        -------
        pd.DataFrame
            WDC Dst data, indexed by the start of the hour.
        """
        if not text.strip():
            df = pd.DataFrame({"dst": pd.Series(dtype=float)}, index=pd.DatetimeIndex([], tz="UTC"))
        else:
            df = pd.read_csv(io.StringIO(text), names=["t", "dst", "version"], usecols=["t", "dst"])
            df.index = pd.to_datetime(df["t"], utc=True).dt.floor("h")
            df = df[["dst"]].astype(float)
            df = df[df["dst"] != self.FILL_VALUE]

        df.index.rename("timestamp", inplace=True)

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

        Examples
        --------
        >>> reader = DSTWDC(data_dir="/path/to/wdc")
        >>> reader.read(start_time, end_time, download=True)
        """

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

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
                    logger.warning(f"File {file_path} not found")
                    continue

            if not file_path.exists():
                logger.warning(f"File {file_path} not found, filling with NaNs")
                continue

            df_one_file = self._read_single_file(file_path)
            data_out = df_one_file.combine_first(data_out)

        data_out = data_out.truncate(
            before=start_time - timedelta(hours=0.9999),
            after=end_time + timedelta(hours=0.9999),
        )
        data_out.index.rename("t", inplace=True)
        data_out = data_out[["dst", "file_name"]]

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
        df.index.rename("t", inplace=True)
        df["file_name"] = file_path
        df.loc[df["dst"].isna(), "file_name"] = None

        return df
