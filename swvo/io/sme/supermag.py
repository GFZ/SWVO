# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Dmitrii Gurev
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling SuperMAG SME data.
"""

import json
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
import requests

from swvo.io.utils import enforce_utc_timezone

logger = logging.getLogger(__name__)

logging.captureWarnings(True)

class SMESuperMAG:
    """Class for SuperMAG SME data.

    Parameters
    ----------
    username : str
        SuperMAG username used for authenticated data access (register at the SuperMAG website to obtain one)
    data_dir : Path | None
        Data directory for the SuperMAG SME data. If not provided, it will be read from the environment variable

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Raised if the required environment variable is not set.
    """

    ENV_VAR_NAME = "SUPERMAG_STREAM_DIR"

    def __init__(self, username: str, data_dir: Optional[Path] = None) -> None:
        self.username = username
        
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                msg = f"Necessary environment variable {self.ENV_VAR_NAME} not set!"
                raise ValueError(msg)
            data_dir = os.environ.get(self.ENV_VAR_NAME)  # ty: ignore[invalid-assignment]

        self.data_dir: Path = Path(data_dir)  # ty:ignore[invalid-argument-type]
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SuperMAG SME data directory: {self.data_dir}")

    def download_and_process(self, start_time: datetime, end_time: datetime, reprocess_files: bool = False) -> None:
        """Download and process SuperMAG SME data files.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to download. Must be timezone-aware.
        end_time : datetime
            End time of the data to download. Must be timezone-aware.
        reprocess_files : bool, optional
            Download and process files again. Defaults to False.

        Returns
        -------
        None
        """

        assert start_time < end_time, "Start time must be before end time"

        temporary_dir = Path("./temp_supermag")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        file_paths, time_intervals = self._get_processed_file_list(start_time, end_time)

        for file_path, time_interval in zip(file_paths, time_intervals):
            if file_path.exists() and not reprocess_files:
                continue

            tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")

            try:
                start_str = time_interval.strftime("%Y-%m-%dT%H:%M")
                extent = int(timedelta(days=1).total_seconds())
                url = (
                    "https://supermag.jhuapl.edu/services/indices.php"
                    f"?python&nohead&logon={self.username}"
                    f"&start={start_str}"
                    f"&extent={extent}"
                    "&indices=sme"
                )

                logger.debug(f"Downloading data from {url} ...")

                response = requests.get(url)
                response.raise_for_status()

                data = response.text.splitlines()
                if data[0].startswith("ERROR"):
                    logger.info(f"SuperMAG {data[0]}")

                filename = "index.html"
                with open(temporary_dir / filename, "w") as file:
                    file.write("\n".join(data))

                logger.debug("Processing file ...")

                processed_df = self._process_single_file(temporary_dir / filename)
                processed_df.to_csv(tmp_path, index=True, header=True)
                tmp_path.replace(file_path)

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                if tmp_path.exists():
                    tmp_path.unlink()
                continue
        
        rmtree(temporary_dir, ignore_errors=True)
    
    def _get_processed_file_list(self, start_time: datetime, end_time: datetime) -> Tuple[List, List]:
        """Get a list of file paths and their corresponding time intervals.

        Returns
        -------
        Tuple[List, List]
            List of file paths and time intervals.
        """

        file_paths = []
        time = []

        current_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = (end_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))

        while current_time < end_time:
            file_path = self.data_dir / f"SuperMAG_SME_{current_time.strftime('%Y%m%d')}.csv"
            file_paths.append(file_path)

            file_time = current_time

            time.append(file_time)
            
            # Increment the day
            current_time = current_time + timedelta(days=1)

        return file_paths, time
    
    def _process_single_file(self, file_path: Path) -> pd.DataFrame:
        """Process daily SuperMAG SME file into a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Processed SuperMAG SME data.

        Raises
        ------
        ValueError
            If no JSON object/array is found in the downloaded file.
        """


        with open(file_path, "r") as file:
            text = file.read()

        match = re.search(r'(\{.*\}|\[.*\])', text, re.S)
        if match is None:
            raise ValueError("No JSON object/array found in file")
        json_text = match.group(1)
        data = json.loads(json_text)

        df = pd.DataFrame(data)

        df["timestamp"] = pd.to_datetime(df["tval"], unit="s", utc=True)
        df.index = df["timestamp"]
        df.drop(columns=["timestamp", "tval"], inplace=True)
        df = df.rename(columns={'SME': 'sme'})

        mask = df["sme"] >= 999998
        df.loc[mask, "sme"] = np.nan

        return df

    def read(self, start_time: datetime, end_time: datetime, download: bool = False) -> pd.DataFrame:
        """
        Read SuperMAG SME data for a given time range.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read. Must be timezone-aware.
        end_time : datetime
            End time of the data to read. Must be timezone-aware.
        download : bool, optional
            Download missing data files on demand. Defaults to False.

        Returns
        -------
        :class:`pandas.DataFrame`
           SuperMAG SME data.
        """
        
        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        assert start_time < end_time, "Start time must be before end time!"

        file_paths, _ = self._get_processed_file_list(start_time, end_time)
        t = pd.date_range(
            datetime(start_time.year, start_time.month, start_time.day),
            datetime(end_time.year, end_time.month, end_time.day, 23, 59, 00),
            freq=timedelta(minutes=1),
            tz=timezone.utc,
        )
        data_out = pd.DataFrame(index=t)
        data_out["sme"] = np.array([np.nan] * len(t))
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
            before=start_time - timedelta(minutes=0.9999),
            after=end_time + timedelta(minutes=0.9999),
        )

        return data_out
    
    def _read_single_file(self, file_path: Path) -> pd.DataFrame:
        """Read a daily SuperMAG SME file into a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from daily SuperMAG SME file.
        """
        df = pd.read_csv(file_path)
        
        df.index = pd.to_datetime(df["timestamp"], utc=True)
        df.drop(columns=["timestamp"], inplace=True)
        df.index.name = None

        df["file_name"] = file_path
        df.loc[df["sme"].isna(), "file_name"] = None

        return df
