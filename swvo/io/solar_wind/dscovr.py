# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling DSCOVR Solar Wind data.
"""

import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from swvo.io.utils import enforce_utc_timezone, sw_mag_propagation

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


class DSCOVR:
    """This is a class for the DSCOVR Solar Wind data.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for the DSCOVR Solar Wind data. If not provided, it will be read from the environment variable

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.
    """

    ENV_VAR_NAME = "SW_DSCOVR_STREAM_DIR"

    URL = "https://services.swpc.noaa.gov/products/solar-wind/"
    NAME_MAG = "mag-1-day.json"
    NAME_SWEPAM = "plasma-1-day.json"

    SWEPAM_FIELDS = ["speed", "proton_density", "temperature"]
    MAG_FIELDS = ["bx_gsm", "by_gsm", "bz_gsm", "bavg"]

    LABEL = "dscovr"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)  # ty: ignore[invalid-assignment]

        self.data_dir: Path = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DSCOVR data directory: {self.data_dir}")

    def download_and_process(self, request_time: datetime) -> None:
        """
        Download and process DSCOVR data, splitting data across midnight into appropriate day files.

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

        if current_time - request_time > timedelta(hours=24):
            logger.debug("We can only download DSCOVR data for the last 23 hours and a hour in past!")
            return

        temporary_dir = Path("./temp_sw_dscovr_wget")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        self._download(temporary_dir, self.NAME_MAG)
        self._download(temporary_dir, self.NAME_SWEPAM)

        logger.debug("Processing file ...")
        processed_df = self._process_single_file(temporary_dir)

        unique_dates = np.unique(processed_df.index.date)  # ty: ignore[possibly-missing-attribute]

        for date in unique_dates:
            file_path = self.data_dir / date.strftime("%Y/%m") / f"DSCOVR_SW_NOWCAST_{date.strftime('%Y%m%d')}.csv"
            tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")

            try:
                day_start = enforce_utc_timezone(datetime.combine(date, datetime.min.time()))
                day_end = enforce_utc_timezone(datetime.combine(date, datetime.max.time()))

                day_data = processed_df[(processed_df.index >= day_start) & (processed_df.index <= day_end)]

                if file_path.exists():
                    logger.debug(f"Found previous file for {date}. Loading and combining ...")
                    previous_df = self._read_single_file(file_path)

                    previous_df.drop("file_name", axis=1, inplace=True)
                    day_data = day_data.combine_first(previous_df)

                logger.debug(f"Saving processed file for {date}")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                day_data.to_csv(tmp_path, index=True, header=True)
                tmp_path.replace(file_path)

            except Exception as e:
                logger.error(f"Failed to process file for {date}: {e}")
                if tmp_path.exists():
                    tmp_path.unlink()
                continue

        rmtree(temporary_dir, ignore_errors=True)

    def _download(self, temporary_dir: Path, file_name: str) -> None:
        """Download a file from DSCOVR server.

        Parameters
        ----------
        temporary_dir : Path
            Temporary directory to store the downloaded file.
        file_name : str
            Name of the file to download.

        Raises
        ------
        requests.HTTPError
            If the HTTP request fails.
        FileNotFoundError
            If the downloaded file is empty.
        """
        logger.debug(f"Downloading file {self.URL + file_name} ...")
        response = requests.get(self.URL + file_name)
        response.raise_for_status()

        with open(temporary_dir / file_name, "wb") as f:
            f.write(response.content)

        if (temporary_dir / file_name).stat().st_size == 0:
            raise FileNotFoundError(f"Error while downloading file: {self.URL + file_name}!")

    def read(
        self,
        start_time: datetime,
        end_time: datetime,
        download: bool = False,
        propagation: bool = False,
    ) -> pd.DataFrame:
        """
        Read DSCOVR data for the specified time range.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read. Must be timezone-aware.
        end_time : datetime
            End time of the data to read. Must be timezone-aware.
            If not provided, it defaults to 3 days after the start time.
            If `propagation` is True, it defaults to 2 days after the start time.
            If `propagation` is False, it defaults to 3 days after the start time.
        download : bool, optional
            Download data on the go, defaults to False.
        propagation : bool, optional
            Propagate the data from L1 to near-Earth, defaults to False.

        Returns
        -------
        :class:`pandas.DataFrame`
            DataFrame containing DSCOVR Solar Wind data for the requested period.

        Raises
        ------
        AssertionError
            Raises `AssertionError` if the start time is before the end time.
        """
        assert start_time < end_time, "Start time must be before end time!"

        if propagation:
            logger.info("Shiting start day by -1 day to account for propagation")
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
                "pdyn": nan_data,
            },
        )

        for file_path in file_paths:
            if not file_path.exists() and download:
                file_date = enforce_utc_timezone(datetime.strptime(file_path.stem.split("_")[-1], "%Y%m%d"))
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
            file_path = (
                self.data_dir
                / current_time.strftime("%Y/%m")
                / f"DSCOVR_SW_NOWCAST_{current_time.strftime('%Y%m%d')}.csv"
            )
            file_paths.append(file_path)

            interval_start = current_time
            interval_end = datetime(current_time.year, current_time.month, current_time.day, 23, 59, 59)

            time_intervals.append((interval_start, interval_end))
            current_time += timedelta(days=1)

        return file_paths, time_intervals

    def _read_single_file(self, file_path) -> pd.DataFrame:
        """Read DSCOVR file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from DSCOVR file.
        """
        df = pd.read_csv(file_path)

        df["t"] = pd.to_datetime(df["t"], utc=True)
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)

        df["file_name"] = file_path
        df.loc[df["bavg"].isna() & df["temperature"].isna(), "file_name"] = None

        return df

    def _process_single_file(self, temporary_dir: Path) -> pd.DataFrame:
        """Process mag and swepam DSCOVR file to a DataFrame.

        Returns
        -------
        pd.DataFrame
            DSCOVR data.
        """
        data_mag = self._process_mag_file(temporary_dir)
        data_swepam = self._process_swepam_file(temporary_dir)

        data = pd.concat([data_swepam, data_mag], axis=1)

        start_time = data.index.min()
        end_time = data.index.max()
        complete_range = pd.date_range(start=start_time, end=end_time, freq="1min", tz="UTC")

        data = data.reindex(complete_range)
        data.index.name = "t"

        return data

    def _process_mag_file(self, temporary_dir: Path) -> pd.DataFrame:
        """
        Reads magnetic instrument last available real time DSCOVR data.

        Returns
        -------

        pd.DataFrame
            Dataframe with magnetic field components and timestamp sampled every minute.
        """

        data_mag = pd.read_json(temporary_dir / self.NAME_MAG)
        data_mag.columns = data_mag.iloc[0]
        data_mag = data_mag.iloc[1:].reset_index(drop=True)
        data_mag["t"] = pd.to_datetime(data_mag["time_tag"])
        data_mag.index = data_mag["t"]
        data_mag.index = enforce_utc_timezone(data_mag.index)
        data_mag.drop(
            ["lon_gsm", "lat_gsm", "time_tag", "t"],
            axis=1,
            inplace=True,
        )

        data_mag.rename(columns={"bt": "bavg"}, inplace=True)

        return data_mag

    def _process_swepam_file(self, temporary_dir: Path) -> pd.DataFrame:
        """
        This method reads faraday cup SWEPAM instrument daily file from DSCOVR original data.


        Returns
        -------

        pd.DataFrame
            Dataframe  with solar wind speed, proton density, temperature and timestamp, sampled every minute.
        """

        data_plasma = pd.read_json(temporary_dir / self.NAME_SWEPAM)
        data_plasma.columns = data_plasma.iloc[0]
        data_plasma = data_plasma.iloc[1:].reset_index(drop=True)
        data_plasma["t"] = data_plasma["time_tag"]
        data_plasma.index = pd.to_datetime(data_plasma["t"])
        data_plasma.index = enforce_utc_timezone(data_plasma.index)
        data_plasma.drop(
            ["time_tag", "t"],
            axis=1,
            inplace=True,
        )

        data_plasma.rename(columns={"bt": "bavg", "density": "proton_density"}, inplace=True)
        data_plasma = data_plasma.astype(float)
        data_plasma["pdyn"] = 2e-6 * data_plasma["proton_density"].values * data_plasma["speed"].values ** 2

        return data_plasma

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
        index_date = row.name.date()  # ty: ignore[unresolved-attribute]
        return "propagated from previous DSCOVR NOWCAST file" if file_date != index_date else row["file_name"]
