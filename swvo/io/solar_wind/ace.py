# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling ACE Solar Wind data.
"""

import logging
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests

from swvo.io.base import BaseIO
from swvo.io.utils import enforce_utc_timezone, sw_mag_propagation

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


class SWACE(BaseIO):
    """This is a class for the ACE Solar Wind data.

    Parameters
    ----------
    data_dir : Path | None
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

    URL = "https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/"
    NAME_MAG = "{date:%Y%m%d}_ace_mag_1m.txt"
    NAME_SWEPAM = "{date:%Y%m%d}_ace_swepam_1m.txt"

    SWEPAM_FIELDS = ["speed", "proton_density", "temperature"]
    MAG_FIELDS = ["bx_gsm", "by_gsm", "bz_gsm", "bavg"]

    LABEL = "ace"

    def download_and_process(self, start_time: datetime, end_time: datetime) -> None:
        """
        Download and process ACE data, splitting data across midnight into appropriate day files.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to download. Must be timezone-aware.
        end_time : datetime
            End time of the data to download. Must be timezone-aware.

        Raises
        ------
        AssertionError
            If the requested interval is invalid or extends into a future UTC date.
        FileNotFoundError
            If the downloaded files are empty.

        Returns
        -------
        None
        """

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        assert start_time < end_time, "Start time must be before end time!"

        temporary_dir = Path("./temp_sw_ace_wget")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        try:
            for date in pd.date_range(start=start_time.date(), end=end_time.date(), freq="D"):
                request_date = date.to_pydatetime().replace(tzinfo=timezone.utc)
                mag_file_name = self._dated_file_name(self.NAME_MAG, request_date)
                swepam_file_name = self._dated_file_name(self.NAME_SWEPAM, request_date)

                self._download(temporary_dir, mag_file_name)
                self._download(temporary_dir, swepam_file_name)

                logger.debug(f"Processing ACE source files for {request_date.date()} ...")
                processed_df = self._process_single_file(temporary_dir, mag_file_name, swepam_file_name)
                self._save_processed_data(processed_df)
        finally:
            rmtree(temporary_dir, ignore_errors=True)

    def _download(self, temporary_dir: Path, file_name: str) -> None:
        """Download a file from ACE server.

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
        response = requests.get(self.URL + file_name, timeout=10)
        response.raise_for_status()

        with open(temporary_dir / file_name, "wb") as f:
            f.write(response.content)

        if (temporary_dir / file_name).stat().st_size == 0:
            raise FileNotFoundError(f"Error while downloading file: {self.URL + file_name}!")

    def _dated_file_name(self, template: str, request_time: datetime) -> str:
        request_time = enforce_utc_timezone(request_time)
        return template.format(date=request_time)

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
        ValueError
            Raises `ValueError` if the start time is after the end time.
        """

        if start_time > end_time:
            msg = "start_time must be before end_time"
            logger.error(msg)
            raise ValueError(msg)

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        if propagation:
            logger.info("Shifting start day by -1 day to account for propagation")
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

        if download and any(not file_path.exists() for file_path in file_paths):
            try:
                self.download_and_process(start_time, end_time)
            except AssertionError as e:
                logger.error(f"`download_and_process` failed because: {e}")

        for file_path in file_paths:
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
                self.data_dir / current_time.strftime("%Y/%m") / f"ACE_SW_NOWCAST_{current_time.strftime('%Y%m%d')}.csv"
            )
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
        index_date = row.name.date()  # ty: ignore[unresolved-attribute]
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

    def _save_processed_data(self, processed_df: pd.DataFrame) -> None:
        unique_dates = np.unique(processed_df.index.date)  # ty: ignore[unresolved-attribute]

        for date in unique_dates:
            file_path = self.data_dir / date.strftime("%Y/%m") / f"ACE_SW_NOWCAST_{date.strftime('%Y%m%d')}.csv"
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

    def _process_single_file(
        self,
        temporary_dir: Path,
        mag_file_name: str | None = None,
        swepam_file_name: str | None = None,
    ) -> pd.DataFrame:
        """Process mag and swepam ACE file to a DataFrame.

        Returns
        -------
        pd.DataFrame
            ACE data.
        """
        data_mag = self._process_mag_file(temporary_dir, mag_file_name)
        data_swepam = self._process_swepam_file(temporary_dir, swepam_file_name)

        data = pd.concat([data_swepam, data_mag], axis=1)

        return data

    def _process_mag_file(self, temporary_dir: Path, file_name: str | None = None) -> pd.DataFrame:
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

        data_file = self._resolve_data_file(temporary_dir, file_name, "*_ace_mag_1m.txt")
        data_mag = pd.read_csv(
            data_file,
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

    def _process_swepam_file(self, temporary_dir: Path, file_name: str | None = None) -> pd.DataFrame:
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

        data_file = self._resolve_data_file(temporary_dir, file_name, "*_ace_swepam_1m.txt")
        data_sw = pd.read_csv(
            data_file,
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

    def _resolve_data_file(self, temporary_dir: Path, file_name: str | None, pattern: str) -> Path:
        if file_name is not None:
            return temporary_dir / file_name

        matches = sorted(temporary_dir.glob(pattern))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Found multiple ACE source files matching {pattern} in {temporary_dir}")
        raise FileNotFoundError(f"No ACE source file matching {pattern} found in {temporary_dir}")

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
