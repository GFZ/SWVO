# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling DSCOVR Solar Wind data.
"""

import json
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


class DSCOVR(BaseIO):
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

    URL = "https://www.ncei.noaa.gov/cloud-access/space-weather-portal/api/v1/values"
    NAME_DATA = "dscovr.json"

    SWEPAM_FIELDS = ["speed", "proton_density", "temperature"]
    MAG_FIELDS = ["bx_gsm", "by_gsm", "bz_gsm", "bavg"]
    MAG_PARAMETER_FIELDS = {
        "bavg": "bt",
        "bx_gsm": "bx_gse",
        "by_gsm": "by_gse",
        "bz_gsm": "bz_gse",
    }
    SWEPAM_PARAMETER_FIELDS = {
        "speed": "proton_speed",
        "proton_density": "proton_density",
        "temperature": "proton_temperature",
    }

    LABEL = "dscovr"

    def download_and_process(self, start_time: datetime, end_time: datetime) -> None:
        """
        Download and process DSCOVR data, splitting data across midnight into appropriate day files.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to download. Must be timezone-aware. If `end_time` is not provided, this is treated
            as a nowcast request time and the previous 24 hours are downloaded.
        end_time : datetime | None
            End time of the data to download. Must be timezone-aware.

        Raises
        ------
        AssertionError
            If the request_time is in the future or if start_time is after end_time.
        FileNotFoundError
            If the downloaded files are empty.
        ValueError
            If the end_time is after 2026-06-30, since DSCOVR data is only available until that date.

        Returns
        -------
        None
        """
        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        if end_time > datetime(2026, 6, 29, 23, 59, 59, tzinfo=timezone.utc):
            raise ValueError(
                "DSCOVR data is only available until 2026-06-29 23:59:59 UTC. Please choose an earlier end_time."
            )

        assert start_time < end_time, "Start time must be before end time!"

        temporary_dir = Path("./temp_sw_dscovr_wget")
        temporary_dir.mkdir(exist_ok=True, parents=True)

        self._download(temporary_dir, start_time, end_time)

        logger.debug("Processing file ...")
        processed_df = self._process_single_file(temporary_dir)

        unique_dates = np.unique(processed_df.index.date)  # ty: ignore[unresolved-attribute]

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
                # if tmp_path.exists():
                # tmp_path.unlink()
                continue

        rmtree(temporary_dir, ignore_errors=True)

    def _download(self, temporary_dir: Path, start_time: datetime, end_time: datetime) -> None:
        """Download a DSCOVR data file from the NOAA NCEI Space Weather Portal API.

        Parameters
        ----------
        temporary_dir : Path
            Temporary directory to store the downloaded file.
        start_time : datetime
            Start time of the data to download.
        end_time : datetime
            End time of the data to download.

        Raises
        ------
        requests.HTTPError
            If the HTTP request fails.
        FileNotFoundError
            If the downloaded file is empty.
        """
        params = {
            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "parameters": ";".join(self._portal_parameters()),
            "format": "json",
            "time_format": "iso",
        }
        logger.debug(f"Downloading DSCOVR data from {self.URL} for {start_time} - {end_time} ...")
        response = requests.get(self.URL, params=params, timeout=30)
        response.raise_for_status()

        with open(temporary_dir / self.NAME_DATA, "wb") as f:
            f.write(response.content)

        if (temporary_dir / self.NAME_DATA).stat().st_size == 0:
            raise FileNotFoundError(f"Error while downloading file: {self.URL}!")

    def _portal_parameters(self) -> list[str]:
        """Build the parameter list for the NCEI portal values API."""
        mag_parameters = [f"DSCOVR:m1m_dscovr:{field}" for field in self.MAG_PARAMETER_FIELDS.values()]
        swepam_parameters = [f"DSCOVR:f1m_dscovr:{field}" for field in self.SWEPAM_PARAMETER_FIELDS.values()]
        return mag_parameters + swepam_parameters

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
            Raises `AssertionError` if the end time is before the start time.
        """
        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        if propagation:
            logger.info("Shifting start day by -1 day to account for propagation")
            start_time = start_time - timedelta(days=1)
        assert start_time < end_time, "Start time must be before end time!"

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
        """Process combined MAG and SWEPAM DSCOVR data to a DataFrame.

        Returns
        -------
        pd.DataFrame
            DSCOVR data.
        """
        with open(temporary_dir / self.NAME_DATA, "r") as file:
            payload = json.load(file)

        status_code = payload.get("status", {}).get("code")
        if status_code is not None and status_code != 200:
            msg = payload.get("status", {}).get("message", f"NOAA DSCOVR API returned status code {status_code}.")
            logger.error(msg)
            raise FileNotFoundError(msg)

        data = pd.DataFrame(payload.get("data", {}))

        if len(data["time"]) == 0:
            raise FileNotFoundError("No data found in the downloaded DSCOVR file.")

        data["t"] = pd.to_datetime(data["time"], utc=True)
        data.index = data["t"]
        data.drop(["time", "t"], axis=1, inplace=True)

        data.rename(columns=self._portal_column_map(), inplace=True)
        expected_columns = [*self.MAG_FIELDS, *self.SWEPAM_FIELDS]
        for column in expected_columns:
            if column not in data.columns:
                data[column] = np.nan
        data = data[expected_columns].apply(pd.to_numeric, errors="coerce")
        self._replace_invalid_values(data)

        start_time = data.index.min()
        end_time = data.index.max()
        complete_range = pd.date_range(start=start_time, end=end_time, freq="1min", tz="UTC")

        data = data.reindex(complete_range)
        data.index.name = "t"
        data["pdyn"] = 2e-6 * data["proton_density"].values * data["speed"].values ** 2

        return data

    def _portal_column_map(self) -> dict[str, str]:
        """Map NCEI portal response columns to SWVO DSCOVR columns."""
        mag_map = {f"m1m_dscovr.{source}": target for target, source in self.MAG_PARAMETER_FIELDS.items()}
        swepam_map = {f"f1m_dscovr.{source}": target for target, source in self.SWEPAM_PARAMETER_FIELDS.items()}
        return mag_map | swepam_map

    def _replace_invalid_values(self, data: pd.DataFrame) -> None:
        """Replace known DSCOVR missing-value sentinels with NaN."""
        for k in [*self.MAG_FIELDS, *self.SWEPAM_FIELDS]:
            mask = (data[k] < -99999.0) | (data[k] == 0.0)
            data.loc[mask, k] = np.nan

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
