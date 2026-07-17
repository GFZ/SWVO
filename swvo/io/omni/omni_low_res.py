# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling OMNI low resolution data.
"""

import logging
import warnings
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests

from swvo.io.base import BaseIO
from swvo.io.omni.variables import LOW_RES_DEFAULT_VARIABLES, LOW_RES_VARIABLES, resolve_variable_names
from swvo.io.utils import enforce_utc_timezone

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


class OMNILowRes(BaseIO):
    """This is a class for the OMNI Low Resolution data.

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.
    """

    ENV_VAR_NAME = "OMNI_LOW_RES_STREAM_DIR"

    URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/"
    LABEL = "omni"

    TIME_COLUMNS = ("year", "day", "hour")
    HEADER = [*TIME_COLUMNS, *(variable.name for variable in LOW_RES_VARIABLES)]

    @classmethod
    def available_variables(cls) -> pd.DataFrame:
        """Return metadata for every hourly OMNI2 output variable."""

        return pd.DataFrame(
            [
                {
                    "name": variable.name,
                    "description": variable.description,
                    "unit": variable.unit,
                    "fill_value": variable.fill_value,
                    "aliases": variable.aliases,
                }
                for variable in LOW_RES_VARIABLES
            ]
        )

    @staticmethod
    def _cache_contains(file_path: Path, variable_names: Iterable[str]) -> bool:
        """Check a processed file's schema without loading its data."""

        if not file_path.exists():
            return False
        try:
            columns = set(pd.read_csv(file_path, nrows=0).columns)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return False
        return set(variable_names).issubset(columns)

    def download_and_process(self, start_time: datetime, end_time: datetime, reprocess_files: bool = False) -> None:
        """Download and process OMNI Low Resolution data files.

        Parameters
        ----------
        start_time : datetime
            Start time for the data to be downloaded and processed.
        end_time : datetime
            End time for the data to be downloaded and processed.
        reprocess_files : bool, optional
            Downloads and processes the files again, defaults to False, by default False

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``start_time`` is not before ``end_time``.
        """

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)
        if start_time >= end_time:
            raise ValueError("start_time must be before end_time")

        file_paths, time_intervals = self._get_processed_file_list(start_time, end_time)
        complete_schema = [variable.name for variable in LOW_RES_VARIABLES]

        with TemporaryDirectory(prefix="swvo-omni-low-res-") as temporary_dir_name:
            temporary_dir = Path(temporary_dir_name)
            for file_path, time_interval in zip(file_paths, time_intervals):
                if not reprocess_files and self._cache_contains(file_path, complete_schema):
                    continue

                tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")

                try:
                    filename = f"omni2_{time_interval[0].year}.dat"

                    logger.debug(f"Downloading file {self.URL + filename} ...")

                    self._download(temporary_dir, filename)

                    logger.debug("Processing file ...")

                    processed_df = self._process_single_file(temporary_dir / filename)
                    processed_df.to_csv(tmp_path, index=True, header=True)
                    tmp_path.replace(file_path)

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    if tmp_path.exists():
                        tmp_path.unlink()
                    continue

    def _download(self, temporary_dir: Path, filename: str):
        response = requests.get(self.URL + filename, timeout=10)
        response.raise_for_status()

        with open(temporary_dir / filename, "wb") as f:
            f.write(response.content)

    def _get_processed_file_list(self, start_time: datetime, end_time: datetime) -> Tuple[List, List]:
        """Get list of file paths and their corresponding time intervals.

        Returns
        -------
        Tuple[List, List]
            List of file paths and time intervals.
        """

        file_paths = []
        time_intervals = []

        start_year = start_time.year
        end_year = end_time.year

        # Check if end_time is within 3 hours of the next year boundary
        # This ensures we include the next year's file if needed for 3-hour Kp data
        next_year_start = datetime(end_year + 1, 1, 1, 0, 0, 0, tzinfo=end_time.tzinfo)
        time_diff_to_next_year = (next_year_start - end_time).total_seconds() / 3600

        # If end_time is within 3 hours of next year, include the next year
        if time_diff_to_next_year <= 3:
            end_year += 1

        for year in range(start_year, end_year + 1):
            file_path = self.data_dir / f"OMNI_LOW_RES_{year}.csv"
            file_paths.append(file_path)
            if year == start_year:
                interval_start = datetime(year, 1, 1, 0, 0, 0)
            else:
                interval_start = datetime(year, 1, 1, 0, 0, 0)
            if year == end_year:
                interval_end = datetime(year, 12, 31, 23, 59, 59)
            else:
                interval_end = datetime(year, 12, 31, 23, 59, 59)
            time_intervals.append((interval_start, interval_end))

        return file_paths, time_intervals

    def _process_single_file(self, file_path: Path) -> pd.DataFrame:
        """Process yearly OMNI Low Resolution file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Yearly OMNI Low Resolution data.
        """

        try:
            data = pd.read_csv(file_path, sep=r"\s+", header=None)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            raise ValueError(f"Cannot parse OMNI2 source file {file_path}: {error}") from error
        variable_count = data.shape[1] - len(self.TIME_COLUMNS)
        supported_counts = {len(LOW_RES_VARIABLES) - 2, len(LOW_RES_VARIABLES)}
        if variable_count not in supported_counts:
            raise ValueError(
                f"Unsupported OMNI2 record width {data.shape[1]}; expected 55 historic or 57 current fields."
            )

        present_variables = list(LOW_RES_VARIABLES[:variable_count])
        data.columns = [*self.TIME_COLUMNS, *(variable.name for variable in present_variables)]
        try:
            data = data.apply(pd.to_numeric)
        except (TypeError, ValueError) as error:
            raise ValueError(f"OMNI2 source file {file_path} contains nonnumeric fields: {error}") from error

        invalid_time = ~data["day"].between(1, 366) | ~data["hour"].between(0, 23)
        if invalid_time.any():
            raise ValueError("OMNI2 returned an invalid day-of-year or hour field.")

        year_and_day = data["year"].astype(int).astype(str) + data["day"].astype(int).astype(str).str.zfill(3)
        data["timestamp"] = pd.to_datetime(year_and_day, format="%Y%j", utc=True)
        if not (data["timestamp"].dt.year == data["year"]).all():
            raise ValueError("OMNI2 returned an invalid day-of-year for its record year.")
        data["timestamp"] += pd.to_timedelta(data["hour"], unit="h")
        data.set_index("timestamp", inplace=True)

        for variable in present_variables:
            if variable.fill_value is not None:
                data[variable.name] = data[variable.name].where(
                    data[variable.name] < variable.fill_value,
                    other=pd.NA,
                )

        # NASA added Lyman-alpha and the proton quasi-invariant after the
        # historic 55-word layout. They remain explicit NaN columns for old data.
        for variable in LOW_RES_VARIABLES[variable_count:]:
            data[variable.name] = pd.NA

        df = data.loc[:, [variable.name for variable in LOW_RES_VARIABLES]].copy()
        df["kp"] = df["kp"] / 10

        # change rounded numbers to be equal to 1/3 or 2/3 to be consistent with other Kp products
        df.loc[round(df["kp"] % 1, 2) == 0.7, "kp"] = round(df.loc[round(df["kp"] % 1, 2) == 0.7, "kp"]) - 1 / 3
        df.loc[round(df["kp"] % 1, 2) == 0.3, "kp"] = round(df.loc[round(df["kp"] % 1, 2) == 0.3, "kp"]) + 1 / 3

        return df

    def read(
        self,
        start_time: datetime,
        end_time: datetime,
        download: bool = False,
        variables: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Read hourly OMNI2 data for a time range.

        Parameters
        ----------
        start_time : datetime
            Start time for the data to be read.
        end_time : datetime
            End time for the data to be read.
        download : bool, optional
            Download data on the go, defaults to False.
        variables : str or iterable of str or None, optional
            Variables to return. ``None`` preserves the legacy ``dst``, ``kp``,
            and ``f107`` schema, ``"all"`` returns all 54 non-time fields, and
            a name or iterable selects a subset.

        Returns
        -------
        :class:`pandas.DataFrame`
            Selected OMNI variables plus ``file_name`` provenance. The index is
            timezone-aware UTC.

        Raises
        ------
        ValueError
            If the time range is invalid, a variable is unknown, or an existing
            partial or unreadable cache cannot satisfy the request.
        """
        START_YEAR = 1963
        variable_names = resolve_variable_names(
            LOW_RES_VARIABLES,
            variables,
            LOW_RES_DEFAULT_VARIABLES,
        )

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        if start_time >= end_time:
            msg = "start_time must be before end_time"
            logger.error(msg)
            raise ValueError(msg)

        if start_time < datetime(START_YEAR, 1, 1, tzinfo=timezone.utc):
            logger.warning(
                "Start date chosen falls behind the existing data. Moving start date to first"
                " available mission files..."
            )
            start_time = datetime(START_YEAR, 1, 1, tzinfo=timezone.utc)

        if start_time >= end_time:
            raise ValueError(f"Requested time range ends before OMNI data begin in {START_YEAR}.")

        file_paths, time_intervals = self._get_processed_file_list(start_time, end_time)
        t = pd.date_range(
            datetime(start_time.year, start_time.month, start_time.day),
            datetime(end_time.year, end_time.month, end_time.day, 23, 00, 00),
            freq=timedelta(hours=1),
            tz=timezone.utc,
        )
        data_out = pd.DataFrame(np.nan, index=t, columns=variable_names)
        data_out["file_name"] = None

        for file_path, time_interval in zip(file_paths, time_intervals):
            if not file_path.exists():
                if download:
                    self.download_and_process(time_interval[0], time_interval[1])
                if not file_path.exists():
                    warnings.warn(f"File {file_path} not found")
                    continue

            try:
                df_one_file = self._read_single_file(file_path)
            except (OSError, UnicodeDecodeError, ValueError, KeyError, pd.errors.ParserError) as error:
                if not download:
                    raise ValueError(f"Cannot read processed OMNI file {file_path}: {error}") from error
                logger.info(f"Replacing unreadable OMNI cache {file_path}")
                self.download_and_process(time_interval[0], time_interval[1], reprocess_files=True)
                try:
                    df_one_file = self._read_single_file(file_path)
                except (OSError, UnicodeDecodeError, ValueError, KeyError, pd.errors.ParserError) as retry_error:
                    raise ValueError(
                        f"Processed OMNI file {file_path} remains unreadable after attempted cache upgrade: "
                        f"{retry_error}"
                    ) from retry_error
            missing = [name for name in variable_names if name not in df_one_file.columns]
            if missing and download:
                logger.info(f"Upgrading partial OMNI cache {file_path} for variables: {', '.join(missing)}")
                self.download_and_process(time_interval[0], time_interval[1], reprocess_files=True)
                df_one_file = self._read_single_file(file_path)
                missing = [name for name in variable_names if name not in df_one_file.columns]
            if missing:
                raise ValueError(
                    f"Processed OMNI file {file_path} does not contain: {', '.join(missing)}. "
                    "Call read(..., download=True) or download_and_process(..., reprocess_files=True) "
                    "to upgrade the cache."
                )

            selected = df_one_file.loc[:, variable_names].copy()
            selected["file_name"] = file_path
            selected.loc[selected[variable_names].isna().all(axis=1), "file_name"] = None
            data_out = selected.combine_first(data_out)

        return data_out.loc[:, [*variable_names, "file_name"]]

    def _read_single_file(self, file_path: Path) -> pd.DataFrame:
        """Read yearly OMNI Low Resolution file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from yearly OMNI Low Resolution file.
        """
        df = pd.read_csv(file_path, index_col="timestamp")
        df.index = pd.to_datetime(df.index, utc=True)

        return df
