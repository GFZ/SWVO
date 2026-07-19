# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Simon Mischel
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling OMNI high resolution data.
"""

import calendar
import logging
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import pandas as pd
import requests

from swvo.io.base import BaseIO
from swvo.io.omni.variables import HIGH_RES_DEFAULT_VARIABLES, HIGH_RES_VARIABLES, resolve_variable_names
from swvo.io.utils import enforce_utc_timezone

logger = logging.getLogger(__name__)


class OMNIHighRes(BaseIO):
    """This is a class for the OMNI High Resolution data.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for the OMNI High Resolution data. If not provided, it will be read from the environment variable

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.
    """

    ENV_VAR_NAME = "OMNI_HIGH_RES_STREAM_DIR"

    URL = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

    START_YEAR = 1981
    LABEL = "omni"
    PARALLEL_DOWNLOAD_THRESHOLD = 10
    MAX_PARALLEL_DOWNLOADS = 10

    @classmethod
    def available_variables(cls, cadence_min: int = 1) -> pd.DataFrame:
        """Return metadata for variables available at a cadence.

        Parameters
        ----------
        cadence_min : int, optional
            OMNI cadence in minutes. Must be ``1`` or ``5``.

        Returns
        -------
        pandas.DataFrame
            One row per variable with its canonical name, NASA request ID,
            description, unit, supported cadences, and accepted aliases.
        """

        cls._validate_cadence(cadence_min)
        return pd.DataFrame(
            [
                {
                    "name": variable.name,
                    "nasa_id": variable.nasa_id,
                    "description": variable.description,
                    "unit": variable.unit,
                    "fill_value": variable.fill_value,
                    "cadences": variable.cadences,
                    "aliases": variable.aliases,
                }
                for variable in HIGH_RES_VARIABLES
                if cadence_min in variable.cadences
            ]
        )

    @staticmethod
    def _validate_cadence(cadence_min: int) -> None:
        if cadence_min not in (1, 5):
            raise AssertionError("Only 1 or 5 minute cadence can be chosen for high resolution omni data.")

    @staticmethod
    def _cache_contains(file_path, variable_names: Iterable[str]) -> bool:
        """Check a processed file's schema without loading its data."""

        if not file_path.exists():
            return False
        try:
            columns = set(pd.read_csv(file_path, nrows=0).columns)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return False
        return set(variable_names).issubset(columns)

    def download_and_process(
        self,
        start_time: datetime,
        end_time: datetime,
        cadence_min: int = 1,
        reprocess_files: bool = False,
    ) -> None:
        """Download and process OMNI High Resolution data files.

        Parameters
        ----------
        start_time : datetime
            Start time for data download.
        end_time : datetime
            End time for data download.
        cadence_min : int, optional
            Cadence of the data in minutes, defaults to 1
        reprocess_files : bool, optional
            Downloads and processes the files again, defaults to False, by default False

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            Raises `AssertionError` if the cadence is not 1 or 5 minutes.
        ValueError
            If ``start_time`` is not before ``end_time``.
        """

        self._validate_cadence(cadence_min)
        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)
        if start_time >= end_time:
            raise ValueError("start_time must be before end_time")

        complete_schema = resolve_variable_names(
            HIGH_RES_VARIABLES,
            "all",
            HIGH_RES_DEFAULT_VARIABLES,
            cadence=cadence_min,
        )

        file_paths, time_intervals = self._get_processed_file_list(start_time, end_time, cadence_min)

        download_tasks = []
        for file_path, time_interval in zip(file_paths, time_intervals):
            if not reprocess_files and self._cache_contains(file_path, complete_schema):
                continue

            download_tasks.append((file_path, time_interval))

        if len(download_tasks) > self.PARALLEL_DOWNLOAD_THRESHOLD:
            max_workers = min(self.MAX_PARALLEL_DOWNLOADS, len(download_tasks))
            logger.info(f"Downloading {len(download_tasks)} OMNI high resolution files with {max_workers} workers.")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._download_and_process_single_file, file_path, time_interval, cadence_min)
                    for file_path, time_interval in download_tasks
                ]
                for future in as_completed(futures):
                    future.result()
            return

        for file_path, time_interval in download_tasks:
            self._download_and_process_single_file(file_path, time_interval, cadence_min)

    def _download_and_process_single_file(
        self,
        file_path,
        time_interval: Tuple[datetime, datetime],
        cadence_min: int,
    ) -> None:
        """Download and process one monthly OMNI High Resolution file."""

        # Create directory structure if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")

        try:
            variable_names = resolve_variable_names(
                HIGH_RES_VARIABLES,
                "all",
                HIGH_RES_DEFAULT_VARIABLES,
                cadence=cadence_min,
            )
            data = self._get_data_from_omni(
                start=time_interval[0],
                end=time_interval[1],
                cadence=cadence_min,
                variable_names=variable_names,
            )

            logger.debug("Processing file ...")

            processed_df = self._process_single_month(
                data,
                original_end=time_interval[1],
                cadence_min=cadence_min,
                variable_names=variable_names,
            )

            # Do not save empty DataFrames — no data available for this interval
            if processed_df.empty:
                logger.warning(f"Skipping save for {file_path}: no data available for this interval.")
                return

            processed_df.to_csv(tmp_path, index=True, header=True)
            tmp_path.replace(file_path)

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

    def read(
        self,
        start_time: datetime,
        end_time: datetime,
        cadence_min: int = 1,
        download: bool = False,
        variables: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Read high-resolution OMNI data for a time range.

        Parameters
        ----------
        start_time : datetime
            Start time for reading data.
        end_time : datetime
            End time for reading data.
        cadence_min : int, optional
            Cadence of the data in minutes, defaults to 1
        download : bool, optional
            Download data on the go, defaults to False.
        variables : str or iterable of str or None, optional
            Variables to return. ``None`` preserves the legacy nine-column
            schema, ``"all"`` returns every variable supported by the selected
            cadence, and a name or iterable selects a subset. Normalized aliases
            such as ``"sym_h"`` are accepted.

        Returns
        -------
        :class:`pandas.DataFrame`
            Selected OMNI variables plus ``file_name`` provenance. The index is
            timezone-aware UTC.

        Raises
        ------
        AssertionError
            Raises `AssertionError` if the cadence is not 1 or 5 minutes.
        ValueError
            If the time range is invalid, a variable is unknown or unavailable
            at the selected cadence, or an existing partial or unreadable cache
            cannot satisfy the request.
        """
        self._validate_cadence(cadence_min)
        variable_names = resolve_variable_names(
            HIGH_RES_VARIABLES,
            variables,
            HIGH_RES_DEFAULT_VARIABLES,
            cadence=cadence_min,
        )

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        if start_time >= end_time:
            msg = "start_time must be before end_time"
            logger.error(msg)
            raise ValueError(msg)

        if start_time < datetime(self.START_YEAR, 1, 1, tzinfo=timezone.utc):
            logger.warning(
                "Start date chosen falls behind the existing data. Moving start date to first"
                " available mission files..."
            )
            start_time = datetime(self.START_YEAR, 1, 1, tzinfo=timezone.utc)

        if start_time >= end_time:
            raise ValueError(f"Requested time range ends before OMNI data begin in {self.START_YEAR}.")

        file_paths, time_intervals = self._get_processed_file_list(start_time, end_time, cadence_min)

        dfs = []

        for file_path, time_interval in zip(file_paths, time_intervals):
            if not file_path.exists():
                if download:
                    self._download_and_process_single_file(file_path, time_interval, cadence_min)
                else:
                    logger.warning(f"File {file_path} not found")
                    continue

            # Re-check after attempted download — file may not exist if no data
            # was available for this interval (e.g. beyond the OMNI data range)
            if not file_path.exists():
                logger.warning(f"File {file_path} not available after download attempt, skipping.")
                continue

            try:
                df = self._read_single_file(file_path)
            except (OSError, UnicodeDecodeError, ValueError, KeyError, pd.errors.ParserError) as error:
                if not download:
                    raise ValueError(f"Cannot read processed OMNI file {file_path}: {error}") from error
                logger.info(f"Replacing unreadable OMNI cache {file_path}")
                self._download_and_process_single_file(file_path, time_interval, cadence_min)
                try:
                    df = self._read_single_file(file_path)
                except (OSError, UnicodeDecodeError, ValueError, KeyError, pd.errors.ParserError) as retry_error:
                    raise ValueError(
                        f"Processed OMNI file {file_path} remains unreadable after attempted cache upgrade: "
                        f"{retry_error}"
                    ) from retry_error
            missing = [name for name in variable_names if name not in df.columns]
            if missing and download:
                logger.info(f"Upgrading partial OMNI cache {file_path} for variables: {', '.join(missing)}")
                self._download_and_process_single_file(file_path, time_interval, cadence_min)
                df = self._read_single_file(file_path)
                missing = [name for name in variable_names if name not in df.columns]
            if missing:
                raise ValueError(
                    f"Processed OMNI file {file_path} does not contain: {', '.join(missing)}. "
                    "Call read(..., download=True) or download_and_process(..., reprocess_files=True) "
                    "to upgrade the cache."
                )

            selected = df.loc[:, variable_names].copy()
            selected["file_name"] = file_path
            selected.loc[selected[variable_names].isna().all(axis=1), "file_name"] = None
            dfs.append(selected)

        if not dfs:
            raise ValueError("No OMNI High Resolution files are available for the requested time range.")
        data_out = pd.concat(dfs, ignore_index=False)

        if not data_out.empty:
            data_out.index = enforce_utc_timezone(data_out.index)

        data_out = data_out.truncate(
            before=start_time - timedelta(minutes=cadence_min - 0.0000001),
            after=end_time + timedelta(minutes=cadence_min + 0.0000001),
        )

        return data_out

    def _get_processed_file_list(
        self, start_time: datetime, end_time: datetime, cadence_min: float
    ) -> Tuple[List, List]:
        """Get list of file paths and their corresponding time intervals.

        Parameters
        ----------
        start_time : datetime
            Start time for the data.
        end_time : datetime
            End time for the data.
        cadence_min : float
            Cadence of the data in minutes.

        Returns
        -------
        Tuple[List, List]
            List of file paths and time intervals.
        """

        file_paths = []
        time_intervals = []

        # Start from the first day of the start_time month
        current_date = start_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Check if end_time is within cadence_min of the next month boundary
        # This ensures we include the next month's file if needed
        end_year = end_time.year
        end_month = end_time.month

        # Calculate next month start
        if end_month == 12:
            next_month_start = datetime(end_year + 1, 1, 1, 0, 0, 0, tzinfo=end_time.tzinfo)
        else:
            next_month_start = datetime(end_year, end_month + 1, 1, 0, 0, 0, tzinfo=end_time.tzinfo)

        time_diff_to_next_month = (next_month_start - end_time).total_seconds() / 3600

        # If end_time is within `cadence_min` of next month, include the next month
        cadence_hours = cadence_min / 60
        include_next_month = time_diff_to_next_month <= cadence_hours

        while current_date <= end_time or (include_next_month and current_date == next_month_start):
            year = current_date.year
            month = current_date.month

            # directory: YYYY/
            year_dir = self.data_dir / f"{year:04d}"

            # Create file path
            file_path = year_dir / f"OMNI_HIGH_RES_{cadence_min}min_{year:04d}{month:02d}.csv"
            file_paths.append(file_path)

            # Create time interval for current month
            interval_start = datetime(year, month, 1, 0, 0, 0, tzinfo=current_date.tzinfo)

            # Get last day of the month
            last_day = calendar.monthrange(year, month)[1]
            interval_end = datetime(year, month, last_day, 23, 59, 59, tzinfo=current_date.tzinfo)

            time_intervals.append((interval_start, interval_end))

            # Move to next month
            if month == 12:
                current_date = current_date.replace(year=year + 1, month=1)
            else:
                current_date = current_date.replace(month=month + 1)

            # Break condition to avoid infinite loop
            if include_next_month and current_date > next_month_start:
                break

        return file_paths, time_intervals

    def _process_single_month(
        self,
        data: list[str],
        original_end: Optional[datetime] = None,
        cadence_min: int = 1,
        variable_names: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Process monthly OMNI High Resolution data to a DataFrame.

        Parameters
        ----------
        data : list[str]
            Raw data lines from the OMNI service.
        original_end : datetime, optional
            The original requested end time. Used to build a NaN-filled DataFrame
            when no data is available (e.g. the interval is beyond the OMNI data range).
        cadence_min : int, optional
            Cadence of the requested records.
        variable_names : iterable of str, optional
            Canonical variables in the same order as the OMNIWeb request.

        Returns
        -------
        pd.DataFrame
            Monthly OMNI High Resolution data. Returns a NaN-filled DataFrame
            up to ``original_end`` if no data is available, or an empty DataFrame
            if ``original_end`` is not provided.
        """
        columns = resolve_variable_names(
            HIGH_RES_VARIABLES,
            list(variable_names) if variable_names is not None else None,
            HIGH_RES_DEFAULT_VARIABLES,
            cadence=cadence_min,
        )

        # Empty data list signals that no data is available for this interval
        if not data:
            if original_end is None:
                return pd.DataFrame(columns=columns)
            original_end = enforce_utc_timezone(original_end)
            # Build a NaN-filled DataFrame with the correct timestamps up to original_end
            index = pd.date_range(
                start=original_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
                end=original_end,
                freq=pd.tseries.frequencies.to_offset(f"{cadence_min}min"),
                tz=original_end.tzinfo,
            )
            return pd.DataFrame(pd.NA, index=index, columns=columns)

        if not any(line.strip().startswith("YYYY") for line in data):
            raise ValueError("OMNIWeb response does not contain the expected YYYY data header.")

        # OMNIWeb wraps records in HTML and its parameter list contains lines
        # such as ``19 Vx Velocity``. Require all four leading time fields so
        # those labels (and year-prefixed footer text) cannot become data rows.
        record_prefix = re.compile(r"^\s*\d{4}\s+\d{1,3}\s+\d{1,2}\s+\d{1,2}(?:\s|$)")
        data_lines = [line for line in data if record_prefix.match(line)]

        if not data_lines:
            msg = "DataFrame is empty."
            logger.error(msg)
            raise ValueError(msg)

        raw_columns = ["YYYY", "DOY", "HR", "MN", *columns]
        rows = [line.split() for line in data_lines]
        invalid_widths = sorted({len(row) for row in rows if len(row) != len(raw_columns)})
        if invalid_widths:
            raise ValueError(
                f"OMNIWeb returned record widths {invalid_widths}; expected {len(raw_columns)} values "
                f"for {len(columns)} selected variables."
            )

        df = pd.DataFrame(rows, columns=raw_columns)
        df = df.apply(pd.to_numeric)

        invalid_time = ~df["DOY"].between(1, 366) | ~df["HR"].between(0, 23) | ~df["MN"].between(0, 59)
        if invalid_time.any():
            raise ValueError("OMNIWeb returned an invalid day-of-year, hour, or minute field.")

        year_and_day = df["YYYY"].astype(int).astype(str) + df["DOY"].astype(int).astype(str).str.zfill(3)
        df["timestamp"] = pd.to_datetime(year_and_day, format="%Y%j", utc=True)
        if not (df["timestamp"].dt.year == df["YYYY"]).all():
            raise ValueError("OMNIWeb returned an invalid day-of-year for its record year.")
        df["timestamp"] += pd.to_timedelta(df["HR"], unit="h") + pd.to_timedelta(df["MN"], unit="m")

        df.drop(columns=["YYYY", "HR", "MN", "DOY"], inplace=True)
        df.set_index("timestamp", inplace=True)

        metadata = {variable.name: variable for variable in HIGH_RES_VARIABLES}
        for column in columns:
            fill_value = metadata[column].fill_value
            if fill_value is not None:
                df[column] = df[column].where(df[column] < fill_value, other=pd.NA)

        if df.empty:
            msg = "DataFrame is empty after processing the month."
            logger.error(msg)
            raise ValueError(msg)

        return df

    def _read_single_file(self, file_path) -> pd.DataFrame:
        """Read monthly OMNI High Resolution file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from monthly High Resolution file.
        """
        df = pd.read_csv(file_path, index_col=0)

        df.index = pd.to_datetime(df.index, utc=True)

        nan_mask = df.isna().all(axis=1)
        df["file_name"] = file_path
        df.loc[nan_mask, "file_name"] = None

        return df

    def _get_data_from_omni(
        self,
        start: datetime,
        end: datetime,
        cadence: int = 1,
        variable_names: Iterable[str] | None = None,
    ) -> list[str]:
        """
        Fetches data from NASA's OMNIWeb service.

        If an invalid date range error is returned, it automatically finds the
        suggested valid end date and retries the request. If the suggested end
        date falls before the start date, an empty list is returned to signal
        that no data is available for the requested interval.
        """

        self._validate_cadence(cadence)
        start = enforce_utc_timezone(start)
        end = enforce_utc_timezone(end)
        selected_names = resolve_variable_names(
            HIGH_RES_VARIABLES,
            list(variable_names) if variable_names is not None else None,
            HIGH_RES_DEFAULT_VARIABLES,
            cadence=cadence,
        )
        metadata = {variable.name: variable for variable in HIGH_RES_VARIABLES}
        payload: dict[str, str | list[str]] = {
            "activity": "retrieve",
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "vars": [str(metadata[name].nasa_id) for name in selected_names],
        }
        if cadence == 1:
            payload.update({"res": "min", "spacecraft": "omni_min"})
        else:
            payload.update({"res": "5min", "spacecraft": "omni_5min"})
        logger.debug(f"Fetching data from {self.URL} with payload: {payload}")
        response = requests.post(self.URL, data=payload, timeout=30)
        response.raise_for_status()
        data = response.text.splitlines()

        correct_range_line = next((line for line in data if "correct range" in line.lower()), None)
        has_error = any(re.search(r"\berror\b", line, flags=re.IGNORECASE) for line in data)
        if correct_range_line is not None or has_error:
            logger.warning("Received an error response from the server.")

            if correct_range_line is not None:
                # Use regex to find the valid date range (e.g., YYYYMMDD - YYYYMMDD)
                match = re.search(r"correct range:\s*\d{8}\s*-\s*(\d{8})", correct_range_line, re.IGNORECASE)
                if match:
                    new_end_date_str = match.group(1)
                    new_end_date = datetime.strptime(new_end_date_str, "%Y%m%d").replace(tzinfo=timezone.utc)

                    logger.warning(
                        f"Invalid date range detected. Found suggested end date: {new_end_date.strftime('%Y-%m-%d')}"
                    )

                    # If the suggested end date is before the start date, no data is available
                    # for this range — return an empty list to signal an empty DataFrame
                    if new_end_date < start:
                        logger.warning(
                            f"Suggested end date {new_end_date.strftime('%Y-%m-%d')} is before "
                            f"start date {start.strftime('%Y-%m-%d')}. No data available for this range."
                        )
                        return []

                    if new_end_date >= end:
                        raise ValueError(
                            "OMNIWeb returned an invalid corrected end date that would not shorten the request."
                        )

                    # Recursively call the function with the original start date and the new end date
                    return self._get_data_from_omni(
                        start=start,
                        end=new_end_date,
                        cadence=cadence,
                        variable_names=selected_names,
                    )
            msg = f"An unspecified error occurred: {data}"
            logger.error(msg)
            raise ValueError(msg)
        return data
