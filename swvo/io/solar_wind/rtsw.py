# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared base class for L1 solar wind readers whose real-time data comes from the SWPC RTSW feed.
"""

import logging
import warnings
from abc import abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar, List, Literal, Tuple

import numpy as np
import pandas as pd
import requests

from swvo.io.base import BaseIO
from swvo.io.utils import enforce_utc_timezone, sw_mag_propagation

logger = logging.getLogger(__name__)

logging.captureWarnings(True)

DataSource = Literal["archive", "realtime"]


class RTSWSolarWindBase(BaseIO):
    """Base class for L1 solar wind readers with an archive and the SWPC RTSW real-time feed.

    SWPC publishes the last 24 hours of 1-minute magnetic field and plasma data from all of its
    real-time L1 spacecraft (ACE, SOLAR-1 and IMAP) in two JSON files at
    ``https://services.swpc.noaa.gov/json/rtsw/``, with a ``source`` column naming the spacecraft.
    Subclasses select their spacecraft with `RTSW_SOURCE` and implement `_download_archive` for
    their own archive. This class handles everything else:

    Parameters
    ----------
    data_dir : Path | None
        Data directory. If not provided, it will be read from the environment variable `ENV_VAR_NAME`.
    prefer_env_var : bool, optional
        If True, the environment variable takes precedence over the passed `data_dir` argument.
    """

    RTSW_URL: ClassVar[str] = "https://services.swpc.noaa.gov/json/rtsw/"
    RTSW_NAME_MAG: ClassVar[str] = "rtsw_mag_1m.json"
    RTSW_NAME_WIND: ClassVar[str] = "rtsw_wind_1m.json"
    RTSW_SOURCE: ClassVar[str]
    REALTIME_WINDOW: ClassVar[timedelta]
    ARCHIVE_LATENCY: ClassVar[timedelta]
    FILE_PREFIX: ClassVar[str]

    MAG_FIELDS: ClassVar[list[str]] = ["bx_gsm", "by_gsm", "bz_gsm", "bavg"]
    PLASMA_FIELDS: ClassVar[list[str]] = ["speed", "proton_density", "temperature"]

    RTSW_MAG_COLUMNS: ClassVar[dict[str, str]] = {
        "bt": "bavg",
        "bx_gsm": "bx_gsm",
        "by_gsm": "by_gsm",
        "bz_gsm": "bz_gsm",
    }
    RTSW_WIND_COLUMNS: ClassVar[dict[str, str]] = {
        "proton_speed": "speed",
        "proton_density": "proton_density",
        "proton_temperature": "temperature",
    }

    DATA_SOURCE_COLUMN: ClassVar[str] = "data_source"

    @property
    def _name(self) -> str:
        """Spacecraft name used in messages and in the `file_name` column after propagation."""
        return self.LABEL.upper()

    @abstractmethod
    def _download_archive(self, start_time: datetime, end_time: datetime) -> pd.DataFrame | None:
        """Download and process archive data covering `start_time` to `end_time`.

        Implementations may return whole days around the requested range. Days the archive does
        not have yet are skipped rather than raising.

        Returns
        -------
        pd.DataFrame | None
            1-minute data indexed by UTC time with `MAG_FIELDS`, `PLASMA_FIELDS` and `pdyn`
            columns, or None if no archive data is available.
        """

    def _clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Hook for spacecraft-specific cleaning, applied to both archive and real-time data."""
        return data

    def download_and_process(self, start_time: datetime, end_time: datetime) -> None:
        """
        Download and process data, splitting data across midnight into appropriate day files.

        Data older than `REALTIME_WINDOW` is taken from the archive (`_download_archive`). If the
        requested range reaches into the last `REALTIME_WINDOW`, the SWPC RTSW feed is downloaded as
        well. Archive values take precedence over real-time values.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to download.
        end_time : datetime
            End time of the data to download.

        Raises
        ------
        AssertionError
            If `start_time` is after `end_time` or in the future.

        Returns
        -------
        None
        """
        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        current_time = datetime.now(timezone.utc)
        end_time = min(end_time, current_time)

        assert start_time <= end_time, "Start time must not be after end time or in the future!"

        realtime_cutoff = current_time - self.REALTIME_WINDOW
        archive_end = min(end_time, realtime_cutoff)

        self._resolved_urls = []

        if start_time <= archive_end:
            archive_data = self._download_archive(start_time, archive_end)
            if archive_data is not None and not archive_data.empty:
                self._save_processed_data(archive_data, source="archive")

        if end_time >= realtime_cutoff:
            logger.debug(f"Downloading {self._name} real-time data from the SWPC RTSW feed ...")
            realtime_data = self._download_realtime()
            realtime_data = realtime_data[realtime_data.index >= realtime_cutoff]
            if not realtime_data.empty:
                self._save_processed_data(realtime_data, source="realtime")

    def _download_realtime(self) -> pd.DataFrame:
        """Download the SWPC RTSW feed and return this spacecraft's data in SWVO columns.

        Raises
        ------
        requests.HTTPError
            If an HTTP request fails.
        """
        mag = self._fetch_rtsw(self.RTSW_NAME_MAG)
        wind = self._fetch_rtsw(self.RTSW_NAME_WIND)

        frames = []
        for records, columns in ((mag, self.RTSW_MAG_COLUMNS), (wind, self.RTSW_WIND_COLUMNS)):
            df = pd.DataFrame(records)
            if df.empty or "source" not in df.columns:
                frames.append(pd.DataFrame(columns=list(columns.values()), index=pd.DatetimeIndex([], tz="UTC")))
                continue
            df = df[df["source"] == self.RTSW_SOURCE]
            index = pd.to_datetime(df["time_tag"], utc=True).dt.floor("1min")
            df = df[list(columns)].rename(columns=columns).apply(pd.to_numeric, errors="coerce")
            df.index = pd.DatetimeIndex(index)
            frames.append(df[~df.index.duplicated(keep="last")])

        data = pd.concat(frames, axis=1, sort=True)
        return self._finalize(data)

    def _fetch_rtsw(self, file_name: str) -> list[dict]:
        url = self.RTSW_URL + file_name
        logger.debug(f"Downloading {url} ...")
        self._record_url(url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def _finalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Order columns, apply `_clean` and derive `pdyn` [nPa]."""
        for column in [*self.MAG_FIELDS, *self.PLASMA_FIELDS]:
            if column not in data.columns:
                data[column] = np.nan
        data = data[[*self.MAG_FIELDS, *self.PLASMA_FIELDS]].astype(float)
        data = self._clean(data)
        data["pdyn"] = 2e-6 * data["proton_density"].to_numpy() * data["speed"].to_numpy() ** 2
        data.index.name = "t"
        return data.sort_index()

    def _file_path(self, date) -> Path:
        return self.data_dir / date.strftime("%Y/%m") / f"{self.FILE_PREFIX}_{date.strftime('%Y%m%d')}.csv"

    def _save_processed_data(self, processed_df: pd.DataFrame, source: DataSource) -> None:
        """Merge new data into the daily cache files.

        Archive data overwrites what is cached, real-time data only fills minutes that are missing.

        Parameters
        ----------
        processed_df : pd.DataFrame
            New data indexed by UTC time.
        source : {"archive", "realtime"}
            Where the data came from.
        """
        processed_df = processed_df.copy()
        has_data = processed_df[[*self.MAG_FIELDS, *self.PLASMA_FIELDS]].notna().any(axis=1)
        processed_df[self.DATA_SOURCE_COLUMN] = np.where(has_data, source, None)

        for date in np.unique(processed_df.index.date):  # ty: ignore[missing-attribute]
            file_path = self._file_path(date)
            tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")

            try:
                day_start = enforce_utc_timezone(datetime.combine(date, datetime.min.time()))
                day_end = enforce_utc_timezone(datetime.combine(date, datetime.max.time()))
                day_data = processed_df[(processed_df.index >= day_start) & (processed_df.index <= day_end)]

                if file_path.exists():
                    logger.debug(f"Found previous file for {date}. Loading and combining ...")
                    previous_df = self._read_single_file(file_path).drop(columns="file_name")
                    if source == "archive":
                        day_data = day_data.combine_first(previous_df)
                    else:
                        day_data = previous_df.combine_first(day_data)

                if source == "archive":
                    settled = (day_data[self.DATA_SOURCE_COLUMN] == "realtime") & (
                        day_data.index < datetime.now(timezone.utc) - self.ARCHIVE_LATENCY
                    )
                    day_data.loc[settled, self.DATA_SOURCE_COLUMN] = "realtime_no_archive"

                day_data.index.name = "t"
                logger.debug(f"Saving processed file for {date}")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                day_data.to_csv(tmp_path, index=True, header=True)
                tmp_path.replace(file_path)

            except Exception as e:
                logger.error(f"Failed to process file for {date}: {e}")
                if tmp_path.exists():
                    tmp_path.unlink()
                continue

    def _needs_archive_refresh(self, file_path: Path, current_time: datetime) -> bool:
        """Whether a cached day still holds real-time rows that the archive should have by now."""
        if not file_path.exists():
            return False
        df = self._read_single_file(file_path)
        realtime_rows = df.index[df[self.DATA_SOURCE_COLUMN] == "realtime"]
        return bool(len(realtime_rows)) and realtime_rows.min() < current_time - self.ARCHIVE_LATENCY

    def read(
        self,
        start_time: datetime,
        end_time: datetime,
        download: bool = False,
        propagation: bool = False,
    ) -> pd.DataFrame:
        """
        Read data for the specified time range.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read.
        end_time : datetime
            End time of the data to read.
        download : bool, optional
            Download data on the go, defaults to False. Days that are missing, that reach into the
            last `REALTIME_WINDOW`, or that still hold real-time rows older than `ARCHIVE_LATENCY`
            are (re)downloaded.
        propagation : bool, optional
            Propagate the data from L1 to near-Earth, defaults to False.

        Returns
        -------
        :class:`pandas.DataFrame`
            1-minute solar wind data with `MAG_FIELDS`, `PLASMA_FIELDS`, `pdyn` and `file_name`.

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
        data_out = pd.DataFrame(index=t, data={field: np.nan for field in [*self.MAG_FIELDS, *self.PLASMA_FIELDS]})

        if download:
            current_time = datetime.now(timezone.utc)
            # Recent days keep changing while new data arrives, so they are always refreshed.
            needs_realtime = end_time >= current_time - self.REALTIME_WINDOW
            if (
                needs_realtime
                or any(not file_path.exists() for file_path in file_paths)
                or any(self._needs_archive_refresh(file_path, current_time) for file_path in file_paths)
            ):
                try:
                    self.download_and_process(start_time, end_time)
                except AssertionError as e:
                    logger.error(f"`download_and_process` failed because: {e}")

        for file_path in file_paths:
            if not file_path.exists():
                warnings.warn(f"File {file_path} not found")
                continue

            df_one_day = self._read_single_file(file_path).drop(columns=self.DATA_SOURCE_COLUMN)
            data_out = df_one_day.combine_first(data_out)

        data_out = data_out.truncate(
            before=start_time - timedelta(minutes=0.999999),
            after=end_time + timedelta(minutes=0.999999),
        )

        if propagation:
            data_out = sw_mag_propagation(data_out)
            if not data_out.empty:
                data_out["file_name"] = data_out.apply(self._update_filename, axis=1)

        return data_out

    def _get_processed_file_list(self, start_time: datetime, end_time: datetime) -> Tuple[List, List]:
        """Get list of file paths and their corresponding time intervals.

        Parameters
        ----------
        start_time : datetime
            Start time of the data.
        end_time : datetime
            End time of the data.

        Returns
        -------
        Tuple[List, List]
            List of file paths and time intervals.
        """
        file_paths = []
        time_intervals = []

        current_time = datetime(start_time.year, start_time.month, start_time.day, 0, 0, 0)
        end_time = datetime(end_time.year, end_time.month, end_time.day, 0, 0, 0)

        while current_time <= end_time:
            file_paths.append(self._file_path(current_time))

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
        index_date = row.name.date()  # ty: ignore[missing-attribute]
        return f"propagated from previous {self._name} NOWCAST file" if file_date != index_date else row["file_name"]

    def _read_single_file(self, file_path) -> pd.DataFrame:
        """Read one cached daily file to a DataFrame.

        Files written before the source column existed are treated as archive data.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Cached data, with `file_name` and `data_source` columns.
        """
        df = pd.read_csv(file_path, header="infer")

        df["t"] = pd.to_datetime(df["t"], utc=True)
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)

        if self.DATA_SOURCE_COLUMN not in df.columns:
            df[self.DATA_SOURCE_COLUMN] = "archive"

        df["file_name"] = file_path
        df.loc[df["bavg"].isna() & df["temperature"].isna(), "file_name"] = None

        return df
