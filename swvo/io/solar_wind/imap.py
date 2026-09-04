# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling IMAP (Interstellar Mapping and Acceleration Probe) Solar Wind data.
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
import requests
from richpool import p_map

from swvo.io.base import BaseIO
from swvo.io.utils import enforce_utc_timezone, sw_mag_propagation

logger = logging.getLogger(__name__)

logging.captureWarnings(True)

# The only two I-ALiRT `space-weather` instruments this reader ever queries.
Instrument = Literal["mag", "swapi"]


@dataclass
class _DayChunks:
    """Accumulates one UTC day's fetched chunks, grouped by instrument."""

    mag: list = field(default_factory=list)
    swapi: list = field(default_factory=list)
    failed: bool = False


class SWIMAP(BaseIO):
    """This is a class for the IMAP I-ALiRT Solar Wind data.

    IMAP sits at L1 and its I-ALiRT (Active Link for Real-Time) system broadcasts near-real-time
    in-situ data over a public API (``https://ialirt.imap-mission.com/space-weather``). This
    reader pulls the ``mag`` (magnetometer) and ``swapi`` (Solar Wind and Pickup Ion)
    instruments and adapts them to the SWVO solar wind schema.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for the IMAP Solar Wind data. If not provided, it will be read from the environment variable
    prefer_env_var : bool, optional
        If True, the environment variable takes precedence over the passed `data_dir` argument.

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.
    """

    ENV_VAR_NAME = "SW_IMAP_STREAM_DIR"

    URL = "https://ialirt.imap-mission.com/space-weather"

    MAG_FIELDS = ["bx_gsm", "by_gsm", "bz_gsm", "bavg"]
    SWAPI_FIELDS = ["speed", "proton_density", "temperature"]

    # mag_B_GSM arrives as a 3-element [bx, by, bz] list per record rather than separate scalar
    # keys, so it is expanded positionally instead of via a simple rename map.
    MAG_VECTOR_COLUMNS = ["bx_gsm", "by_gsm", "bz_gsm"]
    MAG_MAGNITUDE_FIELD = "mag_B_magnitude"

    SWAPI_PARAMETER_FIELDS = {
        "speed": "swapi_pseudo_proton_speed",
        "proton_density": "swapi_pseudo_proton_density",
        "temperature": "swapi_pseudo_proton_temperature",
    }

    LABEL = "imap"

    _MAX_DOWNLOAD_ATTEMPTS = 4
    _RETRY_BACKOFF_SECONDS = 1.0

    _CHUNK_SIZE = timedelta(hours=1)
    _CHUNK_WORKERS = 8

    # Data older than this requires an API key this reader does not have/use (observed as an
    # HTTP 400 with an "API key required" message).
    _PUBLIC_DATA_START = datetime(2026, 2, 1, tzinfo=timezone.utc)

    def download_and_process(self, start_time: datetime, end_time: datetime) -> None:
        """
        Download and process IMAP data, splitting data across midnight into appropriate day files.

        The whole requested period (rounded out to full UTC days, since output is cached per day)
        is pre-split into fixed 1-hour windows for both `mag` and `swapi`, and every window/
        instrument request is fetched concurrently under a single progress bar (see
        `_fetch_chunks`) rather than looping day by day. A day where any of its chunks
        permanently fails (after retries) is skipped entirely, with a warning, rather than
        aborting the whole range; re-running the same range later retries only the missing days.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to download. Must be timezone-aware.
        end_time : datetime
            End time of the data to download. Must be timezone-aware.

        Raises
        ------
        AssertionError
            If `start_time` is after `end_time`.
        ValueError
            If `start_time` is before `_PUBLIC_DATA_START`, since that data requires an API key.

        Returns
        -------
        None
        """
        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        if start_time < self._PUBLIC_DATA_START:
            raise ValueError(
                f"IMAP data before {self._PUBLIC_DATA_START:%Y-%m-%dT%H:%M:%S} requires an API key, "
                "which this reader does not use. Please choose a later start_time."
            )

        assert start_time < end_time, "Start time must be before end time!"

        self._resolved_urls = []

        day_starts = [
            date.to_pydatetime().replace(tzinfo=timezone.utc)
            for date in pd.date_range(start=start_time.date(), end=end_time.date(), freq="D")
        ]
        range_start = day_starts[0]
        range_end = day_starts[-1] + timedelta(days=1)

        chunks_by_day = self._fetch_chunks(range_start, range_end)

        failed_dates = []
        for day_start in day_starts:
            day = chunks_by_day[day_start.date()]
            if day.failed:
                failed_dates.append(day_start.date())
                continue

            mag_df = pd.concat(day.mag) if day.mag else self._empty_instrument_frame("mag")
            swapi_df = pd.concat(day.swapi) if day.swapi else self._empty_instrument_frame("swapi")
            processed_df = self._merge_instrument_data(mag_df, swapi_df, day_start)
            self._save_processed_data(processed_df, day_start.date())

        if failed_dates:
            dates = ", ".join(str(date) for date in failed_dates)
            day_label = "day" if len(failed_dates) == 1 else "days"
            warnings.warn(
                f"IMAP download failed for {len(failed_dates)} {day_label}: {dates}. "
                "Re-run the same range to retry failed days.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _fetch_chunks(self, range_start: datetime, range_end: datetime) -> dict[date, _DayChunks]:
        """Fetch `mag` and `swapi` for `[range_start, range_end)` as concurrent fixed 1-hour
        requests.

        A chunk that permanently fails (after `_request_instrument`'s own retries) is reported as
        `None` rather than raised, so one bad chunk doesn't abort the whole batch.

        Parameters
        ----------
        range_start : datetime
            Start of the UTC range to fetch (inclusive). Must fall on a day boundary.
        range_end : datetime
            End of the UTC range to fetch (exclusive). Must fall on a day boundary.

        Returns
        -------
        dict[date, _DayChunks]
            One entry per UTC date in the range, holding that day's successfully parsed per-chunk
            DataFrames (grouped by instrument) and whether any chunk for that day failed.
        """
        chunk_count = int((range_end - range_start) / self._CHUNK_SIZE)
        window_starts = [range_start + i * self._CHUNK_SIZE for i in range(chunk_count)]
        tasks = [(instrument, window_start) for window_start in window_starts for instrument in ("mag", "swapi")]

        results = p_map(
            lambda task: self._fetch_chunk(task[0], task[1], task[1] + self._CHUNK_SIZE),
            tasks,
            kind="thread",
            num_cpus=self._CHUNK_WORKERS,
            desc="Fetching IMAP data",
        )

        chunks_by_day = {
            (range_start + i * timedelta(days=1)).date(): _DayChunks() for i in range((range_end - range_start).days)
        }
        for instrument, window_start, chunk_df in results:
            day = chunks_by_day[window_start.date()]
            if chunk_df is None:
                day.failed = True
                continue
            getattr(day, instrument).append(chunk_df)

        return chunks_by_day

    def _fetch_chunk(
        self,
        instrument: Instrument,
        window_start: datetime,
        window_end: datetime,
    ) -> tuple[Instrument, datetime, pd.DataFrame | None]:
        """Fetch one instrument's data for one window, reporting failure instead of raising.

        Parameters
        ----------
        instrument : Instrument
            Either ``"mag"`` or ``"swapi"``.
        window_start : datetime
            Start of the UTC window to request (inclusive).
        window_end : datetime
            End of the UTC window to request (exclusive).

        Returns
        -------
        tuple[Instrument, datetime, pd.DataFrame | None]
            `(instrument, window_start, data)`; `data` is `None` if the request permanently failed.
        """
        try:
            return instrument, window_start, self._request_instrument(instrument, window_start, window_end)
        except Exception as error:
            logger.error(f"Failed to download IMAP {instrument} data for {window_start} - {window_end}: {error}")
            return instrument, window_start, None

    def _empty_instrument_frame(self, instrument: Instrument) -> pd.DataFrame:
        """An empty, correctly indexed and columned frame for `instrument`, for a day with no chunks."""
        fields = self.MAG_FIELDS if instrument == "mag" else self.SWAPI_FIELDS
        return pd.DataFrame(columns=fields, index=pd.DatetimeIndex([], tz="UTC"))

    def _request_instrument(self, instrument: Instrument, window_start: datetime, window_end: datetime) -> pd.DataFrame:
        """Make one (retried) HTTP GET for a single time window.

        Transient failures (timeouts, connection errors, HTTP 429/5xx) are retried with backoff;
        any other error (e.g. a malformed request, or a window still too wide for the API) is
        raised immediately.

        Parameters
        ----------
        instrument : Instrument
            Either ``"mag"`` or ``"swapi"``.
        window_start : datetime
            Start of the UTC window to request (inclusive).
        window_end : datetime
            End of the UTC window to request (exclusive).

        Returns
        -------
        pd.DataFrame
            Parsed instrument data for the window, indexed by UTC timestamp.
        """
        params = {
            "instrument": instrument,
            "time_utc_start": window_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "time_utc_end": window_end.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        for attempt in range(1, self._MAX_DOWNLOAD_ATTEMPTS + 1):
            logger.debug(
                f"Downloading IMAP {instrument} data for {window_start} - {window_end} "
                f"(attempt {attempt}/{self._MAX_DOWNLOAD_ATTEMPTS}) ..."
            )
            try:
                resolved_url = requests.Request("GET", self.URL, params=params).prepare().url
                self._record_url(resolved_url)  # ty:ignore[invalid-argument-type]
                response = requests.get(self.URL, params=params, timeout=30)
                if self._is_api_key_required_response(response):
                    raise ValueError(
                        f"IMAP {instrument} query for {window_start} - {window_end} requires an "
                        "API key, which this reader does not use."
                    )
                response.raise_for_status()
                payload = response.json()
                return self._parse_instrument_payload(instrument, payload)
            except Exception as error:
                if not self._is_retryable_download_error(error) or attempt == self._MAX_DOWNLOAD_ATTEMPTS:
                    raise

                backoff = self._RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1)
                logger.warning(
                    f"Transient IMAP {instrument} failure for {window_start} - {window_end} "
                    f"({self._retry_reason(error)}); retrying in {backoff:.1f}s "
                    f"(attempt {attempt + 1}/{self._MAX_DOWNLOAD_ATTEMPTS})"
                )
                sleep(backoff)

    def _parse_instrument_payload(self, instrument: Instrument, payload: dict) -> pd.DataFrame:
        """Convert one instrument's `space-weather` JSON payload to a DataFrame.

        Parameters
        ----------
        instrument : Instrument
            Either ``"mag"`` or ``"swapi"``.
        payload : dict
            Decoded JSON payload, shaped ``{"meta": {...}, "data": [...]}``.

        Returns
        -------
        pd.DataFrame
            Data indexed by UTC timestamp, with SWVO column names. Empty (but correctly indexed
            and columned) if the payload carries no records.
        """
        fields = self.MAG_FIELDS if instrument == "mag" else self.SWAPI_FIELDS
        records = payload.get("data", [])

        if not records:
            return pd.DataFrame(columns=fields, index=pd.DatetimeIndex([], tz="UTC"))

        df = pd.DataFrame(records)
        index = pd.to_datetime(df["time_utc"], utc=True)

        if instrument == "mag":
            data = self._expand_vector(df.get("mag_B_GSM"), length=len(df))
            data.index = index
            magnitude = df.get(self.MAG_MAGNITUDE_FIELD)
            data["bavg"] = pd.to_numeric(magnitude, errors="coerce").to_numpy() if magnitude is not None else np.nan
        else:
            data = df.rename(columns={source: target for target, source in self.SWAPI_PARAMETER_FIELDS.items()})
            data.index = index
            for column in self.SWAPI_FIELDS:
                if column not in data.columns:
                    data[column] = np.nan
            data = data[self.SWAPI_FIELDS]

        data = data[fields].apply(pd.to_numeric, errors="coerce")
        return data.sort_index()

    def _expand_vector(self, series, length: int) -> pd.DataFrame:
        """Expand a column of `[bx, by, bz]` lists into separate `bx_gsm`/`by_gsm`/`bz_gsm` columns.

        Entries that are missing or not a 3-element list/tuple (e.g. a malformed record) become
        all-NaN rows rather than raising, so a single bad record does not fail the whole day.

        Parameters
        ----------
        series : pd.Series | None
            The `mag_B_GSM` column, or `None` if the key was absent from every record.
        length : int
            Number of rows to produce when `series` is `None`.
        """
        columns = self.MAG_VECTOR_COLUMNS

        def _row(entry):
            if isinstance(entry, (list, tuple)) and len(entry) == len(columns):
                return list(entry)
            return [np.nan] * len(columns)

        if series is None:
            return pd.DataFrame([[np.nan] * len(columns)] * length, columns=columns)
        return pd.DataFrame(series.map(_row).tolist(), columns=columns)

    def _merge_instrument_data(self, mag_df: pd.DataFrame, swapi_df: pd.DataFrame, day_start: datetime) -> pd.DataFrame:
        """Align the `mag` and `swapi` feeds onto a complete 1-minute UTC grid for one day.

        Neither instrument is natively 1-minute (mag is sampled roughly every 4 seconds, swapi
        roughly every 13-24 seconds), unlike ACE/DSCOVR's already-1-minute-averaged products, so
        each is downsampled with a per-minute mean rather than reindexed or nearest-matched.

        Parameters
        ----------
        mag_df : pd.DataFrame
            Parsed `mag` data for the day (may be empty).
        swapi_df : pd.DataFrame
            Parsed `swapi` data for the day (may be empty).
        day_start : datetime
            Start of the UTC day.

        Returns
        -------
        pd.DataFrame
            One row per minute of the day, with `pdyn` derived from `speed`/`proton_density`.
        """
        complete_range = pd.date_range(start=day_start, periods=1440, freq="1min", tz="UTC")

        mag_1m = mag_df.resample("1min", label="left", closed="left").mean().reindex(complete_range)
        swapi_1m = swapi_df.resample("1min", label="left", closed="left").mean().reindex(complete_range)

        data = pd.concat([mag_1m, swapi_1m], axis=1)
        data.index.name = "t"
        data["pdyn"] = 2e-6 * data["proton_density"].to_numpy() * data["speed"].to_numpy() ** 2

        return data

    def _save_processed_data(self, processed_df: pd.DataFrame, date) -> None:
        """Write one day's processed data to its cache file, merging with any existing file.

        Parameters
        ----------
        processed_df : pd.DataFrame
            One day of processed IMAP data, indexed by UTC timestamp.
        date : datetime.date
            The UTC day being written.
        """
        file_path = self.data_dir / date.strftime("%Y/%m") / f"IMAP_SW_NOWCAST_{date.strftime('%Y%m%d')}.csv"
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")

        try:
            if file_path.exists():
                logger.debug(f"Found previous file for {date}. Loading and combining ...")
                previous_df = self._read_single_file(file_path)

                previous_df.drop("file_name", axis=1, inplace=True)
                processed_df = processed_df.combine_first(previous_df)

            logger.debug(f"Saving processed file for {date}")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            processed_df.to_csv(tmp_path, index=True, header=True)
            tmp_path.replace(file_path)

        except Exception as e:
            logger.error(f"Failed to process file for {date}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

    def _is_api_key_required_response(self, response: requests.Response) -> bool:
        """Return whether `response` is the API's "API key required" error for old data.

        Recognized as an HTTP 400 whose JSON body carries an "API key required" message -
        distinguished from other 400s since retrying, chunking, or waiting would not fix it.
        """
        if response.status_code != 400:
            return False
        try:
            payload = response.json()
        except ValueError:
            return False
        message = payload.get("message", "") if isinstance(payload, dict) else ""
        return "api key required" in message.lower()

    def _is_retryable_download_error(self, error: Exception) -> bool:
        """Return whether `error` represents a transient download failure worth retrying."""
        if isinstance(error, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(error, requests.HTTPError) and error.response is not None:
            status_code = error.response.status_code
            return status_code == 429 or status_code >= 500
        return False

    def _retry_reason(self, error: Exception) -> str:
        """Return a short, log-safe description of a transient failure."""
        if isinstance(error, requests.Timeout):
            return "request timed out"
        if isinstance(error, requests.ConnectionError):
            return "connection failed"
        if isinstance(error, requests.HTTPError) and error.response is not None:
            return f"HTTP {error.response.status_code}"
        return str(error)

    def read(
        self,
        start_time: datetime,
        end_time: datetime,
        download: bool = False,
        propagation: bool = False,
    ) -> pd.DataFrame:
        """
        Read IMAP data for the specified time range.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read. Must be timezone-aware.
        end_time : datetime
            End time of the data to read. Must be timezone-aware.
        download : bool, optional
            Download data on the go, defaults to False.
        propagation : bool, optional
            Propagate the data from L1 to near-Earth, defaults to False.

        Returns
        -------
        :class:`pandas.DataFrame`
            DataFrame containing IMAP Solar Wind data for the requested period.

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
            data={field: nan_data for field in [*self.MAG_FIELDS, *self.SWAPI_FIELDS, "pdyn"]},
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
            file_path = (
                self.data_dir
                / current_time.strftime("%Y/%m")
                / f"IMAP_SW_NOWCAST_{current_time.strftime('%Y%m%d')}.csv"
            )
            file_paths.append(file_path)

            interval_start = current_time
            interval_end = datetime(current_time.year, current_time.month, current_time.day, 23, 59, 59)

            time_intervals.append((interval_start, interval_end))
            current_time += timedelta(days=1)

        return file_paths, time_intervals

    def _read_single_file(self, file_path) -> pd.DataFrame:
        """Read IMAP file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from IMAP file.
        """
        df = pd.read_csv(file_path)

        df["t"] = pd.to_datetime(df["t"], utc=True)
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)

        df["file_name"] = file_path
        df.loc[df["bavg"].isna() & df["temperature"].isna(), "file_name"] = None

        return df

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
        return "propagated from previous IMAP NOWCAST file" if file_date != index_date else row["file_name"]
