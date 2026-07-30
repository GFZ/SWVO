# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Dmitrii Gurev
# SPDX-FileContributor: Simon Mischel
#
# SPDX-License-Identifier: Apache-2.0

"""Read the standard SuperMAG auroral electrojet indices."""

from __future__ import annotations

import json
import logging
import re
import warnings
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import requests

from swvo.io.base import BaseIO
from swvo.io.utils import enforce_utc_timezone

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


class _TransientSuperMAGResponseError(ValueError):
    """Response error that can reasonably succeed when requested again."""


class SMESuperMAG(BaseIO):
    """Reader for the SuperMAG SME, SML, and SMU indices.

    The legacy default remains SME only. Pass ``variables="all"`` or an
    ordered selection to :meth:`read` to retrieve SML and SMU as well.

    Parameters
    ----------
    username : str
        SuperMAG username used for authenticated data access. Register at the
        SuperMAG website to obtain one.
    data_dir : Path | None
        Directory containing daily processed files. If omitted, the path is
        read from ``SUPERMAG_STREAM_DIR``.
    prefer_env_var : bool, optional
        If ``True``, the environment variable takes precedence over
        ``data_dir``. Defaults to ``False``.

    Raises
    ------
    ValueError
        If neither ``data_dir`` nor ``SUPERMAG_STREAM_DIR`` is available.
    """

    ENV_VAR_NAME = "SUPERMAG_STREAM_DIR"
    URL = "https://supermag.jhuapl.edu/services/indices.php"
    LABEL = "supermag"

    _VARIABLES = ("sme", "sml", "smu")
    _DEFAULT_VARIABLES = ("sme",)
    _RESPONSE_COLUMNS = {"SME": "sme", "SML": "sml", "SMU": "smu"}
    _FILL_VALUE_THRESHOLD = 999998
    _MAX_DOWNLOAD_ATTEMPTS = 4
    _RETRY_BACKOFF_SECONDS = 1.0

    def __init__(self, username: str, data_dir: Path | None = None, prefer_env_var: bool = False) -> None:
        super().__init__(data_dir, prefer_env_var)
        self.username = username

    def download_and_process(self, start_time: datetime, end_time: datetime, reprocess_files: bool = False) -> None:
        """Download complete daily SME/SML/SMU files.

        Existing complete files are retained unless ``reprocess_files`` is
        true. A legacy SME-only file is considered incomplete and is upgraded
        to the complete schema. Transient failures are retried per day. If all
        attempts for a day fail, batch processing warns, leaves that cache
        untouched, and continues with subsequent days.

        Parameters
        ----------
        start_time : datetime
            First requested instant. Naive values are interpreted as UTC.
        end_time : datetime
            Last requested instant. Naive values are interpreted as UTC.
        reprocess_files : bool, optional
            Replace complete cached files as well. Defaults to ``False``.

        Raises
        ------
        ValueError
            If the time range is invalid or SuperMAG returns a permanent error,
            such as an invalid username.
        requests.RequestException
            If a non-retryable request fails. Existing cache files remain
            untouched.
        """
        if start_time >= end_time:
            raise ValueError("start_time must be before end_time")

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)
        self._resolved_urls = []
        file_paths, time_intervals = self._get_processed_file_list(start_time, end_time)

        failed_dates = []
        for file_path, time_interval in zip(file_paths, time_intervals, strict=True):
            if file_path.exists() and not reprocess_files and self._cache_contains(file_path, self._VARIABLES):
                continue
            try:
                self._download_and_process_single_file(file_path, time_interval)
            except Exception as error:
                if not self._is_retryable_download_error(error):
                    raise
                failed_dates.append(time_interval.date())
                logger.error(
                    "SuperMAG download failed for %s after %d attempts (%s)",
                    time_interval.date(),
                    self._MAX_DOWNLOAD_ATTEMPTS,
                    self._retry_reason(error),
                )

        if failed_dates:
            dates = ", ".join(str(date) for date in failed_dates)
            day_label = "day" if len(failed_dates) == 1 else "days"
            warnings.warn(
                f"SuperMAG download failed after {self._MAX_DOWNLOAD_ATTEMPTS} attempts "
                f"for {len(failed_dates)} {day_label}: {dates}. Re-run the same range to retry failed days.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _download_and_process_single_file(self, file_path: Path, time_interval: datetime) -> None:
        """Download one day and atomically replace its processed cache."""
        tmp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
        params = {
            "python": "",
            "nohead": "",
            "logon": self.username,
            "start": time_interval.strftime("%Y-%m-%dT%H:%M"),
            "extent": int(timedelta(days=1).total_seconds()),
            "indices": ",".join(self._VARIABLES),
        }

        # Do not log the prepared request URL: it contains the SuperMAG username.
        for attempt in range(1, self._MAX_DOWNLOAD_ATTEMPTS + 1):
            logger.debug(
                "Downloading SuperMAG indices for %s (attempt %d/%d)",
                time_interval.date(),
                attempt,
                self._MAX_DOWNLOAD_ATTEMPTS,
            )
            try:
                resolved_url = requests.Request("GET", self.URL, params=params).prepare().url
                if resolved_url is None:
                    raise ValueError("Could not resolve the SuperMAG request URL")
                self._record_url(resolved_url)
                response = requests.get(self.URL, params=params, timeout=10)
                response.raise_for_status()
                processed_df = self._process_response_text(response.text)
                processed_df.to_csv(tmp_path, index=True, header=True)
                tmp_path.replace(file_path)
                return
            except Exception as error:
                tmp_path.unlink(missing_ok=True)
                if not self._is_retryable_download_error(error) or attempt == self._MAX_DOWNLOAD_ATTEMPTS:
                    raise

                backoff = self._RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1)
                logger.warning(
                    "Transient SuperMAG failure for %s (%s); retrying in %.1f seconds (attempt %d/%d)",
                    time_interval.date(),
                    self._retry_reason(error),
                    backoff,
                    attempt + 1,
                    self._MAX_DOWNLOAD_ATTEMPTS,
                )
                sleep(backoff)

    def _get_processed_file_list(self, start_time: datetime, end_time: datetime) -> tuple[list[Path], list[datetime]]:
        """Return daily cache paths and matching midnight request times."""
        file_paths = []
        time_intervals = []

        current_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        final_midnight = end_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        while current_time < final_midnight:
            file_paths.append(self.data_dir / f"SuperMAG_SME_{current_time:%Y%m%d}.csv")
            time_intervals.append(current_time)
            current_time += timedelta(days=1)

        return file_paths, time_intervals

    def _process_single_file(self, file_path: Path) -> pd.DataFrame:
        """Process a downloaded SuperMAG response stored in ``file_path``."""
        return self._process_response_text(file_path.read_text())

    def _process_response_text(self, text: str) -> pd.DataFrame:
        """Parse a SuperMAG indices response into the complete cache schema."""
        response_text = text.strip()
        if not response_text:
            raise _TransientSuperMAGResponseError("SuperMAG returned an empty response")
        if response_text.startswith("ERROR"):
            first_line = response_text.splitlines()[0]
            raise ValueError(f"SuperMAG {first_line}")

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as direct_error:
            # Some server/proxy responses wrap the JSON in surrounding text.
            match = re.search(r"(\[.*\]|\{.*\})", response_text, re.S)
            if match is None:
                raise ValueError("No JSON object or array found in SuperMAG response") from direct_error
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError as wrapped_error:
                raise ValueError("Malformed JSON in SuperMAG response") from wrapped_error

        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list) and not data:
            raise _TransientSuperMAGResponseError("SuperMAG returned no records")
        if not isinstance(data, list):
            raise ValueError("SuperMAG response must contain at least one record")

        df = pd.DataFrame(data)
        required_columns = {"tval", *self._RESPONSE_COLUMNS}
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            raise ValueError(f"SuperMAG response is missing required fields: {', '.join(missing_columns)}")

        timestamps = pd.to_datetime(pd.to_numeric(df["tval"], errors="coerce"), unit="s", utc=True, errors="coerce")
        if timestamps.isna().any():
            raise ValueError("SuperMAG response contains an invalid timestamp")

        df = df.rename(columns=self._RESPONSE_COLUMNS)
        for variable in self._VARIABLES:
            df[variable] = pd.to_numeric(df[variable], errors="coerce")
            df.loc[df[variable] >= self._FILL_VALUE_THRESHOLD, variable] = np.nan

        processed = df.loc[:, self._VARIABLES].copy()
        processed.index = pd.DatetimeIndex(timestamps, name="timestamp")
        return processed

    @staticmethod
    def _is_retryable_download_error(error: Exception) -> bool:
        """Return whether ``error`` represents a transient download failure."""
        if isinstance(error, (_TransientSuperMAGResponseError, requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(error, requests.HTTPError) and error.response is not None:
            status_code = error.response.status_code
            return status_code == 429 or status_code >= 500
        return False

    @staticmethod
    def _retry_reason(error: Exception) -> str:
        """Return a log-safe failure description without a prepared URL."""
        if isinstance(error, _TransientSuperMAGResponseError):
            return str(error)
        if isinstance(error, requests.Timeout):
            return "request timed out"
        if isinstance(error, requests.ConnectionError):
            return "connection failed"
        if isinstance(error, requests.HTTPError) and error.response is not None:
            return f"HTTP {error.response.status_code}"
        return type(error).__name__

    def read(
        self,
        start_time: datetime,
        end_time: datetime,
        download: bool = False,
        variables: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Read SuperMAG auroral electrojet indices.

        Parameters
        ----------
        start_time : datetime
            First requested instant. Naive values are interpreted as UTC.
        end_time : datetime
            Last requested instant. Naive values are interpreted as UTC.
        download : bool, optional
            Download missing files and upgrade incomplete caches. Defaults to
            ``False``.
        variables : str | Iterable[str] | None, optional
            Variables to return. ``None`` preserves the legacy ``sme`` output,
            ``"all"`` returns ``sme``, ``sml``, and ``smu``, and an ordered
            name or iterable returns a subset in caller order. Duplicate names
            are removed after their first occurrence.

        Returns
        -------
        pandas.DataFrame
            Selected one-minute indices in nT, followed by ``file_name``.
            The UTC index covers the requested interval using the reader's
            established one-minute boundary tolerance.

        Raises
        ------
        TypeError
            If ``variables`` or one of its entries is not a string.
        ValueError
            If the interval is invalid, the selection is empty or unknown, or
            a requested variable is absent from a cache and downloading is
            disabled.

        Examples
        --------
        ``reader.read(start, end)`` returns the legacy SME schema.
        ``reader.read(start, end, variables="all")`` returns all three indices.
        ``reader.read(start, end, variables=["smu", "sml"])`` preserves that
        requested column order.
        """
        selected_variables = self._resolve_variables(variables)
        if start_time > end_time:
            raise ValueError("start_time must be before end_time")

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)
        if download:
            self._resolved_urls = []
        file_paths, time_intervals = self._get_processed_file_list(start_time, end_time)
        index = pd.date_range(
            datetime(start_time.year, start_time.month, start_time.day),
            datetime(end_time.year, end_time.month, end_time.day, 23, 59),
            freq=timedelta(minutes=1),
            tz=timezone.utc,
        )
        data_out = pd.DataFrame(index=index, columns=[*selected_variables, "file_name"])

        for file_path, time_interval in zip(file_paths, time_intervals, strict=True):
            if not file_path.exists():
                if download:
                    self._download_and_process_single_file(file_path, time_interval)
                else:
                    warnings.warn(f"File {file_path} not found")
                    continue

            try:
                df_one_file = self._read_single_file(file_path)
            except (OSError, ValueError, pd.errors.ParserError) as error:
                if not download:
                    raise ValueError(
                        f"Cannot read SuperMAG cache {file_path}. "
                        "Re-run with download=True to replace the corrupt file."
                    ) from error
                self._download_and_process_single_file(file_path, time_interval)
                df_one_file = self._read_single_file(file_path)

            required_cache_variables = (
                self._VARIABLES
                if any(variable not in self._DEFAULT_VARIABLES for variable in selected_variables)
                else selected_variables
            )
            missing_variables = [
                variable for variable in required_cache_variables if variable not in df_one_file.columns
            ]
            if missing_variables:
                if not download:
                    missing = ", ".join(missing_variables)
                    raise ValueError(
                        f"SuperMAG cache {file_path} does not contain requested fields: {missing}. "
                        "Re-run with download=True to upgrade the legacy cache."
                    )
                self._download_and_process_single_file(file_path, time_interval)
                df_one_file = self._read_single_file(file_path)

            selected = df_one_file.loc[:, selected_variables].copy()
            selected["file_name"] = file_path
            selected.loc[selected[selected_variables].isna().all(axis=1), "file_name"] = None
            data_out = selected.combine_first(data_out)

        data_out = data_out.truncate(
            before=start_time - timedelta(minutes=0.9999),
            after=end_time + timedelta(minutes=0.9999),
        )
        return data_out.loc[:, [*selected_variables, "file_name"]]

    def _resolve_variables(self, variables: str | Iterable[str] | None) -> list[str]:
        """Validate and normalize a variable selection."""
        if variables is None:
            requested = list(self._DEFAULT_VARIABLES)
        elif isinstance(variables, str):
            requested = list(self._VARIABLES) if variables == "all" else [variables]
        else:
            try:
                requested = list(variables)
            except TypeError as error:
                raise TypeError("variables must be a string, an iterable of strings, or None") from error

        if not requested:
            raise ValueError("variables must contain at least one variable")
        if any(not isinstance(variable, str) for variable in requested):
            raise TypeError("every variables entry must be a string")

        unknown = list(dict.fromkeys(variable for variable in requested if variable not in self._VARIABLES))
        if unknown:
            available = ", ".join(self._VARIABLES)
            raise ValueError(f"Unknown SuperMAG variables: {', '.join(unknown)}. Available variables: {available}")

        return list(dict.fromkeys(requested))

    @staticmethod
    def _cache_contains(file_path: Path, variables: Iterable[str]) -> bool:
        """Return whether a readable cache contains valid ``variables``."""
        try:
            df = SMESuperMAG._read_single_file(file_path)
        except (OSError, ValueError, pd.errors.ParserError):
            return False
        return set(variables).issubset(df.columns)

    @staticmethod
    def _read_single_file(file_path: Path) -> pd.DataFrame:
        """Read a daily processed SuperMAG cache."""
        df = pd.read_csv(file_path)
        if "timestamp" not in df.columns:
            raise ValueError("SuperMAG cache has no timestamp column")

        timestamps = pd.to_datetime(df.pop("timestamp"), utc=True, errors="coerce")
        if timestamps.isna().any():
            raise ValueError("SuperMAG cache contains an invalid timestamp")
        if df.empty:
            raise ValueError("SuperMAG cache contains no records")

        for variable in SMESuperMAG._VARIABLES:
            if variable not in df.columns:
                continue
            numeric = pd.to_numeric(df[variable], errors="coerce")
            if (df[variable].notna() & numeric.isna()).any():
                raise ValueError(f"SuperMAG cache contains non-numeric {variable} values")
            df[variable] = numeric
            df.loc[df[variable] >= SMESuperMAG._FILL_VALUE_THRESHOLD, variable] = np.nan

        df.index = pd.DatetimeIndex(timestamps)
        df.index.name = None
        return df
