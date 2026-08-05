# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Simon Mischel
#
# SPDX-License-Identifier: Apache-2.0

"""Download and read substorm-onset catalogues from SuperMAG."""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep

import pandas as pd
import requests

from swvo.io.base import BaseIO
from swvo.io.substorms.catalogs import (
    SUPERMAG_SUBSTORM_CATALOGS,
    SuperMAGSubstormCatalog,
)
from swvo.io.substorms.catalogs import (
    available_catalogs as get_available_catalogs,
)
from swvo.io.utils import enforce_utc_timezone

logger = logging.getLogger(__name__)


class _TransientSuperMAGEventResponseError(ValueError):
    """Response error that can reasonably succeed when requested again."""


class SubstormsSuperMAG(BaseIO):
    """Reader for the substorm-onset catalogues distributed by SuperMAG.

    The reader retrieves one scientifically distinct catalogue at a time.
    ``"newell"`` is the default; ``"sophie"`` is accepted as an alias for
    SuperMAG's canonical ``"forsyth"`` identifier.

    Parameters
    ----------
    username : str
        Registered SuperMAG username used for data access.
    data_dir : Path | None
        Root directory for SuperMAG caches. If omitted, the directory is read
        from ``SUPERMAG_STREAM_DIR``.
    prefer_env_var : bool, optional
        If ``True``, ``SUPERMAG_STREAM_DIR`` takes precedence over
        ``data_dir``. Defaults to ``False``.

    Raises
    ------
    TypeError
        If ``username`` is not a string.
    ValueError
        If ``username`` is empty or no data directory is configured.
    """

    ENV_VAR_NAME = "SUPERMAG_STREAM_DIR"
    URL = "https://supermag.jhuapl.edu/lib/services/"
    LABEL = "supermag_substorms"

    _DEFAULT_CATALOG = "newell"
    _OUTPUT_COLUMNS = ("mlt", "mlat", "glon", "glat", "catalog", "file_name")
    _TABLE_HEADER = ("<year>", "<month>", "<day>", "<hour>", "<min>", "<mlt>", "<mlat>", "<glon>", "<glat>")
    _MAX_DOWNLOAD_ATTEMPTS = 4
    _RETRY_BACKOFF_SECONDS = 1.0

    def __init__(self, username: str, data_dir: Path | None = None, prefer_env_var: bool = False) -> None:
        if not isinstance(username, str):
            raise TypeError("username must be a string")
        if not username.strip():
            raise ValueError("username must not be empty")
        super().__init__(data_dir, prefer_env_var)
        self.username = username

    def available_catalogs(self) -> pd.DataFrame:
        """Return metadata for the supported substorm-onset catalogues."""

        return get_available_catalogs()

    def download_and_process(
        self,
        start_time: datetime,
        end_time: datetime,
        reprocess_files: bool = False,
        *,
        catalog: str = _DEFAULT_CATALOG,
    ) -> None:
        """Download validated annual files for one SuperMAG onset catalogue.

        SuperMAG's self-documented ASCII response is retained so that its data
        revision, generation date, caveats, and acknowledgement text remain
        available with the cached events.

        Parameters
        ----------
        start_time : datetime
            First requested instant. Naive values are interpreted as UTC.
        end_time : datetime
            Last requested instant. Naive values are interpreted as UTC.
        reprocess_files : bool, optional
            Replace existing annual files. This is useful for the continuously
            revised index-derived catalogues. Defaults to ``False``.
        catalog : str, optional
            Canonical catalogue name or documented alias. Defaults to
            ``"newell"``.

        Raises
        ------
        TypeError
            If ``catalog`` is not a string.
        ValueError
            If the interval or catalogue is invalid, or SuperMAG returns a
            permanent error.
        requests.RequestException
            If a non-retryable request fails.
        """

        if start_time >= end_time:
            raise ValueError("start_time must be before end_time")

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)
        selected_catalog = self._resolve_catalog(catalog)
        self._resolved_urls = []
        file_paths, years = self._get_cache_file_list(start_time, end_time, selected_catalog)

        failed_years: list[int] = []
        for file_path, year in zip(file_paths, years, strict=True):
            if file_path.exists() and not reprocess_files:
                try:
                    self._read_single_file(file_path, selected_catalog, year)
                except (OSError, UnicodeError, ValueError):
                    logger.warning(
                        "Replacing invalid SuperMAG %s event cache for %d",
                        selected_catalog.name,
                        year,
                    )
                else:
                    continue
            try:
                self._download_and_process_single_file(file_path, year, selected_catalog)
            except Exception as error:
                if not self._is_retryable_download_error(error):
                    raise
                failed_years.append(year)
                logger.error(
                    "SuperMAG %s event download failed for %d after %d attempts (%s)",
                    selected_catalog.name,
                    year,
                    self._MAX_DOWNLOAD_ATTEMPTS,
                    self._retry_reason(error),
                )

        if failed_years:
            years_text = ", ".join(str(year) for year in failed_years)
            year_label = "year" if len(failed_years) == 1 else "years"
            warnings.warn(
                f"SuperMAG {selected_catalog.name} event download failed after "
                f"{self._MAX_DOWNLOAD_ATTEMPTS} attempts for {len(failed_years)} "
                f"{year_label}: {years_text}. Re-run the same range to retry failed years.",
                RuntimeWarning,
                stacklevel=2,
            )

    def read(
        self,
        start_time: datetime,
        end_time: datetime,
        download: bool = False,
        *,
        catalog: str = _DEFAULT_CATALOG,
    ) -> pd.DataFrame:
        """Read one SuperMAG substorm-onset catalogue.

        Parameters
        ----------
        start_time : datetime
            First onset time to include. Naive values are interpreted as UTC.
        end_time : datetime
            Last onset time to include. Naive values are interpreted as UTC.
        download : bool, optional
            Download missing annual files and replace corrupt files. Defaults
            to ``False``.
        catalog : str, optional
            Canonical catalogue name or documented alias. Defaults to
            ``"newell"``.

        Returns
        -------
        pandas.DataFrame
            Sparse onset records indexed by timezone-aware UTC ``onset``.
            Columns are ``mlt`` (hours), ``mlat`` (degrees), ``glon``
            (degrees), ``glat`` (degrees), canonical ``catalog``, and
            ``file_name`` provenance.

        Raises
        ------
        TypeError
            If ``catalog`` is not a string.
        ValueError
            If the interval or catalogue is invalid, a cache is corrupt while
            downloading is disabled, or a downloaded response is invalid.

        Notes
        -----
        The location in index-derived catalogues is the location of the
        station contributing to SML at onset, not necessarily the physical
        auroral breakup location.
        """

        if start_time > end_time:
            raise ValueError("start_time must be before or equal to end_time")

        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)
        selected_catalog = self._resolve_catalog(catalog)
        if download:
            self._resolved_urls = []

        file_paths, years = self._get_cache_file_list(start_time, end_time, selected_catalog)
        frames: list[pd.DataFrame] = []

        for file_path, year in zip(file_paths, years, strict=True):
            if not file_path.exists():
                if download:
                    self._download_and_process_single_file(file_path, year, selected_catalog)
                else:
                    warnings.warn(
                        f"File {file_path} not found. Re-run with download=True to retrieve it.",
                        stacklevel=2,
                    )
                    continue

            try:
                frame = self._read_single_file(file_path, selected_catalog, year)
            except (OSError, UnicodeError, ValueError) as error:
                if not download:
                    raise ValueError(
                        f"Cannot read SuperMAG event cache {file_path}. "
                        "Re-run with download=True to replace the corrupt file."
                    ) from error
                self._download_and_process_single_file(file_path, year, selected_catalog)
                frame = self._read_single_file(file_path, selected_catalog, year)

            frame["catalog"] = selected_catalog.name
            frame["file_name"] = file_path
            frames.append(frame)

        if frames:
            result = pd.concat(frames).sort_index(kind="stable")
            result = result.loc[(result.index >= start_time) & (result.index <= end_time)]
        else:
            result = self._empty_events()

        return result.loc[:, self._OUTPUT_COLUMNS]

    def _download_and_process_single_file(
        self,
        file_path: Path,
        year: int,
        catalog: SuperMAGSubstormCatalog,
    ) -> None:
        """Download, validate, and atomically install one annual cache."""

        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        # SuperMAG includes events exactly at both request boundaries. Ending
        # at the last second of the year avoids duplicating a New Year's event
        # in two adjacent annual cache files.
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
        params = {
            "service": "substorms",
            "downloadtype": "substorm_list",
            "user": self.username,
            "fmt": "ascii",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "list": catalog.name,
        }

        self._record_sanitized_url(params)
        for attempt in range(1, self._MAX_DOWNLOAD_ATTEMPTS + 1):
            logger.debug(
                "Downloading SuperMAG %s events for %d (attempt %d/%d)",
                catalog.name,
                year,
                attempt,
                self._MAX_DOWNLOAD_ATTEMPTS,
            )
            try:
                response = requests.get(self.URL, params=params, timeout=30)
                response.raise_for_status()
                parsed = self._process_response_text(response.text, catalog)
                self._validate_cache_year(parsed, year)
                tmp_path.write_text(response.text, encoding="utf-8")
                tmp_path.replace(file_path)
                return
            except Exception as error:
                tmp_path.unlink(missing_ok=True)
                if not self._is_retryable_download_error(error) or attempt == self._MAX_DOWNLOAD_ATTEMPTS:
                    raise

                backoff = self._RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1)
                logger.warning(
                    "Transient SuperMAG %s event failure for %d (%s); retrying in %.1f seconds (attempt %d/%d)",
                    catalog.name,
                    year,
                    self._retry_reason(error),
                    backoff,
                    attempt + 1,
                    self._MAX_DOWNLOAD_ATTEMPTS,
                )
                sleep(backoff)

    def _record_sanitized_url(self, params: dict[str, str]) -> None:
        """Record request provenance without exposing the SuperMAG username."""

        public_params = {**params, "user": "<redacted>"}
        resolved_url = requests.Request("GET", self.URL, params=public_params).prepare().url
        if resolved_url is None:
            raise ValueError("Could not resolve the SuperMAG event request URL")
        self._record_url(resolved_url)

    def _get_cache_file_list(
        self,
        start_time: datetime,
        end_time: datetime,
        catalog: SuperMAGSubstormCatalog,
    ) -> tuple[list[Path], list[int]]:
        """Return annual cache paths and their matching years."""

        years = list(range(start_time.year, end_time.year + 1))
        catalog_dir = self.data_dir / "substorms" / catalog.name
        file_paths = [catalog_dir / f"SuperMAG_SUBSTORMS_{catalog.name.upper()}_{year}.txt" for year in years]
        return file_paths, years

    def _read_single_file(
        self,
        file_path: Path,
        catalog: SuperMAGSubstormCatalog,
        year: int,
    ) -> pd.DataFrame:
        """Parse one cached self-documented SuperMAG response."""

        frame = self._process_response_text(file_path.read_text(encoding="utf-8"), catalog)
        self._validate_cache_year(frame, year)
        return frame

    def _process_response_text(
        self,
        text: str,
        catalog: SuperMAGSubstormCatalog,
    ) -> pd.DataFrame:
        """Parse and validate a self-documented SuperMAG event response."""

        response_text = text.strip()
        if not response_text:
            raise _TransientSuperMAGEventResponseError("SuperMAG returned an empty response")
        if response_text.startswith("ERROR"):
            first_line = response_text.splitlines()[0]
            raise ValueError(f"SuperMAG {first_line}")

        lines = response_text.splitlines()
        header_index = next(
            (index for index, line in enumerate(lines) if tuple(line.strip().lower().split()) == self._TABLE_HEADER),
            None,
        )
        if header_index is None:
            raise ValueError(f"SuperMAG {catalog.name} event response does not contain the expected table header")

        records: list[list[str]] = []
        for line_number, line in enumerate(lines[header_index + 1 :], start=header_index + 2):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != len(self._TABLE_HEADER):
                raise ValueError(
                    f"Malformed SuperMAG event record on response line {line_number}: "
                    f"expected {len(self._TABLE_HEADER)} fields, found {len(fields)}"
                )
            records.append(fields)

        if not records:
            return self._empty_measurements()

        columns = ("year", "month", "day", "hour", "minute", "mlt", "mlat", "glon", "glat")
        frame = pd.DataFrame(records, columns=columns)
        for column in columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.isna().any():
                raise ValueError(f"SuperMAG event response contains a non-numeric {column} value")
            frame[column] = numeric

        # SuperMAG labels the Liou field as MLT but serves the original values
        # in magnetic-longitude degrees. Liou (2010) reports MLT in hours, so
        # normalize that one catalogue by the standard 15 degrees per hour.
        if catalog.name == "liou":
            frame["mlt"] = frame["mlt"] / 15

        timestamps = pd.to_datetime(
            {
                "year": frame.pop("year"),
                "month": frame.pop("month"),
                "day": frame.pop("day"),
                "hour": frame.pop("hour"),
                "minute": frame.pop("minute"),
            },
            utc=True,
            errors="coerce",
        )
        if timestamps.isna().any():
            raise ValueError("SuperMAG event response contains an invalid onset time")

        frame.index = pd.DatetimeIndex(timestamps, name="onset")
        return frame.loc[:, ["mlt", "mlat", "glon", "glat"]]

    def _validate_cache_year(self, frame: pd.DataFrame, year: int) -> None:
        """Ensure an annual response contains no records from another year."""

        if any(pd.Timestamp(value).year != year for value in frame.index):
            raise ValueError(f"SuperMAG event response for {year} contains an onset outside that year")

    def _resolve_catalog(self, catalog: str) -> SuperMAGSubstormCatalog:
        """Validate a catalogue name and normalize documented aliases."""

        if not isinstance(catalog, str):
            raise TypeError("catalog must be a string")

        normalized = catalog.strip().lower().replace("-", "_").replace(" ", "_")
        if not normalized:
            raise ValueError("catalog must not be empty")

        for candidate in SUPERMAG_SUBSTORM_CATALOGS:
            if normalized == candidate.name or normalized in candidate.aliases:
                return candidate

        available = ", ".join(candidate.name for candidate in SUPERMAG_SUBSTORM_CATALOGS)
        raise ValueError(f"Unknown SuperMAG substorm catalog: {catalog!r}. Available catalogs: {available}")

    def _empty_measurements(self) -> pd.DataFrame:
        """Return an empty parsed-event table with a UTC onset index."""

        index = pd.DatetimeIndex([], tz=timezone.utc, name="onset")
        return pd.DataFrame(index=index, columns=["mlt", "mlat", "glon", "glat"], dtype=float)

    def _empty_events(self) -> pd.DataFrame:
        """Return an empty public event table with stable column types."""

        frame = self._empty_measurements()
        frame["catalog"] = pd.Series(index=frame.index, dtype="object")
        frame["file_name"] = pd.Series(index=frame.index, dtype="object")
        return frame

    def _is_retryable_download_error(self, error: Exception) -> bool:
        """Return whether ``error`` represents a transient download failure."""

        if isinstance(
            error,
            (
                _TransientSuperMAGEventResponseError,
                requests.Timeout,
                requests.ConnectionError,
            ),
        ):
            return True
        if isinstance(error, requests.HTTPError) and error.response is not None:
            status_code = error.response.status_code
            return status_code == 429 or status_code >= 500
        return False

    def _retry_reason(self, error: Exception) -> str:
        """Return a log-safe failure description without a prepared URL."""

        if isinstance(error, _TransientSuperMAGEventResponseError):
            return str(error)
        if isinstance(error, requests.Timeout):
            return "request timed out"
        if isinstance(error, requests.ConnectionError):
            return "connection failed"
        if isinstance(error, requests.HTTPError) and error.response is not None:
            return f"HTTP {error.response.status_code}"
        return type(error).__name__
