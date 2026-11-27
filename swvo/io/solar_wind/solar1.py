# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling SOLAR-1 (formerly SWFO-L1) Solar Wind data.
"""

import gzip
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

import numpy as np
import pandas as pd
import requests
import xarray as xr

from swvo.io.solar_wind.rtsw import RTSWSolarWindBase
from swvo.io.utils import enforce_utc_timezone

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


class SWSOLAR1(RTSWSolarWindBase):
    """This is a class for the SOLAR-1 Solar Wind data.

    SOLAR-1 is NOAA's operational L1 solar wind monitor. Data comes from two sources:

    - Archive: the NCEI Space Weather Portal. The magnetic field is the MAG Level 3 1-minute GSM
      average (``mag-l3_solar1``), read through the portal's ``/values`` API. The plasma moments
      are the SWiPS Level 2 1-minute moments (``swips-l2_solar1``), read from the daily NetCDF
      files listed by the ``/files`` API. NCEI publishes each UTC day at about 02:15 UTC on the
      following day.
    - Real time: the last 24 hours from the SWPC RTSW feed (``source == "SOLAR1"``), which NOAA
      states is equivalent to the operational NetCDF data on NCEI.

    Archive values take precedence over real-time values, and days still holding real-time data
    are re-downloaded once NCEI should have published them (see
    :class:`~swvo.io.solar_wind.rtsw.RTSWSolarWindBase`).

    Read https://archive.data.noaa.gov/satellite-spaceweather/SWFO/docs/SWIPS/SWFO_SWIPS_Provisional_Operational_Data_ReadMe_v1.0.pdf
    for data quality issues.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for the SOLAR-1 Solar Wind data. If not provided, it will be read from the environment variable
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

    Notes
    -----
    Shortly before NCEI publishes a day,
    the newest archive data can be up to about 26 hours old while the feed holds only 24 hours,
    so up to about 2 hours can be missing until the files appear.
    """

    ENV_VAR_NAME = "RT_SW_SOLAR1_STREAM_DIR"

    PORTAL_URL: Final = "https://www.ncei.noaa.gov/cloud-access/space-weather-portal/api/v1/"
    URL = PORTAL_URL
    SATELLITE: Final = "SOLAR-1"

    MAG_PRODUCT: Final = "mag-l3_solar1"
    MAG_PARAMETERS: Final = {
        "bx_gsm": "b_gsm_min_x",
        "by_gsm": "b_gsm_min_y",
        "bz_gsm": "b_gsm_min_z",
        "bavg": "b_gsm_sphr_min_x",
    }
    SWIPS_PRODUCT: Final = "swips-l2_solar1"

    DATA_START: Final = datetime(2026, 6, 9, tzinfo=timezone.utc)

    FILL_THRESHOLD: Final = -9998.0
    SPEED_RANGE: Final = (180.0, 2510.0)

    RTSW_SOURCE = "SOLAR1"
    REALTIME_WINDOW = timedelta(hours=24)
    ARCHIVE_LATENCY = timedelta(hours=27)
    FILE_PREFIX = "SOLAR1_SW_NOWCAST"

    LABEL = "solar1"

    def _fallback_url(self) -> list[str]:
        return [
            self.PORTAL_URL + "values",
            self.PORTAL_URL + "files",
            self.RTSW_URL + self.RTSW_NAME_MAG,
            self.RTSW_URL + self.RTSW_NAME_WIND,
        ]

    def download_and_process(self, start_time: datetime, end_time: datetime) -> None:
        """
        Download and process SOLAR-1 data, splitting data across midnight into appropriate day files.

        Requests reaching back before `DATA_START` are clipped to it, and requests ending before it
        download nothing, so the days without SOLAR-1 data stay empty (and a following model in
        `read_solar_wind_from_multiple_models` fills them).

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

        Examples
        --------
        >>> reader = SWSOLAR1(data_dir="/path/to/solar1")
        >>> reader.download_and_process(start_time, end_time)
        """
        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)
        if end_time < self.DATA_START:
            logger.warning(f"No SOLAR-1 data before {self.DATA_START:%Y-%m-%d}, nothing to download.")
            return
        if start_time < self.DATA_START:
            logger.warning(f"No SOLAR-1 data before {self.DATA_START:%Y-%m-%d}, downloading from there on.")
            start_time = self.DATA_START
        super().download_and_process(start_time, end_time)

    def _download_archive(self, start_time: datetime, end_time: datetime) -> pd.DataFrame | None:
        """Download MAG L3 and SWiPS L2 data from NCEI for every day in the range.

        Days NCEI has not published yet are skipped.

        Raises
        ------
        requests.HTTPError
            If an HTTP request fails.
        """
        days = [d.date() for d in pd.date_range(start=start_time.date(), end=end_time.date(), freq="D")]
        swips_links = self._list_swips_files(days[0], days[-1])

        frames = []
        for day in days:
            mag = self._download_mag(day)
            link = swips_links.get(day)
            plasma = self._download_swips(link) if link else None
            if mag.empty and plasma is None:
                logger.info(f"No SOLAR-1 archive data on NCEI for {day} yet")
                continue
            frames.append(pd.concat([mag, plasma], axis=1, sort=True) if plasma is not None else mag)

        return self._finalize(pd.concat(frames)) if frames else None

    def _download_mag(self, day: date) -> pd.DataFrame:
        """Read one day of MAG L3 1-minute GSM data through the portal `/values` API."""
        day_start = datetime.combine(day, datetime.min.time())
        params = {
            "start_time": day_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": (day_start + timedelta(days=1) - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S"),
            "parameters": ";".join(f"{self.SATELLITE}:{self.MAG_PRODUCT}:{p}" for p in self.MAG_PARAMETERS.values()),
            "format": "json",
            "time_format": "iso",
        }
        url = self.PORTAL_URL + "values"
        self._record_url(requests.Request("GET", url, params=params).prepare().url)  # ty: ignore[invalid-argument-type]
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()

        status = payload.get("status", {})
        if status.get("code", 200) != 200:
            raise requests.HTTPError(f"NCEI portal returned {status.get('code')}: {status.get('message')}")

        data = pd.DataFrame(payload.get("data") or {})
        if data.empty or "time" not in data.columns:
            return pd.DataFrame(columns=self.MAG_FIELDS, index=pd.DatetimeIndex([], tz="UTC"))

        data.index = pd.DatetimeIndex(pd.to_datetime(data.pop("time"), utc=True))
        columns = {f"{self.MAG_PRODUCT}.{p}": field for field, p in self.MAG_PARAMETERS.items()}
        data = data.rename(columns=columns).reindex(columns=self.MAG_FIELDS).apply(pd.to_numeric, errors="coerce")
        return data[~data.index.duplicated(keep="last")]

    def _list_swips_files(self, first_day: date, last_day: date) -> dict[date, str]:
        """Map each day to the download link of its SWiPS L2 file (the latest one if reprocessed)."""
        params = {
            "start_time": f"{first_day:%Y-%m-%d}T00:00:00",
            "end_time": f"{last_day:%Y-%m-%d}T23:59:59",
            "sat": self.SATELLITE,
            "prod": self.SWIPS_PRODUCT,
            "order": "asc",
        }
        url = self.PORTAL_URL + "files"
        self._record_url(requests.Request("GET", url, params=params).prepare().url)  # ty: ignore[invalid-argument-type]
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        links: dict[date, tuple[str, str]] = {}
        for record in response.json().get("data") or []:
            day = date.fromisoformat(record["time_coverage_start"][:10])
            if first_day <= day <= last_day:
                # File ids embed the processing time, so the largest id is the latest version.
                if day not in links or record["id"] > links[day][0]:
                    links[day] = (record["id"], record["file_link"])
        return {day: link for day, (_, link) in links.items()}

    def _download_swips(self, link: str) -> pd.DataFrame:
        """Download one SWiPS L2 daily file and return the 1-minute proton moments."""
        self._record_url(link)
        response = requests.get(link, timeout=60)
        response.raise_for_status()

        with TemporaryDirectory(prefix="swvo_solar1_") as tmp:
            nc_path = Path(tmp) / "swips_l2.nc"
            nc_path.write_bytes(gzip.decompress(response.content) if link.endswith(".gz") else response.content)
            with xr.open_dataset(nc_path, mask_and_scale=False) as ds:
                return self._parse_swips(ds)

    def _parse_swips(self, ds: xr.Dataset) -> pd.DataFrame:
        """Extract proton speed, density and temperature from a SWiPS L2 dataset.

        The `*_uncorr` moments are used, since the `*_corr` variables are currently unused and
        filled. Moments are time-tagged with the sweep start time floored to the minute, which
        matches the SWPC RTSW time tags. If several sweeps fall into one minute, the last valid
        one is kept.
        """
        velocity = ds["proton_v_uncorr_gse"].to_numpy().astype(float)
        velocity[velocity <= self.FILL_THRESHOLD] = np.nan
        data = pd.DataFrame(
            {
                "speed": np.linalg.norm(velocity, axis=1),
                "proton_density": ds["proton_n_uncorr"].to_numpy().astype(float),
                "temperature": ds["proton_t_uncorr"].to_numpy().astype(float),
            },
            index=pd.DatetimeIndex(ds["sweep_start_time"].to_numpy()).tz_localize("UTC").floor("1min"),
        )
        data = self._clean(data)
        return data.groupby(level=0).last()

    def _clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Mask fill values and, following the SWiPS ReadMe, drop whole sweeps with any bad moment.

        A sweep (1-minute row) is dropped if any plasma moment is a fill value, missing,
        non-positive (the processing writes zeros next to fill velocities), or the speed is
        outside the instrument range `SPEED_RANGE`.
        """
        data = data.copy()
        for field in [f for f in self.MAG_FIELDS if f in data.columns]:
            data.loc[data[field] <= self.FILL_THRESHOLD, field] = np.nan

        plasma = [f for f in self.PLASMA_FIELDS if f in data.columns]
        if plasma:
            values = data[plasma]
            bad = (values <= 0).any(axis=1) | values.isna().any(axis=1)
            if "speed" in plasma:
                bad |= (data["speed"] < self.SPEED_RANGE[0]) | (data["speed"] > self.SPEED_RANGE[1])
            data.loc[bad, plasma] = np.nan
        return data
