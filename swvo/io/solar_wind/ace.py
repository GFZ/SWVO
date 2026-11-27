# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling ACE Solar Wind data.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import requests

from swvo.io.solar_wind.rtsw import RTSWSolarWindBase
from swvo.io.utils import enforce_utc_timezone

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


class SWACE(RTSWSolarWindBase):
    """This is a class for the ACE Solar Wind data.

    Data older than `REALTIME_WINDOW` (2 hours) comes from the SOHO daily archive files, the most
    recent 2 hours from the SWPC RTSW feed (``source == "ACE"``). Archive values take precedence
    over real-time values. See :class:`~swvo.io.solar_wind.rtsw.RTSWSolarWindBase`.

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

    RTSW_SOURCE = "ACE"
    # The SOHO daily files lag by 1 to 2 hours, so the last 2 hours come from the RTSW feed.
    REALTIME_WINDOW = timedelta(hours=2)
    ARCHIVE_LATENCY = timedelta(hours=3)
    FILE_PREFIX = "ACE_SW_NOWCAST"

    SWEPAM_FIELDS = RTSWSolarWindBase.PLASMA_FIELDS

    LABEL = "ace"

    def _fallback_url(self) -> list[str]:
        """Source URL templates, one per file (`{date}` is unresolved until a download has run)."""
        return [
            self.URL + self.NAME_MAG,  # ty: ignore[unsupported-operator]
            self.URL + self.NAME_SWEPAM,  # ty: ignore[unsupported-operator]
            self.RTSW_URL + self.RTSW_NAME_MAG,
            self.RTSW_URL + self.RTSW_NAME_WIND,
        ]

    def _download_archive(self, start_time: datetime, end_time: datetime) -> pd.DataFrame | None:
        """Download and process the SOHO daily MAG and SWEPAM files for every day in the range.

        Raises
        ------
        requests.HTTPError
            If an HTTP request fails.
        FileNotFoundError
            If a downloaded file is empty.
        """
        temporary_dir = Path(mkdtemp(prefix="swvo_ace_"))
        try:
            days = []
            for date in pd.date_range(start=start_time.date(), end=end_time.date(), freq="D"):
                request_date = date.to_pydatetime().replace(tzinfo=timezone.utc)
                mag_file_name = self._dated_file_name(self.NAME_MAG, request_date)
                swepam_file_name = self._dated_file_name(self.NAME_SWEPAM, request_date)

                self._download(temporary_dir, mag_file_name)
                self._download(temporary_dir, swepam_file_name)

                logger.debug(f"Processing ACE source files for {request_date.date()} ...")
                days.append(self._process_single_file(temporary_dir, mag_file_name, swepam_file_name))
        except Exception as e:
            logger.error(f"Failed to download and process: {e}")
            raise
        finally:
            rmtree(temporary_dir, ignore_errors=True)

        return self._finalize(pd.concat(days)) if days else None

    def _download(self, temporary_dir: Path, file_name: str, base_url: str | None = None) -> None:
        """Download a file from ACE server.

        Parameters
        ----------
        temporary_dir : Path
            Temporary directory to store the downloaded file.
        file_name : str
            Name of the file to download.
        base_url : str, optional
            Base URL of the server. Defaults to the SOHO archive (`URL`).

        Raises
        ------
        requests.HTTPError
            If the HTTP request fails.
        FileNotFoundError
            If the downloaded file is empty.
        """
        url = (base_url or self.URL) + file_name  # ty: ignore[unsupported-operator]
        logger.debug(f"Downloading file {url} ...")
        self._record_url(url)
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(temporary_dir / file_name, "wb") as f:
            f.write(response.content)

        if (temporary_dir / file_name).stat().st_size == 0:
            raise FileNotFoundError(f"Error while downloading file: {url}!")

    def _dated_file_name(self, template: str, request_time: datetime) -> str:
        request_time = enforce_utc_timezone(request_time)
        return template.format(date=request_time)

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
