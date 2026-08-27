# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling NOAA SWPC WSA-ENLIL solar wind model data.
"""

import logging
import multiprocessing
import tarfile
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import rmtree
from typing import Literal, Optional

import astropy.units as u
import numpy as np
import pandas as pd
import requests
import richpool
import xarray as xr
from astropy.coordinates import CartesianRepresentation
from astropy.time import Time
from sunpy.coordinates import GeocentricSolarMagnetospheric, HeliographicStonyhurst

from swvo.io.base import BaseIO
from swvo.io.utils import enforce_utc_timezone

logger = logging.getLogger(__name__)

logging.captureWarnings(True)

PROTON_MASS_KG = 1.67262192369e-27
METERS_PER_SECOND_TO_KM_PER_SECOND = 1e-3
KG_PER_M3_TO_PER_CM3 = 1e-6 / PROTON_MASS_KG

_VECTOR_TRANSFORM_SCALE = 1e9 * u.m / u.T

EARTH_VARIABLES = (
    "Earth_TIME",
    "Earth_X1",
    "Earth_X2",
    "Earth_X3",
    "Earth_B1",
    "Earth_B2",
    "Earth_B3",
    "Earth_V1",
    "Earth_V2",
    "Earth_V3",
    "Earth_Density",
    "Earth_Temperature",
)

OUTPUT_COLUMNS = [
    "bx_gsm",
    "by_gsm",
    "bz_gsm",
    "bavg",
    "speed",
    "proton_density",
    "temperature",
    "pdyn",
    "file_name",
]

Mode = Literal["bkg", "cme"]


def _spherical_vector_to_cartesian(
    theta: np.ndarray, phi: np.ndarray, v_r: np.ndarray, v_theta: np.ndarray, v_phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical (r, theta, phi) vector components to cartesian.

    Parameters
    ----------
    theta : np.ndarray
        Co-latitude in radians.
    phi : np.ndarray
        Longitude in radians.
    v_r, v_theta, v_phi : np.ndarray
        Radial, co-latitudinal and longitudinal vector components.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Cartesian x, y, z components in the same frame as the input spherical grid (HEEQ).
    """
    x = v_r * np.sin(theta) * np.cos(phi) + v_theta * np.cos(theta) * np.cos(phi) - v_phi * np.sin(phi)
    y = v_r * np.sin(theta) * np.sin(phi) + v_theta * np.cos(theta) * np.sin(phi) + v_phi * np.cos(phi)
    z = v_r * np.cos(theta) - v_theta * np.sin(theta)
    return x, y, z


class SWENLIL(BaseIO):
    """
    Base class for handling NOAA SWPC WSA-ENLIL model data.

    ENLIL is run in two modes: `bkg` (ambient background, one run archived
    per day at 00:00 UTC) and `cme` (CME simulation runs, archived only when
    a CME was modeled; zero, one, or several runs can exist for a given day,
    each at its own run time). Both modes are discovered via the NCEI space
    weather portal files API rather than a fixed URL pattern, since `cme`
    run times aren't predictable.

    Instantiate :class:`SWENLIL_BKG` or :class:`SWENLIL_CME` instead.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for the ENLIL data. If not provided, it will be read from the environment variable.

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.
    """

    ENV_VAR_NAME = "SW_ENLIL_STREAM_DIR"

    API_URL = "https://www.ncei.noaa.gov/cloud-access/space-weather-portal/api/v1/files"
    API_PARAMS = {"sat": "SWPC-Models", "inst": "ENLIL"}

    LABEL = "enlil"

    MODE: Mode

    NUM_DOWNLOAD_WORKERS = min(8, multiprocessing.cpu_count())

    PRE_ARCHIVE_CUTOFF = datetime(2023, 4, 4, tzinfo=timezone.utc)
    PRE_ARCHIVE_CUTOFF_MESSAGE = (
        'Note: All "Earth" associated variables are recorded at the location of Earth for model runs '
        "prior to April 04, 2023. Subsequent to that date, these correspond to the first Sun-Earth "
        "Lagrange point L1."
    )

    def _fallback_url(self) -> str:
        return self.API_URL

    @property
    def _product(self) -> str:
        """The NCEI product id this run mode is archived under."""
        return f"swpc_wsaenlil_{self.MODE}"

    def _warn_if_pre_archive_cutoff(self, target_date: datetime) -> None:
        """Log a critical message if `target_date` predates the archive's background-run cutoff.

        Parameters
        ----------
        target_date : datetime
            Date being requested.
        """
        if target_date < self.PRE_ARCHIVE_CUTOFF:
            logger.critical(self.PRE_ARCHIVE_CUTOFF_MESSAGE)

    def _file_path_for(self, target_date: datetime, run_time: str) -> Path:
        """Build the CSV path for one run of this mode.

        Parameters
        ----------
        target_date : datetime
            Date of the run.
        run_time : str
            Run time as `HHMM` (e.g. `"0000"`, `"0237"`).

        Returns
        -------
        Path
            Path to the CSV for this run.
        """
        return (
            self.data_dir
            / target_date.strftime("%Y/%m")
            / f"ENLIL_FORECAST_{self.MODE}_{target_date.strftime('%Y%m%d')}_{run_time}.csv"
        )

    def _find_file_entries(self, target_date: datetime) -> list[dict]:
        """Query the NCEI files API for this mode's ENLIL runs starting on `target_date`.

        Parameters
        ----------
        target_date : datetime
            Date to look up.

        Returns
        -------
        list[dict]
            File entries (each with `id`, `file_link`, `product`, ...) of this mode whose
            `time_coverage_start` is exactly `target_date` at 00:00 UTC.
        """
        day_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)

        params = {
            **self.API_PARAMS,
            "start_time": day_start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_time": day_end.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        request = requests.Request("GET", self.API_URL, params=params).prepare()
        self._record_url(request.url)  # ty: ignore[invalid-argument-type]

        response = requests.get(self.API_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        day_start_str = day_start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return [
            entry
            for entry in payload.get("data", [])
            if entry.get("time_coverage_start") == day_start_str and entry.get("product") == self._product
        ]

    def _missing_run_error(self, target_date: datetime) -> Optional[FileNotFoundError]:
        """Whether a date with no archived run of this mode is an error.

        Overridden by :class:`SWENLIL_BKG`: a `bkg` run is archived every day, so a
        missing one means an incomplete archive. The base implementation treats a
        missing run as normal, which is correct for :class:`SWENLIL_CME`: a `cme` run
        only exists for a date where a CME was modeled.

        Parameters
        ----------
        target_date : datetime
            Date that had no archived run.

        Returns
        -------
        FileNotFoundError | None
            The error to raise for this date, or `None` if a missing run is not an error.
        """
        return None

    def download_and_process(
        self, start_time: datetime, end_time: Optional[datetime] = None, reprocess_files: bool = False
    ) -> None:
        """
        Download and process this mode's WSA-ENLIL runs into CSVs.

        For each date in `[start_time.date(), end_time.date()]`, every run of this
        mode found for that date is downloaded and each saved as its own CSV. Whether a
        date with no run of this mode is an error is up to `_missing_run_error`.

        Parameters
        ----------
        start_time : datetime
            First date to download and process.
        end_time : datetime, optional
            Last date to download and process. If not provided, only `start_time`'s date is processed.
        reprocess_files : bool, optional
            Downloads and processes the files again, defaults to False.

        Raises
        ------
        FileNotFoundError
            If `_missing_run_error` reports an error for every requested date.
        requests.RequestException
            If the archive lookup fails for every requested date.
        """
        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time) if end_time is not None else start_time

        if start_time > end_time:
            msg = "start_time must be before end_time"
            logger.error(msg)
            raise ValueError(msg)

        self._warn_if_pre_archive_cutoff(start_time)

        dates = pd.date_range(
            datetime(start_time.year, start_time.month, start_time.day),
            datetime(end_time.year, end_time.month, end_time.day),
            freq="1D",
            tz="UTC",
        )

        self._resolved_urls = []

        jobs: list[tuple[dict, datetime]] = []
        errors: list[Exception] = []

        for date in dates:
            try:
                entries = self._find_file_entries(date)
            except requests.RequestException as e:
                logger.error(f"Failed to look up ENLIL runs for {date.date()}: {e}")
                errors.append(e)
                continue

            if not entries:
                error = self._missing_run_error(date)
                if error is not None:
                    logger.error(str(error))
                    errors.append(error)
                continue

            jobs.extend((entry, date) for entry in entries)

        self._run_download_jobs(jobs, reprocess_files)

        if errors and len(errors) == len(dates):
            raise errors[0]

    def _run_download_jobs(self, jobs: list[tuple[dict, datetime]], reprocess_files: bool) -> None:
        """Download and process a flat list of runs, with its own progress bar.

        Parameters
        ----------
        jobs : list[tuple[dict, datetime]]
            `(file entry, target_date)` pairs to process.
        reprocess_files : bool
            Downloads and processes runs again if `True`, even if their CSVs already exist.
        """
        if not jobs:
            return

        def process_job(job: tuple[dict, datetime]) -> None:
            entry, target_date = job
            self._download_and_process_single_run(entry, target_date, reprocess_files)

        desc = f"Downloading ENLIL {self.MODE} runs "
        if len(jobs) == 1:
            richpool.t_map(process_job, jobs, desc=desc)
        else:
            richpool.p_map(
                process_job, jobs, kind="process", num_cpus=min(self.NUM_DOWNLOAD_WORKERS, len(jobs)), desc=desc
            )

    def _download_and_process_single_run(self, entry: dict, target_date: datetime, reprocess_files: bool) -> None:
        """Download and process a single run described by an API file entry.

        Parameters
        ----------
        entry : dict
            File entry from the NCEI files API (`id`, `file_link`, ...).
        target_date : datetime
            Date the run belongs to.
        reprocess_files : bool
            Downloads and processes the file again if `True`, even if its CSV already exists.
        """
        run_time = self._run_time_from_id(entry["id"])
        file_path = self._file_path_for(target_date, run_time)

        if file_path.exists() and not reprocess_files:
            return

        temporary_dir = Path(tempfile.mkdtemp(prefix="temp_sw_enlil_wget_"))

        try:
            archive_path = self._download(temporary_dir, entry)

            logger.debug(f"Extracting {archive_path} ...")
            nc_path = self._extract_nc_file(archive_path, temporary_dir)

            processed_df = self._read_single_file(nc_path, target_date)
            processed_df["file_name"] = str(file_path)

            file_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
            processed_df.to_csv(tmp_path, index=True, header=True)
            tmp_path.replace(file_path)

            logger.debug(f"Saving processed file {file_path}")

        finally:
            rmtree(temporary_dir, ignore_errors=True)

    @staticmethod
    def _run_time_from_id(file_id: str) -> str:
        """Extract the `HHMM` run time from a file id like `swpc_wsaenlil_cme_20250818_0237.tar.gz`.

        Parameters
        ----------
        file_id : str
            The `id` field from an NCEI files API entry.

        Returns
        -------
        str
            The `HHMM` run time.
        """
        name = file_id
        while (suffix := Path(name).suffix) != "":
            name = name[: -len(suffix)]
        return name.split("_")[-1]

    def _download(self, temporary_dir: Path, entry: dict) -> Path:
        """Download a run's archive given its NCEI files API entry.

        Parameters
        ----------
        temporary_dir : Path
            Temporary directory to store the downloaded archive.
        entry : dict
            File entry from the NCEI files API (`id`, `file_link`).

        Returns
        -------
        Path
            Path to the downloaded `.tar.gz` archive.
        """
        file_url = entry["file_link"]
        archive_path = temporary_dir / entry["id"]

        response = requests.get(file_url, timeout=120, stream=True)
        response.raise_for_status()

        with open(archive_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

        return archive_path

    def _extract_nc_file(self, archive_path: Path, temporary_dir: Path) -> Path:
        """Extract the single `.nc` file from the archive, discarding everything else.

        Parameters
        ----------
        archive_path : Path
            Path to the downloaded `.tar.gz` archive.
        temporary_dir : Path
            Temporary directory to extract into.

        Returns
        -------
        Path
            Path to the extracted `.nc` file.

        Raises
        ------
        FileNotFoundError
            If the archive does not contain exactly one `.nc` file.
        """
        with tarfile.open(archive_path) as tar:
            nc_members = [m for m in tar.getmembers() if m.name.endswith(".nc")]
            if len(nc_members) != 1:
                raise FileNotFoundError(f"Expected exactly one .nc file in {archive_path}, found {len(nc_members)}.")
            tar.extract(nc_members[0], path=temporary_dir, filter="data")
            nc_path = temporary_dir / nc_members[0].name

        archive_path.unlink()

        return nc_path

    def _read_runs(self, start_time: datetime, end_time: Optional[datetime], download: bool) -> list[pd.DataFrame]:
        """Read every run of this mode keyed by `start_time`'s date.

        Only runs keyed by `start_time`'s date are read; no other dates are combined in.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read. Its date selects the runs.
        end_time : datetime, optional
            End time of the data to read. If not provided, defaults to the last
            timestamp in each run's file.
        download : bool
            Download and process data on the go.

        Returns
        -------
        list[pd.DataFrame]
            One data frame per run of this mode, sorted by run time and truncated to
            `[start_time, end_time]`. Empty if the date has no run of this mode.
        """
        start_time = enforce_utc_timezone(start_time)
        if end_time is not None:
            end_time = enforce_utc_timezone(end_time)

        if end_time is not None and start_time > end_time:
            msg = "start_time must be before end_time"
            logger.error(msg)
            raise ValueError(msg)

        self._warn_if_pre_archive_cutoff(start_time)

        if download and not self._csvs_for(start_time):
            try:
                self.download_and_process(start_time)
            except FileNotFoundError as e:
                logger.error(f"`download_and_process` failed because: {e}")

        return [self._read_and_truncate(fp, start_time, end_time) for fp in self._csvs_for(start_time)]

    def _csvs_for(self, target_date: datetime) -> list[Path]:
        """List the existing CSVs of this mode on `target_date`.

        Parameters
        ----------
        target_date : datetime
            Date to look up.

        Returns
        -------
        list[Path]
            Matching CSV paths, sorted by run time.
        """
        directory = self.data_dir / target_date.strftime("%Y/%m")
        pattern = f"ENLIL_FORECAST_{self.MODE}_{target_date.strftime('%Y%m%d')}_*.csv"
        return sorted(directory.glob(pattern)) if directory.exists() else []

    def _read_and_truncate(self, file_path: Path, start_time: datetime, end_time: Optional[datetime]) -> pd.DataFrame:
        """Read a processed CSV and truncate it to `[start_time, end_time]`.

        Parameters
        ----------
        file_path : Path
            Path to the CSV file.
        start_time : datetime
            Earliest timestamp to keep.
        end_time : datetime, optional
            Latest timestamp to keep. No upper bound if `None`.

        Returns
        -------
        pd.DataFrame
            Data from the CSV file, truncated and indexed by time (UTC).
        """
        data_out = self._read_single_csv(file_path)
        return data_out.truncate(before=start_time, after=end_time)

    def _read_single_csv(self, file_path: Path) -> pd.DataFrame:
        """Read a processed ENLIL CSV file into a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the `.csv` file.

        Returns
        -------
        pd.DataFrame
            Data from the CSV file, indexed by time (UTC).
        """
        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index, format="ISO8601", utc=True)
        df.index.name = "t"

        return df.sort_index()

    def _read_single_file(self, file_path: Path, start_time: datetime) -> pd.DataFrame:
        """Read the Earth-position timeline out of an ENLIL NetCDF file.

        Parameters
        ----------
        file_path : Path
            Path to the `.nc` file.
        start_time : datetime
            Earliest timestamp to keep.

        Returns
        -------
        pd.DataFrame
            Earth solar wind data from this file, indexed by time (UTC).
        """

        with xr.open_dataset(file_path, decode_times=False) as dataset:
            refdate = datetime.strptime(dataset.attrs["REFDATE_CAL"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            earth = {name: dataset[name].to_numpy().astype(float) for name in EARTH_VARIABLES}

        earth_time_sec = earth["Earth_TIME"]
        r, theta, phi = earth["Earth_X1"], earth["Earth_X2"], earth["Earth_X3"]
        b_r, b_theta, b_phi = earth["Earth_B1"], earth["Earth_B2"], earth["Earth_B3"]
        v_r, v_theta, v_phi = earth["Earth_V1"], earth["Earth_V2"], earth["Earth_V3"]
        density = earth["Earth_Density"]
        temperature = earth["Earth_Temperature"]

        timestamps = np.array([refdate + timedelta(seconds=sec) for sec in earth_time_sec])

        keep = timestamps >= start_time
        timestamps = timestamps[keep]
        r, theta, phi = r[keep], theta[keep], phi[keep]
        b_r, b_theta, b_phi = b_r[keep], b_theta[keep], b_phi[keep]
        v_r, v_theta, v_phi = v_r[keep], v_theta[keep], v_phi[keep]
        density = density[keep]
        temperature = temperature[keep]

        columns = ["bx_gsm", "by_gsm", "bz_gsm", "speed", "proton_density", "temperature", "bavg", "pdyn"]
        if len(timestamps) == 0:
            return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC", name="t"))

        bx_gsm, by_gsm, bz_gsm = self._transform_b_to_gsm(r, theta, phi, b_r, b_theta, b_phi, list(timestamps))
        tesla_to_nanotesla = 1e9
        bx_gsm, by_gsm, bz_gsm = bx_gsm * tesla_to_nanotesla, by_gsm * tesla_to_nanotesla, bz_gsm * tesla_to_nanotesla

        df = pd.DataFrame(
            index=pd.DatetimeIndex(timestamps, tz="UTC", name="t"),
            data={
                "bx_gsm": bx_gsm,
                "by_gsm": by_gsm,
                "bz_gsm": bz_gsm,
                "speed": np.sqrt(v_r**2 + v_theta**2 + v_phi**2) * METERS_PER_SECOND_TO_KM_PER_SECOND,
                "proton_density": density * KG_PER_M3_TO_PER_CM3,
                "temperature": temperature,
            },
        )
        df["bavg"] = np.sqrt(df["bx_gsm"] ** 2 + df["by_gsm"] ** 2 + df["bz_gsm"] ** 2)
        df["pdyn"] = 2e-6 * df["proton_density"] * df["speed"] ** 2

        return df[~df.index.duplicated(keep="first")].sort_index()

    def _transform_b_to_gsm(
        self,
        r: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        b_r: np.ndarray,
        b_theta: np.ndarray,
        b_phi: np.ndarray,
        timestamps: list[datetime],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert the Earth magnetic field vector from HEEQ spherical to GSM cartesian, in Tesla.

        `HeliographicStonyhurst` and `GeocentricSolarMagnetospheric` are position
        frames with different origins (Sun center vs. Earth center), so a field
        vector can't be transformed directly: transforming it as a
        `CartesianRepresentation` would also apply the origin's translation.
        Instead we transform two nearby *positions* (Earth's HEEQ position, and
        that position offset by the field vector) and take their difference in
        the target frame, which cancels the translation and keeps only the
        rotation between the two frames.

        Parameters
        ----------
        r, theta, phi : np.ndarray
            HEEQ spherical position of Earth in the model grid (radius in meters,
            co-latitude and longitude in radians).
        b_r, b_theta, b_phi : np.ndarray
            HEEQ spherical magnetic field components at Earth's position, in Tesla.
        timestamps : list[datetime]
            Timezone-aware UTC timestamp for each sample.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            `bx_gsm`, `by_gsm`, `bz_gsm` in Tesla.
        """
        bx_heeq, by_heeq, bz_heeq = _spherical_vector_to_cartesian(theta, phi, b_r, b_theta, b_phi)
        x_heeq, y_heeq, z_heeq = _spherical_vector_to_cartesian(theta, phi, r, np.zeros_like(r), np.zeros_like(r))

        obstime = Time(timestamps)
        frame_from = HeliographicStonyhurst(obstime=obstime)
        frame_to = GeocentricSolarMagnetospheric(obstime=obstime)

        earth_position = CartesianRepresentation(x_heeq * u.m, y_heeq * u.m, z_heeq * u.m)
        offset_position = (
            earth_position + CartesianRepresentation(bx_heeq, by_heeq, bz_heeq, unit=u.T) * _VECTOR_TRANSFORM_SCALE
        )

        earth_position_gsm = frame_from.realize_frame(earth_position).transform_to(frame_to).cartesian
        offset_position_gsm = frame_from.realize_frame(offset_position).transform_to(frame_to).cartesian

        b_gsm = (offset_position_gsm - earth_position_gsm) / _VECTOR_TRANSFORM_SCALE

        return (
            b_gsm.x.to_value(u.T),
            b_gsm.y.to_value(u.T),
            b_gsm.z.to_value(u.T),
        )


class SWENLIL_BKG(SWENLIL):
    """
    A class for handling NOAA SWPC WSA-ENLIL ambient background (`bkg`) runs.

    Exactly one background run is archived per day, at 00:00 UTC, so a read
    returns a single data frame and a date without a run is an error when
    downloading.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for the ENLIL data. If not provided, it will be read from the environment variable.

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.
    """

    MODE: Mode = "bkg"

    LABEL = "enlil_bkg"

    def _missing_run_error(self, target_date: datetime) -> FileNotFoundError:
        """A missing `bkg` run means an incomplete archive, unlike a missing `cme` run."""
        return FileNotFoundError(f"No ENLIL {self.MODE} run found for {target_date.date()}.")

    def read(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        download: bool = False,
    ) -> pd.DataFrame:
        """
        Read the processed ENLIL Earth-position background run of `start_time`'s date.

        Only the run keyed by `start_time`'s date is read; no other dates are combined in.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read. Its date selects the run.
        end_time : datetime, optional
            End time of the data to read. If not provided, defaults to the last
            timestamp in the run's file.
        download : bool, optional
            Download and process data on the go, defaults to False.

        Returns
        -------
        pd.DataFrame
            A data frame with columns `bx_gsm`, `by_gsm`, `bz_gsm`, `bavg`, `speed`,
            `proton_density`, `temperature`, `pdyn`, `file_name`, indexed by time (UTC).
            Empty if no background run exists for that date.
        """
        start_time = enforce_utc_timezone(start_time)
        runs = self._read_runs(start_time, end_time, download)

        if not runs:
            warnings.warn(f"No {self.MODE} file found for {start_time.date()}")
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        return runs[0]


class SWENLIL_CME(SWENLIL):
    """
    A class for handling NOAA SWPC WSA-ENLIL CME simulation (`cme`) runs.

    CME runs are archived only when a CME was modeled, so zero, one, or several
    runs can exist for a given day, each at its own run time. A read therefore
    returns a list of data frames, and a date without a run is not an error.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for the ENLIL data. If not provided, it will be read from the environment variable.

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.
    """

    MODE: Mode = "cme"

    LABEL = "enlil_cme"

    def read(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        download: bool = False,
    ) -> list[pd.DataFrame]:
        """
        Read every processed ENLIL Earth-position CME run of `start_time`'s date.

        Only runs keyed by `start_time`'s date are read; no other dates are combined in.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read. Its date selects the runs.
        end_time : datetime, optional
            End time of the data to read. If not provided, defaults to the last
            timestamp in each run's file.
        download : bool, optional
            Download and process data on the go, defaults to False.

        Returns
        -------
        list[pd.DataFrame]
            One data frame per CME run, sorted by run time, each with columns `bx_gsm`,
            `by_gsm`, `bz_gsm`, `bavg`, `speed`, `proton_density`, `temperature`, `pdyn`,
            `file_name`, indexed by time (UTC). Empty if that date has no CME run.
        """
        return self._read_runs(start_time, end_time, download)
