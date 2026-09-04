# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for handling MIDL Solar Wind data.
"""

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import midl as midl_client
import numpy as np
import pandas as pd
import xarray as xr

from swvo.io.base import BaseIO
from swvo.io.utils import enforce_utc_timezone, sw_mag_propagation

logger = logging.getLogger(__name__)

logging.captureWarnings(True)


class SWMIDL(BaseIO):
    """This is a class for the MIDL (Merged Interplanetary Data from L1) Solar Wind data.

    MIDL is a quality screened, one minute solar wind record merged from the L1 monitors
    (ACE, DSCOVR, WIND and SOHO), maintained by the Center for Space Environment Modeling at the
    University of Michigan. Data are available from 1998 onwards and are refreshed monthly.

    This class is a thin wrapper around the official `csem-midl <https://pypi.org/project/csem-midl/>`_
    client: the client performs the download, and this class adapts its output to the SWVO solar
    wind schema and caches it as daily CSV files, like the other solar wind products.

    Parameters
    ----------
    data_dir : Path | None
        Data directory for the MIDL Solar Wind data. If not provided, it will be read from the environment variable
    prefer_env_var : bool, optional
        If True, the environment variable takes precedence over the passed `data_dir` argument.
    target : float | int | str, optional
        Location at which MIDL returns the solar wind: ``"L1"`` (default) for unpropagated L1
        observations, or a distance in Earth radii along the Sun-Earth line for data MIDL has
        already propagated.
    method : str, optional
        MIDL propagation method, either ``"ballistic"`` (default) or ``"mhd"``. Only relevant if
        `target` is not ``"L1"``. MHD targets must be integer Earth radii in ``[-70, 70]``.

    Methods
    -------
    download_and_process
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set, or if `target` or
        `method` are invalid.
    """

    ENV_VAR_NAME = "SW_MIDL_STREAM_DIR"

    URL = "https://csem.engin.umich.edu/MIDL/data"

    # MIDL is a merged product with no instrument split, so it keeps a single field list rather
    # than the MAG/SWEPAM pair the single spacecraft readers inherit from ACE's instruments.
    # `pdyn` is not listed here: it is derived once the fields are on a complete time grid.
    FIELDS = ["bx_gsm", "by_gsm", "bz_gsm", "bavg", "speed", "proton_density", "temperature"]

    # MIDL variable name -> SWVO column name. The remaining SWVO columns (bavg, speed, pdyn) are
    # derived, since MIDL publishes the field and velocity vectors but no magnitudes.
    VARIABLE_FIELDS = {
        "Bx": "bx_gsm",
        "By": "by_gsm",
        "Bz": "bz_gsm",
        "rho": "proton_density",
        "T": "temperature",
    }

    LABEL = "midl"
    START_YEAR = 1998

    METHODS = ("ballistic", "mhd")

    def __init__(
        self,
        data_dir: Path | None = None,
        prefer_env_var: bool = False,
        target: float | int | str = "L1",
        method: str = "ballistic",
    ) -> None:
        super().__init__(data_dir, prefer_env_var)

        if not isinstance(method, str) or method.lower() not in self.METHODS:
            msg = f"method must be one of {self.METHODS}, got {method!r}"
            logger.error(msg)
            raise ValueError(msg)
        self.method = method.lower()

        if isinstance(target, str):
            if target.lower() != "l1":
                msg = f"target must be a number or 'L1', got {target!r}"
                logger.error(msg)
                raise ValueError(msg)
            self.target: float | str = "L1"
        elif isinstance(target, (int, float)) and not isinstance(target, bool):
            self.target = float(target)
        else:
            msg = f"target must be a number or 'L1', got {type(target).__name__}"
            logger.error(msg)
            raise ValueError(msg)

        if self.target == "L1" and self.method != "ballistic":
            msg = "target='L1' is only valid with method='ballistic'"
            logger.error(msg)
            raise ValueError(msg)

        self._target_token = self._make_target_token()

    def _make_target_token(self) -> str:
        """Build the canonical target token used in cached file names.

        Mirrors MIDL's own naming (``L1``, ``32Re``, ``mhd_030Re``) so that data for different
        targets can live side by side in one data directory.

        Returns
        -------
        str
            Canonical, file name safe target token.
        """
        if self.target == "L1":
            return "L1"

        target_re = float(self.target)
        if self.method == "mhd":
            if not target_re.is_integer() or not (-70 <= target_re <= 70):
                msg = f"MHD target must be an integer number of Earth radii in [-70, 70], got {self.target!r}"
                logger.error(msg)
                raise ValueError(msg)
            return f"mhd_{int(target_re):03d}Re"

        if target_re.is_integer():
            return f"{int(target_re)}Re"
        return f"{target_re:g}Re".replace(".", "p")

    def _fallback_url(self) -> str:
        return self.URL

    def download_and_process(self, start_time: datetime, end_time: datetime) -> None:
        """
        Download and process MIDL data, splitting data across midnight into appropriate day files.

        The download itself is delegated to :func:`midl.load`, which fetches the monthly source
        files and keeps its own cache. Velocities are requested with ``orbital_motion=False`` so
        that they follow the same convention (CDAWeb/OMNI) as the other SWVO solar wind products.

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
            If `start_time` is before the first year of MIDL data.

        Returns
        -------
        None
        """
        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        assert start_time < end_time, "Start time must be before end time!"

        if start_time.year < self.START_YEAR:
            msg = f"MIDL data is only available from {self.START_YEAR} onwards."
            logger.error(msg)
            raise ValueError(msg)

        self._resolved_urls = []
        for url in self._source_urls(start_time, end_time):
            self._record_url(url)

        logger.debug(f"Downloading MIDL data for {start_time} - {end_time} ...")
        # MIDL indexes its files with naive timestamps and slices them with whatever it is
        # handed, so a timezone-aware bound raises a comparison error. SWVO is UTC throughout,
        # so the bounds are simply stripped of their (already UTC) timezone here.
        dataset = midl_client.load(
            start_time.replace(tzinfo=None),
            end_time.replace(tzinfo=None),
            self.target,
            method=self.method,
            coords="GSM",
            orbital_motion=False,
        )

        logger.debug("Processing data ...")
        processed_df = self._process_dataset(dataset)

        if processed_df.empty:
            logger.warning(f"No MIDL data returned for {start_time} - {end_time}")
            return

        unique_dates = np.unique(processed_df.index.date)  # ty: ignore[unresolved-attribute]

        for date in unique_dates:
            file_path = self.data_dir / date.strftime("%Y/%m") / self._file_name(date)
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
                if tmp_path.exists():
                    tmp_path.unlink()
                continue

    def _source_urls(self, start_time: datetime, end_time: datetime) -> list[str]:
        """Build the list of monthly source file URLs spanned by a time range.

        These are the files the MIDL client fetches under the hood; they are recorded so that
        :attr:`url` reports the concrete sources a download used.

        Parameters
        ----------
        start_time : datetime
            Start time of the data.
        end_time : datetime
            End time of the data.

        Returns
        -------
        list[str]
            One URL per month covered by the time range.
        """
        months = pd.date_range(
            datetime(start_time.year, start_time.month, 1),
            datetime(end_time.year, end_time.month, 1),
            freq="MS",
        )

        urls = []
        for month in months:
            sub_dir = "mhd/" if self._target_token.startswith("mhd_") else ""
            urls.append(
                f"{self.URL}/{month:%Y}/{month:%m}/{sub_dir}{month:%Y%m}_{self._target_token}.csv",
            )

        return urls

    def _file_name(self, date) -> str:
        """Build the cached file name for a single day.

        Parameters
        ----------
        date : datetime.date
            Date of the file.

        Returns
        -------
        str
            File name of the processed daily file.
        """
        return f"MIDL_SW_{self._target_token}_{date.strftime('%Y%m%d')}.csv"

    def _process_dataset(self, dataset: xr.Dataset) -> pd.DataFrame:
        """Convert a MIDL dataset to the SWVO Solar Wind schema.

        MIDL publishes the magnetic field and velocity vectors in GSM, plus density and
        temperature. The field magnitude, the bulk speed and the dynamic pressure are derived
        here. MIDL's provenance columns (``X``, ``*_source``, ``*_interp``) are dropped, as SWVO's
        solar wind schema has no place for them.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset as returned by :func:`midl.load`.

        Returns
        -------
        pd.DataFrame
            MIDL data on a complete one minute grid, with the SWVO Solar Wind columns.
        """
        data = dataset.to_dataframe()

        if data.empty:
            return pd.DataFrame(columns=[*self.FIELDS, "pdyn"])

        data.index = pd.to_datetime(data.index, utc=True)
        data.index.name = "t"

        for source, target in self.VARIABLE_FIELDS.items():
            data[target] = pd.to_numeric(data[source], errors="coerce") if source in data.columns else np.nan

        data["bavg"] = np.sqrt(data["bx_gsm"] ** 2 + data["by_gsm"] ** 2 + data["bz_gsm"] ** 2)

        velocity = [
            pd.to_numeric(data[c], errors="coerce") if c in data.columns else np.nan for c in ("Ux", "Uy", "Uz")
        ]
        data["speed"] = np.sqrt(sum(component**2 for component in velocity))

        data = data[self.FIELDS]

        complete_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq="1min", tz="UTC")
        data = data.reindex(complete_range)
        data.index.name = "t"

        data["pdyn"] = 2e-6 * data["proton_density"].values * data["speed"].values ** 2

        return data

    def read(
        self,
        start_time: datetime,
        end_time: datetime,
        download: bool = False,
        propagation: bool = False,
    ) -> pd.DataFrame:
        """
        Read MIDL data for the specified time range.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read. Must be timezone-aware.
        end_time : datetime
            End time of the data to read. Must be timezone-aware.
        download : bool, optional
            Download data on the go, defaults to False.
        propagation : bool, optional
            Propagate the data from L1 to near-Earth, defaults to False. Ignored if the instance
            was constructed with a `target` other than ``"L1"``, since MIDL has then already
            propagated the data.

        Returns
        -------
        :class:`pandas.DataFrame`
            DataFrame containing MIDL Solar Wind data for the requested period.

        Raises
        ------
        AssertionError
            Raises `AssertionError` if the end time is before the start time.
        """
        start_time = enforce_utc_timezone(start_time)
        end_time = enforce_utc_timezone(end_time)

        if propagation and self.target != "L1":
            logger.warning(
                f"Ignoring `propagation` because MIDL data is already propagated to {self._target_token}",
            )
            propagation = False

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
        data_out = pd.DataFrame(index=t, data={field: nan_data for field in [*self.FIELDS, "pdyn"]})

        if download and any(not file_path.exists() for file_path in file_paths):
            try:
                self.download_and_process(start_time, end_time)
                file_paths, _ = self._get_processed_file_list(start_time, end_time)
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
            file_path = self.data_dir / current_time.strftime("%Y/%m") / self._file_name(current_time)
            file_paths.append(file_path)

            interval_start = current_time
            interval_end = datetime(current_time.year, current_time.month, current_time.day, 23, 59, 59)

            time_intervals.append((interval_start, interval_end))
            current_time += timedelta(days=1)

        return file_paths, time_intervals

    def _read_single_file(self, file_path) -> pd.DataFrame:
        """Read MIDL file to a DataFrame.

        Parameters
        ----------
        file_path : Path
            Path to the file.

        Returns
        -------
        pd.DataFrame
            Data from MIDL file.
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
        return "propagated from previous MIDL file" if file_date != index_date else row["file_name"]
