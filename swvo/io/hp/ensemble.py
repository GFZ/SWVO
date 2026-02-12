# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, TypeVar

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Number = TypeVar("Number", int, float)


class HpEnsemble(ABC):
    """This is a base class for Hp ensemble data.

    Parameters
    ----------
    index : str
        Hp index Possible options are: hp30, hp60.
    data_dir : Path | None
        Data directory for the Hp data. If not provided, it will be read from the environment variable

    Methods
    -------
    read

    Raises
    ------
    ValueError
        Returns `ValueError` if necessary environment variable is not set.
    FileNotFoundError
        Returns `FileNotFoundError` if the data directory does not exist.
    """

    ENV_VAR_NAME = "PLACEHOLDER; SEE DERIVED CLASSES BELOW"
    LABEL = "ensemble"

    def __init__(self, index: str, data_dir: Optional[Path] = None) -> None:
        self.index = index
        if self.index not in ("hp30", "hp60"):
            msg = "Encountered invalid index: {self.index}. Possible options are: hp30, hp60!"
            raise ValueError(msg)

        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                msg = f"Necessary environment variable {self.ENV_VAR_NAME} not set!"
                raise ValueError(msg)

            data_dir = os.environ.get(self.ENV_VAR_NAME)  # ty: ignore[invalid-assignment]

        self.data_dir: Path = Path(data_dir)

        logger.info(f"{self.index.upper()} Ensemble data directory: {self.data_dir}")

        if not self.data_dir.exists():
            msg = f"Data directory {self.data_dir} does not exist! Impossible to retrive data!"
            raise FileNotFoundError(msg)

        self.index_number: int = index[2:]

    def read(self, start_time: datetime, end_time: datetime) -> list[pd.DataFrame]:
        """
        Read Hp ensemble data for the requested period.

        Parameters
        ----------
        start_time : datetime
            Start time of the data to read. Must be timezone-aware.
        end_time : datetime
            End time of the data to read. Must be timezone-aware.

        Returns
        -------
        list[:class:`pandas.DataFrame`]
            List of ensemble data frames containing data for the requested period.

        Raises
        ------
        FileNotFoundError
            Returns `FileNotFoundError` if no ensemble file is found for the requested date.
        """
        if start_time is not None and not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time is not None and not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        if start_time is None:
            start_time = datetime.now(timezone.utc)

        if end_time is None:
            end_time = start_time.replace(tzinfo=timezone.utc) + timedelta(days=3)

        start_date = start_time.replace(microsecond=0, minute=0, second=0)
        str_date = start_date.strftime("%Y%m%dT%H0000")

        file_list = self._ensemble_file_list(str_date)
        data = []

        if len(file_list) == 0:
            msg = f"No {self.index} ensemble file found for requested date {start_date}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        for file in file_list:
            hp_df = pd.read_csv(file, names=["t", self.index])

            hp_df["t"] = pd.to_datetime(hp_df["t"], utc=True)
            hp_df.index = hp_df["t"]
            hp_df = hp_df.drop(labels=["t"], axis=1)

            hp_df = hp_df.truncate(
                before=start_time - timedelta(minutes=int(self.index_number) - 0.01),
                after=end_time + timedelta(minutes=int(self.index_number) + 0.01),
            )

            data.append(hp_df)

        return data

    def _ensemble_file_list(self, str_date: str) -> list[Path]:
        """Check for the existence of ensemble files for a given date.

        Parameters
        ----------
        str_date : str
            Date string in the format YYYYMMDDTHH0000.

        Returns
        -------
        list[Path]
            A list of file paths for the ensemble files.

        Warnings
        --------
        DeprecationWarning
            Warns if deprecated file naming convention is used.
        """

        file_list_old_name = sorted(
            self.data_dir.glob(f"FORECAST_{self.index.title()}_{str_date}_ensemble_*.csv"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )

        file_list_new_name = sorted(
            self.data_dir.glob(f"FORECAST_{self.index.upper()}_SWIFT_DRIVEN_swift_{str_date}_ensemble_*.csv"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )

        file_list: list[Path]

        if len(file_list_new_name) == 0 and len(file_list_old_name) == 0:
            file_list = []
        elif len(file_list_new_name) > 0:
            file_list = file_list_new_name
        elif len(file_list_old_name) > 0:
            warnings.warn(
                "The use of FORECAST_HP*_SWIFT_DRIVEN_swift_* files is deprecated. However since we still have these files from the PAGER project with this prefix, this will be supported",
                DeprecationWarning,
            )
            file_list = file_list_old_name

        return file_list

    @abstractmethod
    def read_with_horizon(self, start_time: datetime, end_time: datetime, horizon: Number) -> list[pd.DataFrame]:
        if start_time is not None and not start_time.tzinfo:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time is not None and not end_time.tzinfo:
            end_time = end_time.replace(tzinfo=timezone.utc)

        if not (0 <= horizon <= 72):  # ty: ignore[unsupported-operator]
            raise ValueError("Horizon must be between 0 and 72 hours")

        if self.index == "hp30":
            freq = "0.5h"
            if horizon % 0.5 != 0:
                raise ValueError("Horizon for hp30 must be in 0.5 hour increments")
        elif self.index == "hp60":
            freq = "1h"
            if horizon % 1 != 0:  # ty: ignore[unsupported-operator]
                raise ValueError("Horizon for hp60 must be in 1 hour increments")

        align_start_to_hp_hr = start_time.replace(hour=start_time.hour, minute=0, second=0, microsecond=0)
        align_end_to_hp_hr = end_time.replace(hour=end_time.hour, minute=0, second=0, microsecond=0)

        full_time_range = pd.date_range(align_start_to_hp_hr, align_end_to_hp_hr, freq=freq, tz=timezone.utc)

        file_offsets, time_indices = self._get_file_offsets_and_time_indices(full_time_range, horizon)

        max_ensembles = 30  # Maximum number of ensemble files to check
        ensemble_dfs = [pd.DataFrame(index=full_time_range) for _ in range(max_ensembles)]

        for file_time, file_offset, iloc in zip(full_time_range, file_offsets, time_indices):
            str_date = (file_time + timedelta(hours=file_offset)).strftime("%Y%m%dT%H0000")
            file_list = self._ensemble_file_list(str_date)
            for ensemble_idx in range(max_ensembles):
                df = ensemble_dfs[ensemble_idx]
                if ensemble_idx < len(file_list):
                    data = pd.read_csv(
                        file_list[ensemble_idx],
                        names=["Forecast Time", self.index],
                        parse_dates=["Forecast Time"],
                    ).iloc[iloc]
                    data["source"] = str_date
                    data["Forecast Time"] = data["Forecast Time"].tz_localize("UTC")
                    df.loc[file_time, "Forecast Time"] = data["Forecast Time"]
                    df.loc[file_time, self.index] = data[self.index]
                    df.loc[file_time, "source"] = file_list[ensemble_idx].stem
                else:
                    df.loc[file_time, self.index] = np.nan

        for df in ensemble_dfs:
            df["horizon"] = horizon
            df.index.name = "Time"
        ensemble_dfs = [df for df in ensemble_dfs if not df[self.index].isna().all()]

        if len(ensemble_dfs) == 0:
            msg = f"No ensemble data found for the requested period {start_time} to {end_time} and horizon {horizon} hours. Check the data directory {self.data_dir} for available data."
            logger.error(msg)
            raise FileNotFoundError(msg)

        return ensemble_dfs

    def _get_file_offsets_and_time_indices(
        self, file_time_range: Iterable, forecast_horizon: float
    ) -> tuple[list[int], list[int]]:
        """
        Compute file offsets and time indices for a given forecast horizon.

        Parameters
        ----------
        file_time_range : iterable
            Available file time steps.
        forecast_horizon : float
            Forecast horizon in hours (can be fractional, e.g., 0.5, 1.5, 72.0).

        Returns
        -------
        file_offsets : list[int]
            Offsets from the given time steps indicating which files to read.
        time_indices : list[int]
            Time indices to use for each file.
        """
        file_offsets = []
        time_indices = []

        for current_time in file_time_range:
            current_hour = current_time.hour
            current_minute = current_time.minute
            current_fractional_hour = current_hour + current_minute / 60.0

            # Target forecast time (fractional hours from same midnight)
            target_fractional_hour = current_fractional_hour + forecast_horizon

            # Find the file base hour (multiple of 3) that contains this target
            # Files are grouped by 3s: [0,1,2] use base 0, [3,4,5] use base 3, etc.
            # Each file covers 72 hours from its base hour

            # Determine which 3-hour group the target falls into
            target_hour_int = int(target_fractional_hour)
            target_base_hour = (target_hour_int // 3) * 3

            if target_fractional_hour <= target_base_hour + 72:
                file_base_hour = target_base_hour
            else:
                # use next group
                file_base_hour = target_base_hour + 3

            # Determine which file in the group (base, base+1, base+2) to use
            # Use the file that was created at or before current_time
            if current_hour >= file_base_hour:
                # Use the latest file in the group that's <= current_hour
                file_hour = min(current_hour, file_base_hour + 2)
            else:
                file_hour = file_base_hour

            file_offset = file_hour - current_hour
            file_offsets.append(file_offset)
            hours_from_file_start = target_fractional_hour - file_base_hour
            resolution = 0.5 if self.index == "hp30" else 1.0
            time_index = int(hours_from_file_start / resolution)
            time_indices.append(time_index)

        return file_offsets, time_indices


class Hp30Ensemble(HpEnsemble):
    """A class to handle Hp30 ensemble data.

    Parameters
    ----------
    data_dir : str | Path, optional
        Data directory for the Hp30 ensemble data. If not provided, it will be read from the environment variable
    """

    ENV_VAR_NAME = "HP30_ENSEMBLE_FORECAST_DIR"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        super().__init__("hp30", data_dir)

    def read_with_horizon(self, start_time: datetime, end_time: datetime, horizon: float) -> list[pd.DataFrame]:
        """Read Ensemble Hp30 forecast data for a given time range and forecast horizon.

        Parameters
        ----------
        start_time : datetime
            Start time of the period for which to read the data.
        end_time : datetime
            End time of the period for which to read the data.
        horizon : float
            Forecast horizon (in hours).

        Returns
        -------
        list[:class:`pandas.DataFrame`]
            A list of data frames containing ensemble data for the requested period.

        Raises
        ------
        ValueError
            Raises `ValueError` if the horizon is not between 0 and 72 hours.
        ValueError
            Raises `ValueError` if the horizon is not in 0.5 hour increments.
        """
        return super().read_with_horizon(start_time, end_time, horizon)


class Hp60Ensemble(HpEnsemble):
    """A class to handle Hp60 ensemble data.

    Parameters
    ----------
    data_dir : str | Path, optional
        Data directory for the Hp60 ensemble data. If not provided, it will be read from the environment variable
    """

    ENV_VAR_NAME = "HP60_ENSEMBLE_FORECAST_DIR"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        super().__init__("hp60", data_dir)

    def read_with_horizon(self, start_time: datetime, end_time: datetime, horizon: int) -> list[pd.DataFrame]:
        """Read Ensemble Hp60 forecast data for a given time range and forecast horizon.

        Parameters
        ----------
        start_time : datetime
            Start time of the period for which to read the data.
        end_time : datetime
            End time of the period for which to read the data.
        horizon : int
            Forecast horizon (in hours).

        Returns
        -------
        list[:class:`pandas.DataFrame`]
            A list of data frames containing ensemble data for the requested period.

        Raises
        ------
        ValueError
            Raises `ValueError` if the horizon is not between 0 and 72 hours.
        ValueError
            Raises `ValueError` if the horizon is not in 1 hour increments.
        """
        return super().read_with_horizon(start_time, end_time, horizon)
