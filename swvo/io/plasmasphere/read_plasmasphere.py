# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Stefano Bianco
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlasmasphereDensityCube:
    """A structured container for plasmaspheric electron density data.

    Attributes
    -------
    time : np.ndarray[datetime]
        Array of Python datetime values.
    l : np.ndarray
        Array of L-values.
    mlt : np.ndarray
        Array of MLT-values.
    l_grid : np.ndarray
        Grid of L-values. L x MLT shape.
    mlt_grid : np.ndarray
        Grid of MLT-values. L x MLT shape.
    density_grid : np.ndarray or list[np.ndarray]
        Grid of electron density values. time x L x MLT shape if `density_column` is a single column, otherwise a list of such arrays (one per density column).
    density_column : str or list[str]
        Name(s) of the column(s) containing electron density data.
    """

    time: np.ndarray[datetime]  # ty: ignore[invalid-type-arguments]
    l: np.ndarray  # noqa: E741
    mlt: np.ndarray
    l_grid: np.ndarray
    mlt_grid: np.ndarray
    density_grid: list[np.ndarray]
    density_column: str | list[str]

    def __str__(self) -> str:
        """Readable summary for logging and printing."""
        num_times = len(self.time)

        l_range = f"[{self.l.min():.2f}, {self.l.max():.2f}]"
        mlt_range = f"[{self.mlt.min():.2f}, {self.mlt.max():.2f}]"

        summary = [
            "--- Plasmasphere Density Cube ---",
            f"Temporal Span : {num_times} steps ({self.time[0]} to {self.time[-1]})",
            f"Spatial L-Bins: {len(self.l)} {l_range}",
            f"Spatial MLT-Bins: {len(self.mlt)} {mlt_range}",
            f"Density Grid Geometry per Time Step : {self.density_grid[0].shape if isinstance(self.density_grid, list) else self.density_grid.shape} (Time x L x MLT)",
            f"Data Columns  : {self.density_column}",
            "----------------------------------",
        ]
        return "\n".join(summary)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlasmasphereDensityCube):
            return NotImplemented

        if not np.array_equal(self.time, other.time):
            return False
        if not np.array_equal(self.l, other.l):
            return False
        if not np.array_equal(self.mlt, other.mlt):
            return False
        if not np.array_equal(self.l_grid, other.l_grid):
            return False
        if not np.array_equal(self.mlt_grid, other.mlt_grid):
            return False

        if isinstance(self.density_grid, np.ndarray) and isinstance(other.density_grid, np.ndarray):
            if not np.array_equal(self.density_grid, other.density_grid):
                return False
        elif isinstance(self.density_grid, list) and isinstance(other.density_grid, list):
            if len(self.density_grid) != len(other.density_grid):
                return False
            for grid_self, grid_other in zip(self.density_grid, other.density_grid):
                if not np.array_equal(grid_self, grid_other):
                    return False
        else:
            return False

        if self.density_column != other.density_column:
            return False

        return True

    def get_density_at_time(self, time: datetime) -> np.ndarray | list[np.ndarray]:
        """Extract density grid for a specific time.

        Parameters
        ----------
        time : datetime
            The specific time for which to extract the density grid.

        Returns
        -------
        np.ndarray or list[np.ndarray]
            The density grid corresponding to the specified time.

        Raises
        ------
        IndexError
            If the specified time is not found in the density cube.
        """
        if time not in self.time:
            logger.error(f"Requested time {time} not found in density cube.")
            raise IndexError(f"Requested time {time} not found in density cube.")

        time_index = np.where(self.time == time)[0][0]

        if isinstance(self.density_grid, list):
            return [grid[time_index] for grid in self.density_grid]
        else:
            return [self.density_grid[time_index]]


class PlasmaspherePredictionReader:
    """Reads one of the available PAGER plasmasphere density prediction.

    Parameters
    ----------
    folder : str
        The folder where the plasmasphere prediction files are stored.

    Raises
    ------
    FileNotFoundError
        If the data folder does not exist.
    RuntimeError
        If the source of data requested is not among the available ones.
    """

    ENV_VAR_NAME = "PLASMASPHERE_OUTPUT_DIR"
    LABEL = "plasmsphere"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)  # ty: ignore[invalid-assignment]

        self.data_dir: Path = Path(data_dir)  # ty:ignore[invalid-argument-type]

        logger.info(f"Plasmasphere data directory: {self.data_dir}")

        if not self.data_dir.exists():
            msg = f"Plasmasphere directory {self.data_dir} does not exist! Impossible to retrive data!"
            logger.error(msg)
            raise FileNotFoundError(msg)

    def _parse_none_date(self, date: datetime | None) -> datetime:
        if date is None:
            return datetime.now(timezone.utc).replace(microsecond=0, minute=0, second=0)
        return date.replace(minute=0, second=0, microsecond=0)

    def read(self, requested_date: datetime | None = None) -> pd.DataFrame | None:
        """
        Reads one of the available PAGER plasmasphere density prediction.

        Parameters
        ----------
        requested_date : datetime.datetime or None
            Date of plasma density prediction thar we want to read up to hour precision.

        Raises
        ------
        RuntimeError
            if the sources of data requested is not among the available ones.

        Returns
        -------
        pd.DataFrame or None
            pandas.DataFrame with L, MLT, density and date as columns
        """

        requested_date = self._parse_none_date(requested_date)

        file_name = f"plasmasphere_density_{requested_date.strftime('%Y%m%dT%H00')}.csv"

        file_path = os.path.join(self.data_dir, file_name)
        logger.info(f"Looking for file {file_path} for date {requested_date}")
        if not os.path.isfile(file_path):
            msg = f"No suitable files ({file_path}) found for the requested date {requested_date}. Returning None."
            logger.error(msg)
            return None

        logger.info(f"Reading plasmasphere density data from {file_path}")

        data = pd.read_csv(file_path, parse_dates=["date"])
        data["t"] = data["date"]
        data.drop(labels=["date"], axis=1, inplace=True)
        return data

    def _validate_data(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            msg = f"data must be an instance of a pandas dataframe, instead it is of type {type(data)}"
            logger.error(msg)
            raise TypeError(msg)

        required_columns = ["L", "MLT", "t"]
        for column in required_columns:
            if column not in data.columns:
                msg = f"column {column} is missing"
                logger.error(msg)
                raise ValueError(msg)

        if data.empty:
            msg = "data dataframe is empty"
            logger.error(msg)
            raise ValueError(msg)

        if not pd.api.types.is_datetime64_any_dtype(data["t"]):
            msg = "values of 't' column must be datetime objects"
            logger.error(msg)
            raise TypeError(msg)

    def _get_density_columns(self, data: pd.DataFrame) -> list[str]:
        density_columns = [column for column in data.columns if "predicted_densities" in column]
        if not density_columns:
            msg = "no columns matching 'predicted_densities' were found"
            logger.error(msg)
            raise ValueError(msg)
        return density_columns

    def _resolve_density_column(self, data: pd.DataFrame, density_column: str | None) -> str:
        density_columns = self._get_density_columns(data)
        if density_column is None:
            return density_columns[0]
        if density_column not in density_columns:
            msg = f"density_column '{density_column}' is not valid. Available columns: {density_columns}"
            logger.error(msg)
            raise ValueError(msg)
        return density_column

    def _legacy_reshape_2d(self, df_date: pd.DataFrame, density_column: str) -> tuple:
        l_values = df_date["L"].to_numpy()
        mlt_values = df_date["MLT"].to_numpy()
        density_values = df_date[density_column].to_numpy(dtype=float)

        l_axis = np.unique(l_values)
        mlt_axis = np.unique(mlt_values)

        expected_points = len(l_axis) * len(mlt_axis)
        if len(df_date) != expected_points:
            msg = "data for a single timestamp does not form a complete L-MLT grid. Expected n_L * n_MLT rows."
            logger.error(msg)
            raise ValueError(msg)

        l_grid = np.reshape(l_values, (len(l_axis), len(mlt_axis)), order="F")
        mlt_grid = np.reshape(mlt_values, (len(l_axis), len(mlt_axis)), order="F")
        density_2d = np.reshape(density_values, (len(l_axis), len(mlt_axis)), order="F")

        return l_axis, mlt_axis, l_grid, mlt_grid, density_2d

    def build_density_cube(
        self,
        requested_date: datetime | None = None,
        density_column: str | None = None,
    ) -> Optional[PlasmasphereDensityCube]:
        """
        Build density tensor with shape time x L x MLT.

        Parameters
        ----------
        requested_date : datetime.datetime or None
            Date of plasma density prediction that we want to read up to hour precision.

        Returns
        -------
        PlasmasphereDensityCube or None
            If `density_column` is provided, `density_grid` has shape
            (n_time, n_L, n_MLT). If `density_column` is None, `density_grid`
            is a list of arrays with that same shape (one per density column).

            If no data is available for the requested date, returns None.
        """
        requested_date = self._parse_none_date(requested_date)
        data = self.read(requested_date=requested_date)
        if data is None:
            return None
        self._validate_data(data)

        if density_column is None:
            resolved_density_columns = self._get_density_columns(data)
        else:
            resolved_density_columns = [self._resolve_density_column(data, density_column)]
        dates = np.sort(data["t"].unique())
        dates = pd.to_datetime(dates)
        dates_to_return = np.array([dt.to_pydatetime() for dt in dates.to_list()])
        density_slices_by_column = {column: [] for column in resolved_density_columns}

        l_axis_ref = None
        mlt_axis_ref = None
        l_grid_ref = None
        mlt_grid_ref = None

        for date in dates:
            df_date = data[pd.to_datetime(data["t"]) == date]
            for column in resolved_density_columns:
                l_axis, mlt_axis, l_grid, mlt_grid, density_2d = self._legacy_reshape_2d(df_date, column)

                if l_axis_ref is None:
                    l_axis_ref = l_axis
                    mlt_axis_ref = mlt_axis
                    l_grid_ref = l_grid
                    mlt_grid_ref = mlt_grid
                else:
                    assert mlt_axis_ref is not None
                    if not np.array_equal(l_axis_ref, l_axis) or not np.array_equal(mlt_axis_ref, mlt_axis):
                        msg = "Inconsistent L/MLT axes across timestamps."
                        logger.error(msg)
                        raise ValueError(msg)

                density_slices_by_column[column].append(density_2d)

        if l_axis_ref is None or mlt_axis_ref is None or l_grid_ref is None or mlt_grid_ref is None:
            msg = "Unable to build density cube axes from input data."
            logger.error(msg)
            raise RuntimeError(msg)

        if len(resolved_density_columns) == 1:
            resolved_density_column: str | list[str] = resolved_density_columns[0]
            density_grid = [
                np.stack(
                    density_slices_by_column[resolved_density_columns[0]],
                    axis=0,
                )
            ]
        else:
            resolved_density_column = resolved_density_columns
            density_grid = [np.stack(density_slices_by_column[column], axis=0) for column in resolved_density_columns]

        return PlasmasphereDensityCube(
            time=dates_to_return,
            l=l_axis_ref,
            mlt=mlt_axis_ref,
            l_grid=l_grid_ref,
            mlt_grid=mlt_grid_ref,
            density_grid=density_grid,
            density_column=resolved_density_column,
        )
