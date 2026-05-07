# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import fnmatch
import logging
import pickle
import re
import typing
import warnings
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import pandas as pd
from mat73 import loadmat
from numpy.typing import NDArray
from scipy.io import loadmat as sci_loadmat

from swvo.io.utils import enforce_utc_timezone

logger = logging.getLogger(__name__)


def join_var(var1: NDArray[np.generic], var2: NDArray[np.generic]) -> NDArray[np.generic]:
    """Join two variables along the first axis."""
    return np.concatenate((var1, var2), axis=0)


def get_file_path_any_format(folder_path: Path, file_stem: str, preferred_ext: str, nc_mode: bool) -> Path | None:
    """Get the file path for a given file stem and preferred extension."""

    if not nc_mode:
        folder_path = folder_path / "Processed_Mat_Files"

    pattern = re.compile(fnmatch.translate(file_stem + ".*"), re.IGNORECASE)
    try:
        all_files = [p for p in folder_path.iterdir() if pattern.match(p.name)]
    except FileNotFoundError:
        all_files = []

    if len(all_files) == 0:
        logger.warning(f"File not found: {folder_path / (file_stem + '.*')}")
        return None

    if len(all_files) >= 1:
        extensions_found = [file.suffix[1:] for file in all_files]
        print(f"{preferred_ext=}, {extensions_found=}")
        if len(all_files) > 1:
            if preferred_ext in extensions_found:
                logger.warning(
                    (
                        f"Several files found for {folder_path / (file_stem + '.*')} with extensions: {extensions_found}. "
                        f"Choosing: {preferred_ext}."
                    )
                )

                return folder_path / (file_stem + "." + preferred_ext)

            msg = (
                f"Several files found for {folder_path / (file_stem + '.*')} with extensions: {extensions_found}. "
                f"However, the preferred extension ({preferred_ext}) is not available!"
            )
            raise ValueError(msg)

        if len(all_files) == 1:
            if preferred_ext not in extensions_found:
                logger.warning(
                    (
                        f"Files found for {folder_path / (file_stem + '.*')} with extensions: {extensions_found[0]}. "
                        f"Preferred extension ({preferred_ext}) is not available! "
                        f"Choosing: {extensions_found[0]}."
                    )
                )
            return all_files[0]

        logger.warning(
            f"File not found: {folder_path / (file_stem + '.' + preferred_ext)}",
            stacklevel=2,
        )

    return None


def read_all_datasets_netcdf(file_path: str | Path) -> dict[str, Any]:
    """Reads all datasets (variables) from a NetCDF file, including those in groups.

    This function recursively traverses all groups and variables in a NetCDF-4
    file and stores their data in a dictionary. The key for each dataset is its
    full hierarchical path.

    Args:
        file_path (str | Path): The path to the NetCDF file.

    Returns:
        Dict[str, Any]: A dictionary where keys are the full variable paths
                        and values are the corresponding NumPy arrays.
    """
    datasets: dict[str, Any] = {}
    file_path = Path(file_path)

    def _read_all_recursively(group: netCDF4.Group | netCDF4.Dataset, path: str = ""):
        for var_name, var_obj in group.variables.items():
            full_path = f"{path}/{var_name}" if path else var_name
            datasets[full_path] = var_obj[:]

        for group_name, group_obj in group.groups.items():
            new_path = f"{path}/{group_name}" if path else group_name
            _read_all_recursively(group_obj, new_path)

    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return {}

    with netCDF4.Dataset(file_path, "r") as nc_file:
        _read_all_recursively(nc_file)

    return datasets


def read_all_datasets_h5(file_path: str | Path) -> dict[str, Any]:
    """Reads all datasets (variables) from an HDF5 file, including those in groups.

    This function recursively traverses all groups and datasets in an HDF5
    file and stores their data in a dictionary. The key for each dataset is its
    full hierarchical path.

    Args:
        file_path (str | Path): The path to the HDF5 file.

    Returns:
        Dict[str, Any]: A dictionary where keys are the full dataset paths
                        and values are the corresponding NumPy arrays.
    """
    import h5py

    datasets: dict[str, Any] = {}
    file_path = Path(file_path)

    def _read_all_recursively(group: h5py.Group, path: str = ""):
        for name, item in group.items():
            full_path = f"{path}/{name}" if path else name
            if isinstance(item, h5py.Dataset):
                datasets[full_path] = item[()]
            elif isinstance(item, h5py.Group):
                _read_all_recursively(item, full_path)

    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return {}

    with h5py.File(file_path, "r") as h5_file:
        _read_all_recursively(h5_file)

    return datasets


def read_all_datasets_cdf(file_path: str | Path) -> dict[str, Any]:
    """Reads all datasets (variables) from a CDF file, including those in groups.

    This function recursively traverses all groups and variables in a CDF file
    and stores their data in a dictionary. The key for each dataset is its full
    hierarchical path.

    Args:
        file_path (str | Path): The path to the CDF file.
    Returns:
        Dict[str, Any]: A dictionary where keys are the full variable paths
                        and values are the corresponding NumPy arrays.
    """
    pass


def load_file_any_format(file_path: Path) -> dict[str, Any]:
    """Load a file in any supported format and return its content."""
    match file_path.suffix:
        case ".mat":
            try:
                file_content = typing.cast(dict[str, NDArray[np.generic] | str], loadmat(file_path))
            except TypeError:
                file_content = typing.cast(
                    dict[str, NDArray[np.generic] | str],
                    sci_loadmat(file_path, squeeze_me=True),
                )

        case ".pickle":
            with file_path.open("rb") as file:
                file_content = typing.cast(dict[str, NDArray[np.generic] | str], pickle.load(file))

        case _:
            msg = f"Loading file extension {file_path.suffix} is not supported yet!"
            raise NotImplementedError(msg)

    return file_content


def round_seconds(obj: datetime) -> datetime:
    """Round datetime object to the nearest second."""
    if obj.microsecond >= 500_000:
        obj += timedelta(seconds=1)
    return obj.replace(microsecond=0)


def python2matlab(datenum: datetime) -> float:
    """Convert Python datetime to MATLAB datenum."""
    mdn = datenum + timedelta(days=366)
    frac = (datenum - datetime(datenum.year, datenum.month, datenum.day, 0, 0, 0, tzinfo=timezone.utc)).seconds / (
        24.0 * 60.0 * 60.0
    )
    return mdn.toordinal() + round(frac, 6)


def matlab2python(datenum: float | Iterable[float]) -> Iterable[datetime] | datetime:
    """Convert MATLAB datenum to Python datetime."""
    warnings.filterwarnings("ignore", message="Discarding nonzero nanoseconds in conversion")

    datenum = np.asarray(datenum, dtype=float)
    datenum = pd.to_datetime(datenum - 719529, unit="D", origin=pd.Timestamp("1970-01-01")).to_pydatetime()  # ty:ignore[unresolved-attribute]

    if isinstance(datenum, Iterable):
        datenum = enforce_utc_timezone(list(datenum))  # ty:ignore[no-matching-overload]
        datenum = [  # ty:ignore[invalid-assignment]
            round_seconds(x) for x in datenum
        ]
    else:
        datenum = round_seconds(enforce_utc_timezone(datenum))  # ty:ignore[invalid-assignment]

    return datenum


def pol2cart(
    theta: NDArray[np.float64], radius: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """transforms polar coordinates theta (in rad) and radius to cartesian coordinates x, y"""
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return (x, y)


def cart2pol(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """transforms cartesian coordinates x, y to polar coordinates theta (in rad) and radius"""
    z = x + 1j * y
    return np.angle(z), np.abs(z)
