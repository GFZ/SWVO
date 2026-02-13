# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle
import typing
import warnings
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mat73 import loadmat
from numpy.typing import NDArray
from scipy.io import loadmat as sci_loadmat

from swvo.io.utils import enforce_utc_timezone


def join_var(var1: NDArray[np.generic], var2: NDArray[np.generic]) -> NDArray[np.generic]:
    """Join two variables along the first axis."""
    return np.concatenate((var1, var2), axis=0)


def get_file_path_any_format(folder_path: Path, file_stem: str, preferred_ext: str) -> Path | None:
    """Get the file path for a given file stem and preferred extension."""
    all_files = list(folder_path.glob(file_stem + ".*"))

    if len(all_files) == 0:
        warnings.warn(f"File not found: {folder_path / (file_stem + '.*')}", stacklevel=2)
        return None

    if len(all_files) >= 1:
        extensions_found = [file.suffix[1:] for file in all_files]
        if len(all_files) > 1:
            if preferred_ext in extensions_found:
                warnings.warn(
                    (
                        f"Several files found for {folder_path / (file_stem + '.*')} with extensions: {extensions_found}. "
                        f"Choosing: {preferred_ext}."
                    ),
                    stacklevel=2,
                )

                return folder_path / (file_stem + "." + preferred_ext)

            msg = (
                f"Several files found for {folder_path / (file_stem + '.*')} with extensions: {extensions_found}. "
                f"However, the preferred extension ({preferred_ext}) is not available!"
            )
            raise ValueError(msg)

        if len(all_files) == 1:
            return all_files[0]

        warnings.warn(
            f"File not found: {folder_path / (file_stem + '.' + preferred_ext)}",
            stacklevel=2,
        )

    return None


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
    datenum = pd.to_datetime(datenum - 719529, unit="D", origin=pd.Timestamp("1970-01-01")).to_pydatetime()

    if isinstance(datenum, Iterable):
        datenum = enforce_utc_timezone(list(datenum))
        datenum = [round_seconds(x) for x in datenum]
    else:
        datenum = round_seconds(enforce_utc_timezone(datenum))

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
