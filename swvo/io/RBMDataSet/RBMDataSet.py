# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
import typing
from dataclasses import replace
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import distance
import numpy as np
from dateutil.relativedelta import relativedelta
from numpy.typing import NDArray

from swvo.io.exceptions import VariableNotFoundError
from swvo.io.RBMDataSet import (
    FileCadenceEnum,
    FolderTypeEnum,
    InstrumentEnum,
    InstrumentLike,
    MfmEnum,
    MfmLike,
    SatelliteEnum,
    SatelliteLike,
    Variable,
    VariableEnum,
    VariableLiteral,
)
from swvo.io.RBMDataSet.custom_enums import DummyEnum, DummyLike
from swvo.io.RBMDataSet.utils import (
    get_file_path_any_format,
    join_var,
    load_file_any_format,
    matlab2python,
)


class RBMDataSet:
    """RBMDataSet class for loading and managing data.

    This class can load data either from files or from a dictionary (ElPaso format).

    For file-based loading, provide start_time, end_time, and folder_path.
    For dictionary-based loading, initialize without these parameters and use update_from_dict().

    Parameters
    ----------
    satellite : :class:`SatelliteLike`
        Satellite identifier as enum or string.
    instrument : :class:`InstrumentLike`
        Instrument enumeration or string.
    mfm : :class:`MfmLike`
        Magnetic field model enum or string.
    start_time : dt.datetime, optional
        Start time for file-based loading.
    end_time : dt.datetime, optional
        End time for file-based loading.
    folder_path : Path, optional
        Base folder path for file-based loading.
    preferred_extension : Literal["mat", "pickle"], optional
        Preferred file extension for file-based loading. Default is "pickle".
    verbose : bool, optional
        Whether to print verbose output. Default is True.

    Attributes
    ----------
    datetime : list[dt.datetime]
    time : NDArray[np.float64]
    energy_channels : NDArray[np.float64]
    alpha_local : NDArray[np.float64]
    alpha_eq_model : NDArray[np.float64]
    alpha_eq_real : NDArray[np.float64]
    InvMu : NDArray[np.float64]
    InvMu_real : NDArray[np.float64]
    InvK : NDArray[np.float64]
    InvV : NDArray[np.float64]
    Lstar : NDArray[np.float64]
    Flux : NDArray[np.float64]
    PSD : NDArray[np.float64]
    MLT : NDArray[np.float64]
    B_SM : NDArray[np.float64]
    B_total : NDArray[np.float64]
    B_sat : NDArray[np.float64]
    xGEO : NDArray[np.float64]
    P : NDArray[np.float64]
    R0 : NDArray[np.float64]
    density : NDArray[np.float64]

    """

    _preferred_ext: str

    datetime: list[dt.datetime]
    time: NDArray[np.float64]
    energy_channels: NDArray[np.float64]
    alpha_local: NDArray[np.float64]
    alpha_eq_model: NDArray[np.float64]
    alpha_eq_real: NDArray[np.float64]
    InvMu: NDArray[np.float64]
    InvMu_real: NDArray[np.float64]
    InvK: NDArray[np.float64]
    InvV: NDArray[np.float64]
    Lstar: NDArray[np.float64]
    Flux: NDArray[np.float64]
    PSD: NDArray[np.float64]
    MLT: NDArray[np.float64]
    B_SM: NDArray[np.float64]
    B_total: NDArray[np.float64]
    B_sat: NDArray[np.float64]
    xGEO: NDArray[np.float64]
    P: NDArray[np.float64]
    R0: NDArray[np.float64]
    density: NDArray[np.float64]

    def __init__(
        self,
        satellite: SatelliteLike | DummyLike,
        instrument: InstrumentLike | DummyLike,
        mfm: MfmLike | DummyLike,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        folder_path: Path | None = None,
        preferred_extension: Literal["mat", "pickle"] = "pickle",
        *,
        verbose: bool = True,
    ) -> None:
        self.ep_variables = list(VariableLiteral.__args__)
        # Handle satellite conversion with special cases for GOES
        if isinstance(satellite, str):
            if satellite.lower() == "goesprimary":
                satellite = SatelliteEnum["GOESPrimary"]
            elif satellite.lower() == "goessecondary":
                satellite = SatelliteEnum["GOESSecondary"]
            else:
                satellite = SatelliteEnum[satellite.upper()]

        if isinstance(instrument, str):
            instrument = InstrumentEnum[instrument.upper()]

        if isinstance(mfm, str):
            mfm = MfmEnum[mfm.upper()]

        # Store the original satellite enum for properties and other attributes
        self._satellite_enum = satellite
        self._instrument = instrument
        self._mfm = mfm
        self._verbose = verbose

        # For dict-based loading (ElPaso mode), modify satellite properties
        if start_time is None and end_time is None and folder_path is None:
            # no file loading needed
            satellite_obj = replace(
                satellite.value,
                folder_type=FolderTypeEnum.NoFolder,
                file_cadence=FileCadenceEnum.NoCadence,
            )
            self._satellite = satellite_obj
            self._mfm_prefix = DummyEnum.MFM.value if isinstance(mfm, DummyEnum) else MfmEnum[mfm.name].value
            self._file_loading_mode = False
        else:
            # File loading mode: need all parameters
            if start_time is None or end_time is None or folder_path is None:
                msg = "For file-based loading, start_time, end_time, and folder_path must all be provided"
                raise ValueError(msg)

            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)

            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)

            self._start_time = start_time
            self._end_time = end_time
            self._satellite = satellite
            self._folder_path = Path(folder_path)
            self._preferred_ext = preferred_extension
            self._folder_type = self._satellite.folder_type
            self._file_path_stem = self._create_file_path_stem()
            self._file_name_stem = self._create_file_name_stem()
            self._file_cadence = self._satellite.file_cadence
            self._date_of_files = self._create_date_list()
            self._file_loading_mode = True

    def __repr__(self):
        return f"{self.__class__.__name__}({self._satellite_enum}, {self._instrument}, {self._mfm})"

    def __str__(self):
        return self.__repr__()

    def __dir__(self):
        return list(super().__dir__()) + [var.var_name for var in VariableEnum]

    def __getattr__(self, name: str):
        # Avoid recursion for internal attributes
        if name.startswith("_"):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        # Handle computed properties for both modes
        if name == "P":
            if not hasattr(self, "MLT") or getattr(self, "MLT") is None or not isinstance(self.MLT, np.ndarray):
                raise AttributeError("Cannot compute `P` because `MLT` is missing, not loaded or is not valid array.")
            return ((self.MLT + 12) / 12 * np.pi) % (2 * np.pi)

        if name == "InvV":
            if not all(hasattr(self, attr) for attr in ("InvK", "InvMu")):
                raise AttributeError("Cannot compute `InvV` because `InvK` or `InvMu` is missing.")
            if not isinstance(self.InvK, np.ndarray) or not isinstance(self.InvMu, np.ndarray):
                raise AttributeError("Cannot compute `InvV` because required arrays are invalid or not loaded.")
            if self.InvK.ndim < 1 or self.InvMu.ndim < 2:
                raise AttributeError("Cannot compute `InvV` because array dimensions are insufficient.")
            inv_K_repeated = np.repeat(self.InvK[:, np.newaxis, :], self.InvMu.shape[1], axis=1)
            return self.InvMu * (inv_K_repeated + 0.5) ** 2

        # check if a sat variable is requested
        # if we find a similar word, suggest that to the user
        sat_variable = None
        sat_variable, levenstein_info = self.find_similar_variable(name)

        if sat_variable is not None and self._file_loading_mode:
            self._load_variable(sat_variable)
            return getattr(self, name)

        if not self._file_loading_mode and name in self.ep_variables:
            raise AttributeError(
                f"Attribute '{name}' exists in `VariableLiteral` but has not been set. "
                "Call `update_from_dict()` before accessing it."
            )

        if levenstein_info["min_distance"] <= 2:
            msg = f"{self.__class__.__name__} object has no attribute {name}. Maybe you meant {levenstein_info['var_name']}?"
        else:
            msg = f"{self.__class__.__name__} object has no attribute {name}"

        raise AttributeError(msg)

    def find_similar_variable(self, name):
        levenstein_info: dict[str, Any] = {"min_distance": 10, "var_name": ""}
        sat_variable = None
        for var in self.ep_variables:
            if name == var:
                sat_variable = var
                break
            else:
                dist = distance.levenshtein(name, var)
                if name.lower() in var.lower():
                    dist = 1

                if dist < levenstein_info["min_distance"]:
                    levenstein_info["min_distance"] = dist
                    levenstein_info["var_name"] = var
        return sat_variable, levenstein_info

    @property
    def satellite(self) -> SatelliteEnum:
        """Returns the satellite enum."""
        return self._satellite_enum

    @property
    def instrument(self) -> InstrumentEnum:
        """Returns the instrument enum."""
        return self._instrument

    @property
    def mfm(self) -> MfmEnum:
        """Returns the MFM enum."""
        return self._mfm

    def update_from_dict(self, source_dict: dict[str, VariableLiteral]) -> None:
        """Get data from ElPaso data dictionary and update the object.

        Parameters
        ----------
        source_dict : dict[str, VariableLiteral]
            Dictionary containing the data to be loaded into the object.

        """
        for key, value in source_dict.items():
            _, levenstein_info = self.find_similar_variable(key)
            if key in self.ep_variables:
                setattr(self, key, value)
            elif levenstein_info["min_distance"] <= 2:
                msg = f"Key '{key}' is not a valid `VariableLiteral`. Maybe you meant '{levenstein_info['var_name']}'?"
                raise VariableNotFoundError(msg)
            else:
                msg = f"Key '{key}' is not a valid `VariableLiteral`."
                raise VariableNotFoundError(msg)

    def get_var(self, var: VariableEnum):
        return getattr(self, var.var_name)

    def _create_date_list(self) -> list[dt.datetime]:
        match self._file_cadence:
            case FileCadenceEnum.Daily:
                time_delta = timedelta(days=1)
            case FileCadenceEnum.Monthly:
                time_delta = relativedelta(months=1)
            case _:
                msg = "Encounterd invalid file cadence!"
                raise ValueError(msg)

        start_date = self._start_time.date()
        date_of_files = np.asarray([dt.datetime(start_date.year, start_date.month, 1, tzinfo=timezone.utc)])
        while (date_of_files[-1] + time_delta) < self._end_time:
            date_of_files = np.append(date_of_files, date_of_files[-1] + time_delta)

        return list(date_of_files)

    def _create_file_path_stem(self) -> Path:
        # implement special cases here
        # if self._satellite == SatelliteEnum.THEMIS:
        #     pass
        if self._folder_type == FolderTypeEnum.DataServer:
            return self._folder_path / self._satellite.mission / self._satellite.sat_name / "Processed_Mat_Files"

        if self._folder_type == FolderTypeEnum.SingleFolder:
            return self._folder_path

        msg = "Encountered invalid FolderTypeEnum!"
        raise ValueError(msg)

    def _create_file_name_stem(self) -> str:
        # implement special cases here
        # if self._satellite == SatelliteEnum.THEMIS:
        #     pass

        return self._satellite.sat_name + "_" + self._instrument.value + "_"

    def get_satellite_name(self) -> str:
        return self._satellite.sat_name

    def get_satellite_and_instrument_name(self) -> str:
        return self._satellite.sat_name + "_" + self._instrument.value

    def set_file_path_stem(self, file_path_stem: Path):
        self._file_path_stem = file_path_stem
        return self

    def set_file_name_stem(self, file_name_stem: Path):
        self._file_path_stem = file_name_stem
        return self

    def set_file_cadence(self, file_cadence: FileCadenceEnum):
        self._file_cadence = file_cadence
        self._date_of_files = self._create_date_list()
        return self

    def get_print_name(self) -> str:
        return self._satellite.sat_name + " " + self._instrument.value

    def _load_variable(self, var: Variable | VariableEnum) -> None:
        loaded_var_arrs: dict[str, NDArray[np.number]] = {}
        var_names_storred: list[str] = []

        # computed values
        if isinstance(var, VariableEnum) and var == VariableEnum.INV_V:
            inv_K_repeated = np.repeat(self.InvK[:, np.newaxis, :], self.InvMu.shape[1], axis=1)

            self.InvV = self.InvMu * (inv_K_repeated + 0.5) ** 2
            return

        if isinstance(var, VariableEnum) and var == VariableEnum.P:
            self.P = ((self.MLT + 12) / 12 * np.pi) % (2 * np.pi)
            return

        for date in self._date_of_files:
            if self._folder_type == FolderTypeEnum.DataServer:
                start_month = date.replace(day=1)
                next_month = start_month + relativedelta(months=1, days=-1)
                date_str = start_month.strftime("%Y%m%d") + "to" + next_month.strftime("%Y%m%d")

                file_name_no_format = self._file_name_stem + date_str + "_" + var.mat_file_prefix

                if var.mat_has_B:
                    file_name_no_format += "_n4_4_" + self._mfm.value

                file_name_no_format += "_ver4"
            else:
                raise NotImplementedError

            full_file_path = get_file_path_any_format(self._file_path_stem, file_name_no_format, self._preferred_ext)

            if full_file_path is None:
                print(f"File not found: {self._file_path_stem}, {file_name_no_format}")
                continue

            if self._verbose:
                print(f"\tLoading {full_file_path}")

            file_content = load_file_any_format(full_file_path)

            # also store python datetimes for binning
            datetimes = typing.cast(
                NDArray[np.object_],
                np.asarray([matlab2python(t) for t in file_content["time"]]),
            )  # type: ignore
            file_content["datetime"] = datetimes

            # limit in time
            correct_time_idx = (datetimes >= self._start_time) & (datetimes <= self._end_time)

            for key in file_content:
                # if key == 'time' and var not in [VariableEnum.Time, VariableEnum.DateTime]:
                # only save time if directly requested
                #    continue

                var_arr = file_content[key]
                if ((not isinstance(var_arr, np.ndarray)) or (not np.issubdtype(var_arr.dtype, np.number))) and (
                    key != "datetime"
                ):
                    # var represents some strings or metadata objects; don't read them
                    continue
                var_arr = typing.cast(NDArray[np.number], var_arr)

                # check if var is time dependent
                if var_arr.shape[0] == correct_time_idx.shape[0]:
                    var_arr = var_arr[correct_time_idx.reshape(-1), ...]

                    joined_value = join_var(loaded_var_arrs[key], var_arr) if key in loaded_var_arrs else var_arr
                else:
                    joined_value = var_arr

                loaded_var_arrs[key] = joined_value

                if key not in var_names_storred:
                    var_names_storred.append(key)

        # not a single file was found
        if var.var_name not in var_names_storred:
            setattr(self, var.var_name, np.asarray([]))

        for var_name in var_names_storred:
            if var_name == "datetime":
                loaded_var_arrs[var_name] = list(loaded_var_arrs[var_name])  # type: ignore

            setattr(self, var_name, loaded_var_arrs[var_name])

    def get_loaded_variables(self) -> list[str]:
        """Get a list of currently loaded variable names."""
        loaded_vars = []
        for var in VariableEnum:
            if hasattr(self, var.var_name):
                loaded_vars.append(var.var_name)
        return loaded_vars

    def __eq__(self, other: RBMDataSet) -> bool:
        if (
            self._file_loading_mode != other._file_loading_mode
            or self._satellite != other._satellite
            or self._instrument != other._instrument
            or self._mfm != other._mfm
        ):
            return False

        self_vars = self.get_loaded_variables()
        other_vars = other.get_loaded_variables()
        if self_vars != other_vars:
            return False
        variables = self_vars

        for var in variables:
            self_var = getattr(self, var)
            other_var = getattr(other, var)

            if not isinstance(other_var, type(self_var)):
                return False

            if isinstance(self_var, list):
                if len(self_var) != len(other_var) or any(a != b for a, b in zip(self_var, other_var)):
                    return False
            elif isinstance(self_var, np.ndarray):
                if self_var.shape != other_var.shape or not np.allclose(self_var, other_var, equal_nan=True):
                    return False
            elif self_var != other_var:
                return False

        return True

    from .bin_and_interpolate_to_model_grid import bin_and_interpolate_to_model_grid
    from .interp_functions import interp_flux
