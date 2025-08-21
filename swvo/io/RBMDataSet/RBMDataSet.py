# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
import typing
import warnings
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import distance
import numpy as np
from dateutil.relativedelta import relativedelta
from numpy.typing import NDArray

from swvo.io.RBMDataSet import (
    FileCadenceEnum,
    FolderTypeEnum,
    InstrumentEnum,
    MfmEnum,
    SatelliteEnum,
    SatelliteLike,
    Variable,
    VariableEnum,
)
from swvo.io.RBMDataSet.utils import (
    get_file_path_any_format,
    join_var,
    load_file_any_format,
    matlab2python,
)


class RBMDataSet:
    """RBMDataSet class for loading and managing data.

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
        start_time: dt.datetime,
        end_time: dt.datetime,
        folder_path: Path,
        satellite: SatelliteLike,
        instrument: InstrumentEnum,
        mfm: MfmEnum,
        preferred_extension: str = "pickle",
        *,
        verbose: bool = True,
    ) -> None:
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        self._start_time = start_time
        self._end_time = end_time

        if isinstance(satellite, str):
            satellite = SatelliteEnum[satellite.upper()]
        self._satellite = satellite
        self._instrument = instrument
        self._mfm = mfm
        self._folder_path = Path(folder_path)

        self._preferred_ext = preferred_extension
        self._folder_type = self._satellite.folder_type
        self._verbose = verbose

        self._file_path_stem = self._create_file_path_stem()
        self._file_name_stem = self._create_file_name_stem()
        self._file_cadence = self._satellite.file_cadence
        self._date_of_files = self._create_date_list()

    def __repr__(self):
        return f"{self.__class__.__name__}({self._satellite}, {self._instrument}, {self._mfm})"
    
    def __str__(self):
        return self.__repr__()


    def __dir__(self):
        return super().__dir__() + [var.var_name for var in VariableEnum]

    def __getattr__(self, name: str):
        # check if a sat variable is requested
        # if we find a similar word, suggest that to the user
        sat_variable = None
        levenstein_info: dict[str, Any] = {"min_distance": 10, "var_name": ""}
        for var in VariableEnum:
            if name == var.var_name:
                sat_variable = var
                break
            else:
                dist = distance.levenshtein(name, var.var_name)
                if name.lower() in var.name.lower():
                    dist = 1

                if dist < levenstein_info["min_distance"]:
                    levenstein_info["min_distance"] = dist
                    levenstein_info["var_name"] = var.var_name

        # if yes, load it
        if sat_variable is not None:
            self._load_variable(sat_variable)

            return getattr(self, name)

        if levenstein_info["min_distance"] <= 2:
            msg = f"{self.__class__.__name__} object has no attribute {name}. Maybe you meant {levenstein_info['var_name']}?"
        else:
            msg = f"{self.__class__.__name__} object has no attribute {name}"

        raise AttributeError(msg)

    # def __getitem__(self, key:str):
    #     return getattr(self, key:str)

    # def __setitem__(self, key, value):
    #     setattr(self, key, value)

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

                file_name_no_format = self._file_name_stem + date_str + "_" + var.data_server_file_prefix

                if var.data_server_has_B:
                    file_name_no_format += "_" + self._mfm.value

                file_name_no_format += "_ver4"
            else:
                raise NotImplementedError

            full_file_path = get_file_path_any_format(self._file_path_stem, file_name_no_format, self._preferred_ext)

            if full_file_path is None:
                print(f"File not found {full_file_path}")
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

    from .bin_and_interpolate_to_model_grid import bin_and_interpolate_to_model_grid
    from .interp_functions import interp_flux
