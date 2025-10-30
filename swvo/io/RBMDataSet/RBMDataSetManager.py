# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, overload

from swvo.io.RBMDataSet.custom_enums import (
    FolderTypeEnum,
    InstrumentEnum,
    MfmEnum,
    Satellite,
    SatelliteEnum,
    SatelliteLike,
)
from swvo.io.RBMDataSet.RBMDataSet import RBMDataSet

RBMDataSetHash = tuple[
    datetime,
    datetime,
    Path,
    Satellite | SatelliteEnum,
    InstrumentEnum,
    MfmEnum,
    FolderTypeEnum,
]


class RBMDataSetManager:
    """
    RBMDataSetManager class for managing RBMDataSet instances.

    Notes
    -----
    Use the `load` class method to create and retrieve datasets. Direct instantiation is not allowed.

    Raises
    ------
    RuntimeError
        If the constructor is called directly instead of using the `load` method.
    """

    _instance = None
    data_set_dict: dict[RBMDataSetHash, RBMDataSet]

    def __init__(self) -> None:
        msg = "Call load() instead!"
        raise RuntimeError(msg)

    @overload
    @classmethod
    def load(
        cls,
        start_time: datetime,
        end_time: datetime,
        folder_path: Path,
        satellite: SatelliteLike,
        instrument: InstrumentEnum,
        mfm: MfmEnum,
        folder_type: FolderTypeEnum = FolderTypeEnum.DataServer,
        *,
        verbose: bool = True,
        preferred_extension: str = "pickle",
    ) -> RBMDataSet: ...

    @overload
    @classmethod
    def load(
        cls,
        start_time: datetime,
        end_time: datetime,
        folder_path: Path,
        satellite: Iterable[SatelliteLike],
        instrument: InstrumentEnum,
        mfm: MfmEnum,
        folder_type: FolderTypeEnum = FolderTypeEnum.DataServer,
        *,
        verbose: bool = True,
        preferred_extension: str = "pickle",
    ) -> list[RBMDataSet]: ...

    @classmethod
    def load(
        cls,
        start_time: datetime,
        end_time: datetime,
        folder_path: Path,
        satellite: SatelliteLike | Iterable[SatelliteLike],
        instrument: InstrumentEnum,
        mfm: MfmEnum,
        folder_type: FolderTypeEnum = FolderTypeEnum.DataServer,
        *,
        verbose: bool = True,
        preferred_extension: str = "pickle",
    ) -> RBMDataSet | list[RBMDataSet]:
        """Loads an RBMDataSet or a list of RBMDataSets based on the provided parameters.

        Parameters
        ----------
        start_time : datetime
            Start time of the data set.
        end_time : datetime
            End time of the data set.
        folder_path : Path
            Path to the folder where the data set is stored.
        satellite : :class:`SatelliteLike` | Iterable[:class:`SatelliteLike`]
            Satellite identifier(s) as enum or string. If a single satellite is provided, it can be a string or an enum.
        instrument : :class:`InstrumentEnum`
            Instrument enumeration, e.g., :class:`InstrumentEnum.HOPE`.
        mfm : :class:`MfmEnum`
            Magnetic field model enum, e.g., :class:`MfmEnum.T89`.
        folder_type : :class:`FolderTypeEnum`, optional
            Type of folder where the data is stored, by default :class:`FolderTypeEnum.DataServer`.
        verbose : bool, optional
            Whether to print verbose output, by default True.
        preferred_extension : str, optional
            Preferred file extension for the data set to be loaded, by default "pickle".

        Returns
        -------
        Union[:class:`RBMDataSet`, list[:class:`RBMDataSet`]]
            An instance of RBMDataSet or a list of RBMDataSet instances, depending on the input parameters.
            Variables are lazily loaded from the file system when accessed.
        """
        if cls._instance is None:
            print("Initiating new RBMDataSetManager!")
            cls._instance = cls.__new__(cls)
            cls._instance.data_set_dict = {}

        if isinstance(satellite, str):
            satellite = SatelliteEnum[satellite]

        if not isinstance(satellite, Iterable):
            satellite = (satellite,)

        return_list: list[RBMDataSet] | RBMDataSet = []
        for sat in satellite:
            key_tuple = (
                start_time,
                end_time,
                folder_path,
                sat,
                instrument,
                mfm,
                folder_type,
            )

            if key_tuple in cls._instance.data_set_dict:
                return_list.append(cls._instance.data_set_dict[key_tuple])
            else:
                cls._instance.data_set_dict[key_tuple] = RBMDataSet(
                    satellite=sat,
                    instrument=instrument,
                    mfm=mfm,
                    start_time=start_time,
                    end_time=end_time,
                    folder_path=folder_path,
                    verbose=verbose,
                    preferred_extension=preferred_extension,
                )
                return_list.append(cls._instance.data_set_dict[key_tuple])

        if len(return_list) == 1:
            return_list = return_list[0]

        return return_list
