from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, overload

from data_management.io.RBMDataSet.custom_enums import (
    FolderTypeEnum,
    InstrumentEnum,
    MfmEnum,
    Satellite,
    SatelliteEnum,
    SatelliteLike,
)
from data_management.io.RBMDataSet.RBMDataSet import RBMDataSet

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
                    start_time,
                    end_time,
                    folder_path,
                    sat,
                    instrument,
                    mfm,
                    verbose=verbose,
                    preferred_extension=preferred_extension,
                )
                return_list.append(cls._instance.data_set_dict[key_tuple])

        if len(return_list) == 1:
            return_list = return_list[0]

        return return_list
