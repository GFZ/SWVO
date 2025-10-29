# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

####


class FolderTypeEnum(Enum):
    Default = 0
    SingleFolder = 1
    DataServer = 2
    NoFolder = -1


class FileCadenceEnum(Enum):
    Daily = 0
    Monthly = 1
    NoCadence = -1


@dataclass(frozen=True)
class Variable:
    var_name: str
    mat_file_prefix: str
    mat_has_B: bool


without_B: bool = False
with_B: bool = True


class VariableEnum(Variable, Enum):
    DATETIME = "datetime", "mlt", with_B
    TIME = "time", "mlt", with_B
    ENERGY = "energy_channels", "alpha_and_energy", with_B
    ALPHA_LOCAL = "alpha_local", "alpha_and_energy", with_B
    ALPHA_EQ_MODEL = "alpha_eq_model", "alpha_and_energy", with_B
    ALPHA_EQ_REAL = "alpha_eq_real", "real_invmu_and_alpha", with_B
    INV_MU = "InvMu", "invmu_and_invk", with_B
    INV_MU_REAL = "InvMu_real", "real_invmu_and_alpha", with_B
    INV_K = "InvK", "invmu_and_invk", with_B
    INV_V = "InvV", "", without_B
    L_STAR = "Lstar", "lstar", with_B
    FLUX = "Flux", "flux", without_B
    PSD = "PSD", "psd", without_B
    MLT = "MLT", "mlt", with_B
    B_VEC = "B_SM", "bfield", with_B
    B_TOTAL = "B_total", "bfield", with_B
    B_SAT = "B_sat", "real_bfield", without_B
    XGEO = "xGEO", "xGEO", without_B
    P = "P", "mlt", with_B
    R_0 = "R0", "R0", with_B
    DENSITY = "density", "density", without_B


VariableLiteral = Literal[
    "datetime",
    "time",
    "energy_channels",
    "alpha_local",
    "alpha_eq_model",
    "alpha_eq_real",
    "InvMu",
    "InvMu_real",
    "InvK",
    "InvV",
    "Lstar",
    "Flux",
    "PSD",
    "MLT",
    "B_SM",
    "B_total",
    "B_sat",
    "xGEO",
    "P",
    "R0",
    "density",
]


@dataclass(frozen=True)
class Satellite:
    sat_name: str
    mission: str
    folder_type: FolderTypeEnum = FolderTypeEnum.DataServer
    file_cadence: FileCadenceEnum = FileCadenceEnum.Monthly


class SatelliteEnum(Satellite, Enum):
    RBSPA = "rbspa", "RBSP"
    RBSPB = "rbspb", "RBSP"

    GOES13 = "goes13", "GOES"
    GOES14 = "goes14", "GOES"
    GOES15 = "goes15", "GOES"

    GOESPrimary = "primary", "GOES"
    GOESSecondary = "secondary", "GOES"

    ARASE = "arase", "ARASE"

    NOAA15 = "noaa15", "poes"
    NOAA16 = "noaa16", "poes"
    NOAA18 = "noaa18", "poes"
    NOAA19 = "noaa19", "poes"
    METOP1 = "metop1", "poes"
    METOP2 = "metop2", "poes"

    DSX = "dsx", "DSX"


SatelliteLiteral = Literal[
    "RBSPA",
    "RBSPB",
    "GOES13",
    "GOES14",
    "GOES15",
    "GOESPrimary",
    "GOESSecondary",
    "ARASE",
    "NOAA15",
    "NOAA16",
    "NOAA18",
    "NOAA19",
    "METOP1",
    "METOP2",
    "DSX",
]
SatelliteLike: TypeAlias = SatelliteLiteral | SatelliteEnum | Satellite


class InstrumentEnum(Enum):
    # RBSP
    HOPE = "hope"
    MAGEIS = "mageis"
    REPT = "rept"
    ECT_COMBINED = "ect_combined"

    # GOES
    MAGEDandEPEAD = "MAGEDandEPEAD"
    MAGED = "MAGED"

    # ARASE
    XEP = "XEP"
    MEPE = "mepe-l3"
    PWE = "PWE-density"

    # DSX
    ORBIT = "orbit"

    # POES
    TED = "TED-electron"


InstrumentLiteral = Literal[
    "hope",
    "mageis",
    "rept",
    "ect_combined",
    "MAGEDandEPEAD",
    "MAGED",
    "XEP",
    "mepe",
    "PWE-density",
    "orbit",
    "TED-electron",
]
InstrumentLike: TypeAlias = InstrumentLiteral | InstrumentEnum


class MfmEnum(Enum):
    T89 = "T89"
    T04s = "T04s"
    T96 = "T96"
    TS04 = "T04s"
    OP77 = "OP77"
    T04 = "T04"


MfmEnumLiteral = Literal["T89", "T04s", "TS04", "T96", "OP77", "T04"]
MfmLike: TypeAlias = MfmEnumLiteral | MfmEnum


class DummyEnum(Enum):
    SATELLITE = Satellite(
        sat_name="dummy",
        mission="dummy",
        folder_type=FolderTypeEnum.NoFolder,
        file_cadence=FileCadenceEnum.NoCadence,
    )
    INSTRUMENT = "dummy instrument"
    MFM = "dummy mfm"


DummyLike: TypeAlias = Literal["dummy"] | DummyEnum
