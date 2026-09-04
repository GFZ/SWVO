# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from swvo.io.solar_wind.ace import SWACE as SWACE
from swvo.io.solar_wind.omni import SWOMNI as SWOMNI
from swvo.io.solar_wind.swift import SWSWIFTEnsemble as SWSWIFTEnsemble
from swvo.io.solar_wind.dscovr import DSCOVR as DSCOVR
from swvo.io.solar_wind.midl import SWMIDL as SWMIDL
from swvo.io.solar_wind.enlil import (
    SWENLIL as SWENLIL,
    SWENLIL_BKG as SWENLIL_BKG,
    SWENLIL_CME as SWENLIL_CME,
)

AVERAGE_VALUES_TO_FILL: dict[str, float] = {
    "bavg": 5.7501048842758955,
    "bx_gsm": -0.0008639005912272984,
    "by_gsm": -0.12753220183522668,
    "bz_gsm": -0.10594003748277739,
    "speed": 425.7842473380121,
    "proton_density": 6.593453185227736,
    "temperature": 91260.37300814023,
    "pdyn": 2.1816079947051628,
    "sym-h": -11.375495589373424,
}

# This has to be imported after the models and constants to avoid a circular import
from swvo.io.solar_wind.read_solar_wind_from_multiple_models import (  # noqa: E402
    read_solar_wind_from_multiple_models as read_solar_wind_from_multiple_models,
    SWModel as SWModel,
)
