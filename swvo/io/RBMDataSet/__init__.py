# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""RBM dataset loading utilities.

.. deprecated::

    This subpackage is deprecated. RBM dataset handling now lives in
    `el_paso <https://github.com/GFZ/EL_PASO>`_. It is kept here only for
    backward compatibility and will not receive new features.
"""

from swvo.io.RBMDataSet.custom_enums import (
    FolderTypeEnum as FolderTypeEnum,
    FileCadenceEnum as FileCadenceEnum,
    Variable as Variable,
    VariableEnum as VariableEnum,
    Satellite as Satellite,
    SatelliteLike as SatelliteLike,
    SatelliteEnum as SatelliteEnum,
    Instrument as Instrument,
    InstrumentEnum as InstrumentEnum,
    InstrumentLike as InstrumentLike,
    Mfm as Mfm,
    MfmEnum as MfmEnum,
    MfmLike as MfmLike,
    SatelliteLiteral as SatelliteLiteral,
    VariableLiteral as VariableLiteral,
)
from swvo.io.RBMDataSet.RBMDataSet import RBMDataSet as RBMDataSet
from swvo.io.RBMDataSet.interp_functions import TargetType as TargetType
from swvo.io.RBMDataSet.scripts.create_RBSP_line_data import create_RBSP_line_data as create_RBSP_line_data
