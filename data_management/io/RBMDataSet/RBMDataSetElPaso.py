from __future__ import annotations

import datetime as dt
from datetime import timezone
from dataclasses import replace
from typing import Any, ClassVar, TypeVar
from data_management.io.RBMDataSet.custom_enums import ElPasoMFMEnum
from data_management.io.RBMDataSet.utils import matlab2python, python2matlab
import numpy as np
from numpy.typing import NDArray

from data_management.io.RBMDataSet import (
    FileCadenceEnum,
    FolderTypeEnum,
    InstrumentEnum,
    MfmEnum,
    SatelliteEnum,
    SatelliteLike,
    VariableEnum,
)

Variable = TypeVar(
    "Variable"
)  # this is a placeholder for the actual Variable class from elpaso and not the one in RBMDataSet


class RBMDataSetElPaso:
    """RBMDataSetElPaso class for loading ElPaso data to RBMDataSet.

    Parameters
    ----------
    satellite : :class:`SatelliteLike`
        Satellite identifier as enum or string.
    instrument : :class:`InstrumentEnum`
        Instrument enumeration.
    mfm : :class:`MfmEnum`
        Magnetic field model enum.


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

    _variable_mapping: ClassVar[dict[str, str]] = {
        "Epoch_posixtime": "time",
        "Energy_FEDU": "energy_channels",
        "Energy_FPDU": "energy_channels",
        "Energy_FEIU": "energy_channels",
        "Energy_FEDO": "energy_channels",
        "PA_local": "alpha_local",
        "PA_eq_": "alpha_eq_model",
        "alpha_eq_real": "alpha_eq_real",
        "invMu_": "InvMu",
        "InvMu_real": "InvMu_real",
        "invK_": "InvK",
        # "InvV": "InvV",# computed property
        "Lstar_": "Lstar",
        "FEDU": "Flux",
        "FPDU": "Flux",
        "FEIU": "Flux",
        "FEDO": "Flux",
        "PSD_FEDU": "PSD",
        "PSD_FPDU": "PSD",
        "PSD_FEIU": "PSD",
        "PSD_FEDO": "PSD",
        "MLT_": "MLT",
        "B_SM": "B_SM",
        "B_eq_": "B_total",
        "B_local_": "B_sat",
        "xGEO": "xGEO",
        # "P": "P",# computed property
        "R_eq_": "R0",
        "density": "density",
    }

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
        satellite: SatelliteLike,
        instrument: InstrumentEnum,
        mfm: MfmEnum,
    ) -> None:
        if isinstance(satellite, str):
            if satellite.lower() == "goesprimary":
                satellite_enum = SatelliteEnum["GOESPrimary"]
            elif satellite.lower() == "goessecondary":
                satellite_enum = SatelliteEnum["GOESSecondary"]
            else:
                satellite_enum = SatelliteEnum[satellite.upper()]
        else:
            satellite_enum = satellite

        if isinstance(instrument, str):
            instrument = InstrumentEnum[instrument.upper()]
        satellite_obj = replace(
            satellite_enum.value,
            folder_type=FolderTypeEnum.NoFolder,
            file_cadence=FileCadenceEnum.NoCadence,
        )

        self._satellite = satellite_obj
        self._instrument = instrument
        self._mfm = mfm
        self._mfm_prefix = ElPasoMFMEnum[self._mfm.name].value

    @property
    def satellite(self) -> SatelliteEnum:
        """Returns the satellite enum."""
        return self._satellite

    @property
    def instrument(self) -> InstrumentEnum:
        """Returns the instrument enum."""
        return self._instrument

    @property
    def mfm(self) -> MfmEnum:
        """Returns the MFM enum."""
        return self._mfm

    @property
    def variable_mapping(self) -> dict[str, str]:
        """Returns the variable mapping dictionary."""
        return self._variable_mapping

    def __dir__(self):
        return super().__dir__() + [var.var_name for var in VariableEnum]

    def update_from_dict(self, source_dict: dict[str, Variable]) -> None:
        """Get data from ElPaso data dictionary and update the object.

        Parameters
        ----------
        source_dict : dict[str, Any]
            Dictionary containing the data to be loaded into the object.

        """
        for _, value in source_dict.items():
            if value.standard_name in self._variable_mapping:
                target_attr = self._variable_mapping[value.standard_name]

                if value.standard_name == "Epoch_posixtime" and target_attr == "time":
                    datetimes = [
                        dt.datetime.fromtimestamp(ts, tz=timezone.utc)
                        for ts in value.data
                    ]
                    setattr(self, "datetime", datetimes)
                    setattr(self, "time", [python2matlab(i) for i in datetimes])
                else:
                    setattr(self, target_attr, value.data)

            elif value.standard_name.endswith(self._mfm_prefix):
                base_key = value.standard_name.replace(self._mfm_prefix, "")
                if base_key in self._variable_mapping:
                    target_attr = self._variable_mapping[base_key]
                    setattr(self, target_attr, value.data)

    @property
    def P(self) -> NDArray[np.float64]:
        """Calculate P.

        Returns
        -------
        NDArray[np.float64]
            The P value calculated from the MLT.
        """
        return ((self.MLT + 12) / 12 * np.pi) % (2 * np.pi)

    @property
    def InvV(self) -> NDArray[np.float64]:
        """Calculate InvV.

        Returns
        -------
        NDArray[np.float64]
            The InvV value calculated from InvMu and InvK.
        """
        inv_K_repeated = np.repeat(
            self.InvK[:, np.newaxis, :], self.InvMu.shape[1], axis=1
        )
        InvV = self.InvMu * (inv_K_repeated + 0.5) ** 2
        return InvV

    def __getattr__(self, name: str) -> Any:
        if name in self._variable_mapping.values():
            raise AttributeError(
                f"Attribute '{name}' is mapped but has not been set. "
                "Make sure data is loaded or that this attribute is properly assigned."
            )
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.satellite}, {self.instrument}, {self.mfm})"

    def __str__(self):
        return self.__repr__()
