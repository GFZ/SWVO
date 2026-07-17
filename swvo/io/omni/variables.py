# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""Variable metadata shared by the OMNI readers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class OMNIVariable:
    """Description of one processed OMNI variable."""

    name: str
    description: str
    unit: str
    fill_value: float | int | None
    nasa_id: int | None = None
    cadences: tuple[int, ...] = ()
    aliases: tuple[str, ...] = ()


HIGH_RES_VARIABLES: tuple[OMNIVariable, ...] = (
    OMNIVariable("imf_spacecraft_id", "IMF source spacecraft identifier", "", 99, 4, (1, 5)),
    OMNIVariable("plasma_spacecraft_id", "Solar-wind plasma source spacecraft identifier", "", 99, 5, (1, 5)),
    OMNIVariable("imf_point_count", "Fine-scale points in the IMF average", "count", 999, 6, (1, 5)),
    OMNIVariable("plasma_point_count", "Fine-scale points in the plasma average", "count", 999, 7, (1, 5)),
    OMNIVariable("percent_interpolated", "Percentage of interpolated values", "%", 999, 8, (1, 5)),
    OMNIVariable("timeshift", "Bow-shock-nose time shift", "s", 999999, 9, (1, 5), ("timeshift_seconds",)),
    OMNIVariable("timeshift_rms", "RMS time shift", "s", 999999, 10, (1, 5), ("timeshift_rms_seconds",)),
    OMNIVariable("minimum_variance_rms", "RMS minimum-variance vector", "", 99.99, 11, (1, 5)),
    OMNIVariable(
        "observation_gap",
        "Time between observations",
        "s",
        999999,
        12,
        (1, 5),
        ("observation_gap_seconds",),
    ),
    OMNIVariable("bavg", "Average IMF magnitude", "nT", 9999.99, 13, (1, 5)),
    OMNIVariable("bx_gsm", "IMF Bx in GSE/GSM coordinates", "nT", 9999.99, 14, (1, 5), ("bx_gse",)),
    OMNIVariable("by_gse", "IMF By in GSE coordinates", "nT", 9999.99, 15, (1, 5)),
    OMNIVariable("bz_gse", "IMF Bz in GSE coordinates", "nT", 9999.99, 16, (1, 5)),
    OMNIVariable("by_gsm", "IMF By in GSM coordinates", "nT", 9999.99, 17, (1, 5)),
    OMNIVariable("bz_gsm", "IMF Bz in GSM coordinates", "nT", 9999.99, 18, (1, 5)),
    OMNIVariable("sigma_bavg", "RMS standard deviation of IMF magnitude", "nT", 9999.99, 19, (1, 5)),
    OMNIVariable("sigma_b", "RMS standard deviation of the IMF vector", "nT", 9999.99, 20, (1, 5)),
    OMNIVariable("speed", "Solar-wind bulk speed", "km/s", 99999.9, 21, (1, 5)),
    OMNIVariable("vx_gse", "Solar-wind Vx in GSE coordinates", "km/s", 99999.9, 22, (1, 5)),
    OMNIVariable("vy_gse", "Solar-wind Vy in GSE coordinates", "km/s", 99999.9, 23, (1, 5)),
    OMNIVariable("vz_gse", "Solar-wind Vz in GSE coordinates", "km/s", 99999.9, 24, (1, 5)),
    OMNIVariable("proton_density", "Proton number density", "1/cm^3", 999.99, 25, (1, 5)),
    OMNIVariable("temperature", "Proton temperature", "K", 9999999.0, 26, (1, 5)),
    OMNIVariable("pdyn", "Solar-wind flow pressure", "nPa", 99.99, 27, (1, 5), ("flow_pressure",)),
    OMNIVariable("electric_field", "Convective electric field Ey", "mV/m", 999.99, 28, (1, 5), ("ey",)),
    OMNIVariable("plasma_beta", "Plasma beta", "", 999.99, 29, (1, 5)),
    OMNIVariable("alfven_mach_number", "Alfven Mach number", "", 999.9, 30, (1, 5)),
    OMNIVariable("spacecraft_x_gse", "Spacecraft X position in GSE coordinates", "Re", 9999.99, 31, (1, 5)),
    OMNIVariable("spacecraft_y_gse", "Spacecraft Y position in GSE coordinates", "Re", 9999.99, 32, (1, 5)),
    OMNIVariable("spacecraft_z_gse", "Spacecraft Z position in GSE coordinates", "Re", 9999.99, 33, (1, 5)),
    OMNIVariable("bsn_x_gse", "Bow-shock-nose X position in GSE coordinates", "Re", 9999.99, 34, (1, 5)),
    OMNIVariable("bsn_y_gse", "Bow-shock-nose Y position in GSE coordinates", "Re", 9999.99, 35, (1, 5)),
    OMNIVariable("bsn_z_gse", "Bow-shock-nose Z position in GSE coordinates", "Re", 9999.99, 36, (1, 5)),
    OMNIVariable("ae", "Auroral electrojet AE index", "nT", 99999, 37, (1, 5)),
    OMNIVariable("al", "Auroral electrojet AL index", "nT", 99999, 38, (1, 5)),
    OMNIVariable("au", "Auroral electrojet AU index", "nT", 99999, 39, (1, 5)),
    OMNIVariable("sym-d", "SYM/D geomagnetic index", "nT", 99999, 40, (1, 5), ("sym_d",)),
    OMNIVariable("sym-h", "SYM/H geomagnetic index", "nT", 99999, 41, (1, 5), ("sym_h",)),
    OMNIVariable("asy-d", "ASY/D geomagnetic index", "nT", 99999, 42, (1, 5), ("asy_d",)),
    OMNIVariable("asy-h", "ASY/H geomagnetic index", "nT", 99999, 43, (1, 5), ("asy_h",)),
    OMNIVariable("pcn", "Northern polar-cap index", "", 999.99, 44, (1, 5), ("pc",)),
    OMNIVariable("magnetosonic_mach_number", "Magnetosonic Mach number", "", 99.9, 45, (1, 5)),
    OMNIVariable("proton_flux_10_mev", "Integral proton flux above 10 MeV", "1/(cm^2 s sr)", 99999.99, 46, (5,)),
    OMNIVariable("proton_flux_30_mev", "Integral proton flux above 30 MeV", "1/(cm^2 s sr)", 99999.99, 47, (5,)),
    OMNIVariable("proton_flux_60_mev", "Integral proton flux above 60 MeV", "1/(cm^2 s sr)", 99999.99, 48, (5,)),
)


LOW_RES_VARIABLES: tuple[OMNIVariable, ...] = (
    OMNIVariable("bartels_rotation_number", "Bartels rotation number", "", 9999),
    OMNIVariable("imf_spacecraft_id", "IMF source spacecraft identifier", "", 99),
    OMNIVariable("plasma_spacecraft_id", "Solar-wind plasma source spacecraft identifier", "", 99),
    OMNIVariable("imf_point_count", "Fine-scale points in the IMF average", "count", 999),
    OMNIVariable("plasma_point_count", "Fine-scale points in the plasma average", "count", 999),
    OMNIVariable("bavg", "Average IMF magnitude", "nT", 999.9),
    OMNIVariable("magnitude_average_field_vector", "Magnitude of the average IMF vector", "nT", 999.9),
    OMNIVariable("latitude_average_field", "Latitude angle of the average IMF vector", "degree", 999.9),
    OMNIVariable("longitude_average_field", "Longitude angle of the average IMF vector", "degree", 999.9),
    OMNIVariable("bx_gse_gsm", "IMF Bx in GSE/GSM coordinates", "nT", 999.9, aliases=("bx_gse",)),
    OMNIVariable("by_gse", "IMF By in GSE coordinates", "nT", 999.9),
    OMNIVariable("bz_gse", "IMF Bz in GSE coordinates", "nT", 999.9),
    OMNIVariable("by_gsm", "IMF By in GSM coordinates", "nT", 999.9),
    OMNIVariable("bz_gsm", "IMF Bz in GSM coordinates", "nT", 999.9),
    OMNIVariable("sigma_bavg", "RMS standard deviation of IMF magnitude", "nT", 999.9),
    OMNIVariable("sigma_b", "RMS standard deviation of the IMF vector", "nT", 999.9),
    OMNIVariable("sigma_bx", "RMS standard deviation of IMF Bx", "nT", 999.9),
    OMNIVariable("sigma_by", "RMS standard deviation of IMF By", "nT", 999.9),
    OMNIVariable("sigma_bz", "RMS standard deviation of IMF Bz", "nT", 999.9),
    OMNIVariable("temperature", "Proton temperature", "K", 9999999.0),
    OMNIVariable("proton_density", "Proton number density", "1/cm^3", 999.9),
    OMNIVariable("speed", "Solar-wind bulk speed", "km/s", 9999.0),
    OMNIVariable("speed_longitude", "Solar-wind flow longitude in GSE coordinates", "degree", 999.9),
    OMNIVariable("speed_latitude", "Solar-wind flow latitude in GSE coordinates", "degree", 999.9),
    OMNIVariable("alpha_proton_ratio", "Alpha-to-proton density ratio", "", 9.999),
    OMNIVariable("pdyn", "Solar-wind flow pressure", "nPa", 99.99, aliases=("flow_pressure",)),
    OMNIVariable("sigma_temperature", "RMS standard deviation of proton temperature", "K", 9999999.0),
    OMNIVariable("sigma_proton_density", "RMS standard deviation of proton density", "1/cm^3", 999.9),
    OMNIVariable("sigma_speed", "RMS standard deviation of solar-wind speed", "km/s", 9999.0),
    OMNIVariable("sigma_speed_longitude", "RMS standard deviation of flow longitude", "degree", 999.9),
    OMNIVariable("sigma_speed_latitude", "RMS standard deviation of flow latitude", "degree", 999.9),
    OMNIVariable("sigma_alpha_proton_ratio", "RMS standard deviation of alpha-to-proton ratio", "", 9.999),
    OMNIVariable("electric_field", "Convective electric field", "mV/m", 999.99, aliases=("e",)),
    OMNIVariable("plasma_beta", "Plasma beta", "", 999.99),
    OMNIVariable("alfven_mach_number", "Alfven Mach number", "", 999.9),
    OMNIVariable("kp", "Three-hour Kp index", "", 99),
    OMNIVariable("sunspot_number", "Daily international sunspot number", "", 999),
    OMNIVariable("dst", "Dst geomagnetic index", "nT", 99999),
    OMNIVariable("ae", "Auroral electrojet AE index", "nT", 9999),
    OMNIVariable("proton_flux_1_mev", "Integral proton flux above 1 MeV", "1/(cm^2 s sr)", 999999.99),
    OMNIVariable("proton_flux_2_mev", "Integral proton flux above 2 MeV", "1/(cm^2 s sr)", 99999.99),
    OMNIVariable("proton_flux_4_mev", "Integral proton flux above 4 MeV", "1/(cm^2 s sr)", 99999.99),
    OMNIVariable("proton_flux_10_mev", "Integral proton flux above 10 MeV", "1/(cm^2 s sr)", 99999.99),
    OMNIVariable("proton_flux_30_mev", "Integral proton flux above 30 MeV", "1/(cm^2 s sr)", 99999.99),
    OMNIVariable("proton_flux_60_mev", "Integral proton flux above 60 MeV", "1/(cm^2 s sr)", 99999.99),
    OMNIVariable("magnetospheric_flux_flag", "Magnetospheric proton-flux contamination flag", "", None),
    OMNIVariable("ap", "Three-hour ap index", "nT", 999),
    OMNIVariable("f107", "Daily F10.7 solar radio flux adjusted to 1 AU", "10^-22 W/(m^2 Hz)", 999.9),
    OMNIVariable("pcn", "Northern polar-cap index", "", 999.9, aliases=("pc",)),
    OMNIVariable("al", "Auroral electrojet AL index", "nT", 99999),
    OMNIVariable("au", "Auroral electrojet AU index", "nT", 99999),
    OMNIVariable("magnetosonic_mach_number", "Magnetosonic Mach number", "", 99.9),
    OMNIVariable("lyman_alpha", "Daily solar Lyman-alpha irradiance", "W/m^2", 0.999999),
    OMNIVariable("proton_quasi_invariant", "Solar-wind proton quasi-invariant", "", 9.9999),
)


HIGH_RES_DEFAULT_VARIABLES: tuple[str, ...] = (
    "bavg",
    "bx_gsm",
    "by_gsm",
    "bz_gsm",
    "speed",
    "proton_density",
    "temperature",
    "pdyn",
    "sym-h",
)

LOW_RES_DEFAULT_VARIABLES: tuple[str, ...] = ("dst", "kp", "f107")


def resolve_variable_names(
    registry: Sequence[OMNIVariable],
    variables: str | Iterable[str] | None,
    default_names: Sequence[str],
    cadence: int | None = None,
) -> list[str]:
    """Resolve user input to unique canonical variable names.

    Aliases are normalized, duplicates are removed while preserving the first
    occurrence, and ``"all"`` is restricted to the requested cadence.

    Raises
    ------
    ValueError
        If the selection is empty, not an iterable of strings, contains an
        unknown name, or requests a variable unavailable at ``cadence``.
    """

    by_name = {variable.name: variable for variable in registry}
    aliases = {alias: variable.name for variable in registry for alias in variable.aliases}

    if variables is None:
        requested = list(default_names)
    elif isinstance(variables, str):
        requested = (
            [variable.name for variable in registry if cadence is None or cadence in variable.cadences]
            if variables == "all"
            else [variables]
        )
    else:
        try:
            requested = list(variables)
        except TypeError as error:
            raise ValueError("variables must be a variable name or an iterable of variable names") from error

    if not requested:
        raise ValueError("variables must contain at least one OMNI variable name")
    if not all(isinstance(name, str) for name in requested):
        raise ValueError("OMNI variable names must be strings")

    resolved: list[str] = []
    invalid: list[str] = []
    unavailable: list[str] = []
    for name in requested:
        canonical_name = aliases.get(name, name)
        variable = by_name.get(canonical_name)
        if variable is None:
            invalid.append(name)
            continue
        if cadence is not None and cadence not in variable.cadences:
            unavailable.append(name)
            continue
        if canonical_name not in resolved:
            resolved.append(canonical_name)

    if invalid:
        available = ", ".join(variable.name for variable in registry if cadence is None or cadence in variable.cadences)
        raise ValueError(f"Unknown OMNI variables: {', '.join(invalid)}. Available variables: {available}")
    if unavailable:
        raise ValueError(f"OMNI variables unavailable at {cadence}-minute cadence: {', '.join(unavailable)}")

    return resolved
