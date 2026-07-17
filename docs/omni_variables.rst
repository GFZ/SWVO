.. SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
..
.. SPDX-License-Identifier: Apache-2.0

Complete OMNI variable access
=============================

SWVO provides two general-purpose OMNI readers:

* :class:`swvo.io.omni.OMNIHighRes` retrieves NASA OMNIWeb data at one- or
  five-minute cadence.
* :class:`swvo.io.omni.OMNILowRes` retrieves the fixed-width hourly OMNI2
  files published by NASA SPDF.

Both readers retain their original default columns. Complete data is opt-in
through the ``variables`` argument, so existing applications and the focused
``SWOMNI``, ``SymhOMNI``, ``KpOMNI``, ``DSTOMNI``, and ``F107OMNI`` readers
continue to receive their established schemas.

Selecting variables
-------------------

``None`` returns the legacy columns, ``"all"`` returns every variable valid
for the selected product and cadence, and a string or iterable selects a
subset. Subset order is preserved, duplicate names are removed, and aliases
are normalized to the canonical output name.

.. code-block:: python

   from datetime import datetime, timezone
   from pathlib import Path

   from swvo.io.omni import OMNIHighRes, OMNILowRes

   start = datetime(2024, 5, 1, tzinfo=timezone.utc)
   end = datetime(2024, 5, 2, tzinfo=timezone.utc)

   high_res = OMNIHighRes(data_dir=Path("./omni/high_res"))
   complete_1min = high_res.read(start, end, download=True, variables="all")
   selected_5min = high_res.read(
       start,
       end,
       cadence_min=5,
       download=True,
       variables=["speed", "bz_gsm", "proton_flux_10_mev"],
   )

   low_res = OMNILowRes(data_dir=Path("./omni/low_res"))
   hourly = low_res.read(start, end, download=True, variables=["kp", "dst", "ae"])

The complete machine-readable registries are available without downloading
data:

.. code-block:: python

   one_minute_metadata = OMNIHighRes.available_variables(cadence_min=1)
   five_minute_metadata = OMNIHighRes.available_variables(cadence_min=5)
   hourly_metadata = OMNILowRes.available_variables()

High-resolution variables
-------------------------

The one-minute product contains the first 42 rows below. The three integral
proton fluxes are only available in the five-minute product. Fill values and
request identifiers follow the `NASA high-resolution OMNI interface
<https://omniweb.gsfc.nasa.gov/form/omni_min.html>`_.

.. csv-table:: High-resolution OMNI registry
   :header: "Canonical name", "NASA ID", "Cadence (min)", "Unit", "Fill value", "Description", "Aliases"
   :widths: 22, 7, 10, 13, 11, 30, 14

   "imf_spacecraft_id", "4", "1, 5", "", "99", "IMF source spacecraft identifier", ""
   "plasma_spacecraft_id", "5", "1, 5", "", "99", "Solar-wind plasma source spacecraft identifier", ""
   "imf_point_count", "6", "1, 5", "count", "999", "Fine-scale points in the IMF average", ""
   "plasma_point_count", "7", "1, 5", "count", "999", "Fine-scale points in the plasma average", ""
   "percent_interpolated", "8", "1, 5", "%", "999", "Percentage of interpolated values", ""
   "timeshift", "9", "1, 5", "s", "999999", "Bow-shock-nose time shift", "timeshift_seconds"
   "timeshift_rms", "10", "1, 5", "s", "999999", "RMS time shift", "timeshift_rms_seconds"
   "minimum_variance_rms", "11", "1, 5", "", "99.99", "RMS minimum-variance vector", ""
   "observation_gap", "12", "1, 5", "s", "999999", "Time between observations", "observation_gap_seconds"
   "bavg", "13", "1, 5", "nT", "9999.99", "Average IMF magnitude", ""
   "bx_gsm", "14", "1, 5", "nT", "9999.99", "IMF Bx in GSE/GSM coordinates", "bx_gse"
   "by_gse", "15", "1, 5", "nT", "9999.99", "IMF By in GSE coordinates", ""
   "bz_gse", "16", "1, 5", "nT", "9999.99", "IMF Bz in GSE coordinates", ""
   "by_gsm", "17", "1, 5", "nT", "9999.99", "IMF By in GSM coordinates", ""
   "bz_gsm", "18", "1, 5", "nT", "9999.99", "IMF Bz in GSM coordinates", ""
   "sigma_bavg", "19", "1, 5", "nT", "9999.99", "RMS standard deviation of IMF magnitude", ""
   "sigma_b", "20", "1, 5", "nT", "9999.99", "RMS standard deviation of the IMF vector", ""
   "speed", "21", "1, 5", "km/s", "99999.9", "Solar-wind bulk speed", ""
   "vx_gse", "22", "1, 5", "km/s", "99999.9", "Solar-wind Vx in GSE coordinates", ""
   "vy_gse", "23", "1, 5", "km/s", "99999.9", "Solar-wind Vy in GSE coordinates", ""
   "vz_gse", "24", "1, 5", "km/s", "99999.9", "Solar-wind Vz in GSE coordinates", ""
   "proton_density", "25", "1, 5", "1/cm^3", "999.99", "Proton number density", ""
   "temperature", "26", "1, 5", "K", "9999999", "Proton temperature", ""
   "pdyn", "27", "1, 5", "nPa", "99.99", "Solar-wind flow pressure", "flow_pressure"
   "electric_field", "28", "1, 5", "mV/m", "999.99", "Convective electric field Ey", "ey"
   "plasma_beta", "29", "1, 5", "", "999.99", "Plasma beta", ""
   "alfven_mach_number", "30", "1, 5", "", "999.9", "Alfven Mach number", ""
   "spacecraft_x_gse", "31", "1, 5", "Re", "9999.99", "Spacecraft X position in GSE coordinates", ""
   "spacecraft_y_gse", "32", "1, 5", "Re", "9999.99", "Spacecraft Y position in GSE coordinates", ""
   "spacecraft_z_gse", "33", "1, 5", "Re", "9999.99", "Spacecraft Z position in GSE coordinates", ""
   "bsn_x_gse", "34", "1, 5", "Re", "9999.99", "Bow-shock-nose X position in GSE coordinates", ""
   "bsn_y_gse", "35", "1, 5", "Re", "9999.99", "Bow-shock-nose Y position in GSE coordinates", ""
   "bsn_z_gse", "36", "1, 5", "Re", "9999.99", "Bow-shock-nose Z position in GSE coordinates", ""
   "ae", "37", "1, 5", "nT", "99999", "Auroral electrojet AE index", ""
   "al", "38", "1, 5", "nT", "99999", "Auroral electrojet AL index", ""
   "au", "39", "1, 5", "nT", "99999", "Auroral electrojet AU index", ""
   "sym-d", "40", "1, 5", "nT", "99999", "SYM/D geomagnetic index", "sym_d"
   "sym-h", "41", "1, 5", "nT", "99999", "SYM/H geomagnetic index", "sym_h"
   "asy-d", "42", "1, 5", "nT", "99999", "ASY/D geomagnetic index", "asy_d"
   "asy-h", "43", "1, 5", "nT", "99999", "ASY/H geomagnetic index", "asy_h"
   "pcn", "44", "1, 5", "", "999.99", "Northern polar-cap index", "pc"
   "magnetosonic_mach_number", "45", "1, 5", "", "99.9", "Magnetosonic Mach number", ""
   "proton_flux_10_mev", "46", "5", "1/(cm^2 s sr)", "99999.99", "Integral proton flux above 10 MeV", ""
   "proton_flux_30_mev", "47", "5", "1/(cm^2 s sr)", "99999.99", "Integral proton flux above 30 MeV", ""
   "proton_flux_60_mev", "48", "5", "1/(cm^2 s sr)", "99999.99", "Integral proton flux above 60 MeV", ""

Hourly variables
----------------

The timestamp index replaces the raw year, day-of-year, and hour words. The
remaining 54 fields follow NASA's current 57-word `hourly OMNI2 record
description <https://omniweb.gsfc.nasa.gov/html/ow_data.html#3>`_. Historic
55-word records are also accepted; ``lyman_alpha`` and
``proton_quasi_invariant`` are returned as ``NaN`` for those records.

.. csv-table:: Hourly OMNI2 registry
   :header: "Canonical name", "Unit", "Fill value", "Description", "Aliases"
   :widths: 25, 16, 12, 37, 14

   "bartels_rotation_number", "", "9999", "Bartels rotation number", ""
   "imf_spacecraft_id", "", "99", "IMF source spacecraft identifier", ""
   "plasma_spacecraft_id", "", "99", "Solar-wind plasma source spacecraft identifier", ""
   "imf_point_count", "count", "999", "Fine-scale points in the IMF average", ""
   "plasma_point_count", "count", "999", "Fine-scale points in the plasma average", ""
   "bavg", "nT", "999.9", "Average IMF magnitude", ""
   "magnitude_average_field_vector", "nT", "999.9", "Magnitude of the average IMF vector", ""
   "latitude_average_field", "degree", "999.9", "Latitude angle of the average IMF vector", ""
   "longitude_average_field", "degree", "999.9", "Longitude angle of the average IMF vector", ""
   "bx_gse_gsm", "nT", "999.9", "IMF Bx in GSE/GSM coordinates", "bx_gse"
   "by_gse", "nT", "999.9", "IMF By in GSE coordinates", ""
   "bz_gse", "nT", "999.9", "IMF Bz in GSE coordinates", ""
   "by_gsm", "nT", "999.9", "IMF By in GSM coordinates", ""
   "bz_gsm", "nT", "999.9", "IMF Bz in GSM coordinates", ""
   "sigma_bavg", "nT", "999.9", "RMS standard deviation of IMF magnitude", ""
   "sigma_b", "nT", "999.9", "RMS standard deviation of the IMF vector", ""
   "sigma_bx", "nT", "999.9", "RMS standard deviation of IMF Bx", ""
   "sigma_by", "nT", "999.9", "RMS standard deviation of IMF By", ""
   "sigma_bz", "nT", "999.9", "RMS standard deviation of IMF Bz", ""
   "temperature", "K", "9999999", "Proton temperature", ""
   "proton_density", "1/cm^3", "999.9", "Proton number density", ""
   "speed", "km/s", "9999", "Solar-wind bulk speed", ""
   "speed_longitude", "degree", "999.9", "Solar-wind flow longitude in GSE coordinates", ""
   "speed_latitude", "degree", "999.9", "Solar-wind flow latitude in GSE coordinates", ""
   "alpha_proton_ratio", "", "9.999", "Alpha-to-proton density ratio", ""
   "pdyn", "nPa", "99.99", "Solar-wind flow pressure", "flow_pressure"
   "sigma_temperature", "K", "9999999", "RMS standard deviation of proton temperature", ""
   "sigma_proton_density", "1/cm^3", "999.9", "RMS standard deviation of proton density", ""
   "sigma_speed", "km/s", "9999", "RMS standard deviation of solar-wind speed", ""
   "sigma_speed_longitude", "degree", "999.9", "RMS standard deviation of flow longitude", ""
   "sigma_speed_latitude", "degree", "999.9", "RMS standard deviation of flow latitude", ""
   "sigma_alpha_proton_ratio", "", "9.999", "RMS standard deviation of alpha-to-proton ratio", ""
   "electric_field", "mV/m", "999.99", "Convective electric field", "e"
   "plasma_beta", "", "999.99", "Plasma beta", ""
   "alfven_mach_number", "", "999.9", "Alfven Mach number", ""
   "kp", "", "99", "Three-hour Kp index", ""
   "sunspot_number", "", "999", "Daily international sunspot number", ""
   "dst", "nT", "99999", "Dst geomagnetic index", ""
   "ae", "nT", "9999", "Auroral electrojet AE index", ""
   "proton_flux_1_mev", "1/(cm^2 s sr)", "999999.99", "Integral proton flux above 1 MeV", ""
   "proton_flux_2_mev", "1/(cm^2 s sr)", "99999.99", "Integral proton flux above 2 MeV", ""
   "proton_flux_4_mev", "1/(cm^2 s sr)", "99999.99", "Integral proton flux above 4 MeV", ""
   "proton_flux_10_mev", "1/(cm^2 s sr)", "99999.99", "Integral proton flux above 10 MeV", ""
   "proton_flux_30_mev", "1/(cm^2 s sr)", "99999.99", "Integral proton flux above 30 MeV", ""
   "proton_flux_60_mev", "1/(cm^2 s sr)", "99999.99", "Integral proton flux above 60 MeV", ""
   "magnetospheric_flux_flag", "", "", "Magnetospheric proton-flux contamination flag", ""
   "ap", "nT", "999", "Three-hour ap index", ""
   "f107", "10^-22 W/(m^2 Hz)", "999.9", "Daily F10.7 solar radio flux adjusted to 1 AU", ""
   "pcn", "", "999.9", "Northern polar-cap index", "pc"
   "al", "nT", "99999", "Auroral electrojet AL index", ""
   "au", "nT", "99999", "Auroral electrojet AU index", ""
   "magnetosonic_mach_number", "", "99.9", "Magnetosonic Mach number", ""
   "lyman_alpha", "W/m^2", "0.999999", "Daily solar Lyman-alpha irradiance", ""
   "proton_quasi_invariant", "", "9.9999", "Solar-wind proton quasi-invariant", ""

Cache behavior and compatibility
--------------------------------

Processed filenames and directory layouts are unchanged. A newly downloaded
period contains the complete schema for its product and cadence. A legacy
cache containing only the former default columns remains valid for default
reads.

When a selected variable is absent from an existing cache:

* ``download=True`` downloads and atomically replaces only the incomplete
  period with a complete processed file.
* ``download=False`` raises ``ValueError`` listing the missing columns and the
  command needed to upgrade the cache.

Temporary output is written beside the destination with a ``.tmp`` suffix and
is moved into place only after successful parsing. The hourly downloader uses
an operating-system temporary directory, so the workflow is safe on Linux and
macOS and does not depend on the current working directory.

Missing values and provenance
-----------------------------

NASA fill values are converted to ``NaN`` while processing. Kp retains SWVO's
existing conversion from OMNI's integer tenths to conventional thirds. The
``file_name`` column is not a NASA variable; it identifies the processed source
file and is null when every selected variable is missing for a row.

Errors are raised before network access for unknown names, empty selections,
non-string selections, invalid or equal time ranges, and one-minute requests
for the five-minute-only proton fluxes. Source records with missing headers,
unexpected word counts, nonnumeric fields, or invalid day/hour/minute values
are rejected instead of being silently shifted or partially interpreted.

An unreadable processed cache raises a path-specific ``ValueError`` when
``download=False``. With ``download=True`` SWVO attempts one atomic replacement;
if the replacement cannot be read, the error states that the cache remains
unreadable. OMNIWeb corrected-range responses are followed only when the
suggested end date shortens the request, preventing retry loops.

Data source and acknowledgement
-------------------------------

High-resolution records are retrieved from the `NASA GSFC/SPDF OMNIWeb
service <https://omniweb.gsfc.nasa.gov/ow_min.html>`_. Hourly records are
retrieved from the `NASA SPDF low-resolution OMNI archive
<https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/>`_. Publications using
these data should follow the acknowledgement and citation guidance in NASA's
OMNI documentation.
