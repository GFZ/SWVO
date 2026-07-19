<!--
SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Matyas Szabo-Roberts
SPDX-FileContributor: Ruggero Vasile
SPDX-FileContributor: Sahil Jhawar
SPDX-FileContributor: Simon Mischel

SPDX-License-Identifier: Apache-2.0
-->

# SWVO @ GFZ

[![PyPI version](https://badge.fury.io/py/swvo.svg)](https://badge.fury.io/py/swvo)
[![Build Sphinx HTML](https://app.readthedocs.org/projects/swvo/badge/?version=latest)](https://swvo.readthedocs.io/en/latest/)
[![SWVO Tests](https://github.com/GFZ/SWVO/actions/workflows/tests.yml/badge.svg)](https://github.com/GFZ/SWVO/actions/workflows/tests.yml)
[![Python version](https://img.shields.io/pypi/pyversions/swvo.svg)](https://badge.fury.io/py/swvo)
[![Coverage Status](https://coveralls.io/repos/github/GFZ/SWVO/badge.svg?branch=main)](https://coveralls.io/github/GFZ/SWVO?branch=main)
[![REUSE status](https://api.reuse.software/badge/github.com/GFZ/SWVO)](https://api.reuse.software/info/github.com/GFZ/SWVO)

## Introduction

This package provides a set of tools for managing solar data in Python. It includes functionalities for reading, writing, and processing data from various sources.

## Example

```python
from datetime import datetime, timezone
from swvo.io.solar_wind import SWACE

ACE_DIR = "./ace_data/" #data directory for ACE data

start = datetime(2024, 11, 20, 0, 0, tzinfo=timezone.utc)
end = datetime(2024, 11, 20, 6, 0, tzinfo=timezone.utc)

#Read ACE solar wind data with downloading
ace_df = swace.read(start, end, download=True)
```

See here for a detailed example <a href="docs/examples/solar_wind_example.ipynb">docs/examples/solar_wind_example.ipynb</a>

## Space Weather Data Overview

This package provides tools to read, process, and analyze several key solar and geomagnetic indices. For each index, the available data sources and the corresponding reader classes are listed below:

- **Kp Index**:  
  A global geomagnetic activity index with a 3-hour cadence, ranging from 0 (quiet) to 9 (extremely disturbed). Used to assess geomagnetic storm conditions.
  - **Sources & Classes:**
    - OMNI: `KpOMNI`
    - SWPC: `KpSWPC`
    - Niemegk: `KpNiemegk`
    - Ensemble: `KpEnsemble`
    - BSG: `KpBGS`
    - SIDC: `KpSIDC`
    - Combined: `read_kp_from_multiple_models`

- **Dst Index**:  
  The Disturbance Storm Time (Dst) index measures the intensity of the Earth's ring current, related to geomagnetic storms. Provided hourly and is negative during storm conditions.
  - **Sources & Classes:**
    - OMNI: `DSTOMNI`
    - WDC: `DSTWDC`
    - Combined: `read_dst_from_multiple_models`

- **Hp Index**:  
  The Hp30 and Hp60 indices are high-cadence (30-minute and 60-minute) geomagnetic indices provided by GFZ, used for detailed geomagnetic activity studies.
  - **Sources & Classes:**
    - GFZ: `HpGFZ`
    - Ensemble: `HpEnsemble`
    - Combined: `read_hp_from_multiple_models`

- **F10.7 Index**:  
  The F10.7 solar radio flux index is a daily measure of solar activity (flux density at 10.7 cm), a standard proxy for solar EUV emissions.
  - **Sources & Classes:**
    - OMNI: `F107OMNI`
    - SWPC: `F107SWPC`
    - Combined: `read_f107_from_multiple_models`

- **SME Index**:  
  The SME (SuperMAG Electrojet) index measures auroral electrojet strength based on SuperMAG ground magnetometers. Requires a valid SuperMAG account.
  - **Sources & Classes:**
    - SuperMAG: `SMESuperMAG`

- **Solar Wind Parameters**:  
  Access to solar wind data (speed, density, magnetic field components) from various spacecraft. Essential for solar-terrestrial interaction studies.
  - **Sources & Classes:**
    - ACE: `SWACE`
    - DSCOVR: `DSCOVR`
    - OMNI: `SWOMNI`
    - SWIFT: `SWSWIFTEnsemble`
    - Combined: `read_solar_wind_from_multiple_models`

- **Plasmasphere Density Predictions**:  
  Reader utilities for PAGER plasmasphere density grids and model combined inputs.
  - **Sources & Classes:**
    - Density predictions: `PlasmaspherePredictionReader`
    - Combined inputs: `PlasmasphereCombinedInputsReader`
    - Density cube container: `PlasmasphereDensityCube`

Each index can be accessed via these dedicated reader classes, which handle downloading and read methods. See the code in `swvo/io` or API documentation for details on each index's implementation.

## Expanded OMNI data access

`OMNIHighRes` and `OMNILowRes` preserve their established default columns and
also support `variables="all"` or an explicit variable-name selection. The
high-resolution reader exposes all 42 one-minute and 45 five-minute OMNIWeb
fields; the hourly reader exposes all 54 non-time OMNI2 fields and accepts both
historic 55-word and current 57-word source records.

Variable metadata can be inspected without making a network request through a
reader instance's `available_variables()` method or the shared cadence-aware
`swvo.io.omni.variables.available_variables()` utility. See the
[complete OMNI variable guide](https://swvo.readthedocs.io/en/latest/omni_variables.html)
for examples, variable tables, cadence restrictions, cache-upgrade behavior,
compatibility guarantees, and NASA source acknowledgements.

## Installation

To install the package, run the following command:

`uv venv`

```bash
uv venv venv --python 3.11 --seed
source venv/bin/activate
uv pip install -e .
```

or it can be installed directly from PyPI:

```bash
uv pip install swvo
```

All the above `uv` commands assume you have `uv` installed, if not then remove `uv` prefix from the commands and run them directly.
