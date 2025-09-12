# SWVO @ GFZ

[![PyPI version](https://badge.fury.io/py/swvo.svg)](https://badge.fury.io/py/swvo)
[![Build Sphinx HTML](https://app.readthedocs.org/projects/swvo/badge/?version=latest)](https://swvo.readthedocs.io/en/latest/)
[![SWVO Tests](https://github.com/GFZ/SWVO/actions/workflows/tests.yml/badge.svg)](https://github.com/GFZ/SWVO/actions/workflows/tests.yml)
[![Python version](https://img.shields.io/pypi/pyversions/swvo.svg)](https://badge.fury.io/py/swvo)
[![Coverage Status](https://coveralls.io/repos/github/GFZ/SWVO/badge.svg?branch=main)](https://coveralls.io/github/GFZ/SWVO?branch=main)


## Introduction
This package provides a set of tools for managing solar data in Python. It includes functionalities for reading, writing, and processing data from various sources.

## Solar Indices Overview

This package provides tools to read, process, and analyze several key solar and geomagnetic indices. For each index, the available data sources and the corresponding reader classes are listed below:

- **Kp Index**:  
  A global geomagnetic activity index with a 3-hour cadence, ranging from 0 (quiet) to 9 (extremely disturbed). Used to assess geomagnetic storm conditions.
  - **Sources & Classes:**
    - OMNI: `KpOMNI`
    - SWPC: `KpSWPC`
    - Niemegk: `KpNiemegk`
    - Ensemble: `KpEnsemble`
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

- **Solar Wind Parameters**:  
  Access to solar wind data (speed, density, magnetic field components) from various spacecraft. Essential for solar-terrestrial interaction studies.
  - **Sources & Classes:**
    - ACE: `SWACE`
    - DSCOVR: `DSCOVR`
    - OMNI: `SWOMNI`
    - SWIFT: `SWSWIFTEnsemble`
    - Combined: `read_solar_wind_from_multiple_models`

Each index can be accessed via these dedicated reader classes, which handle downloading and read methods. See the code in `swvo/io` or API documentation for details on each index's implementation.


## Installation
To install the package, run the following command:

`uv venv`
```bash
source .venv/bin/activate
python -m ensurepip --upgrade
uv pip install --upgrade pip
uv pip install -e .
```

or it can be installed directly from PyPI:
```bash
uv pip install swvo
```

All the above `uv` commands assume you have `uv` installed, if not then remove `uv` prefix from the commands and run them directly.