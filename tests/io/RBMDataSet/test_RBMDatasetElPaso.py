# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from typing import get_args

import numpy as np
import pytest

from swvo.io.RBMDataSet import (
    InstrumentEnum,
    MfmEnum,
    RBMDataSet,
    SatelliteEnum,
    SatelliteLiteral,
    VariableEnum,
)


@pytest.fixture
def dataset():
    return RBMDataSet(
        satellite=SatelliteEnum.GOESSecondary,
        instrument=InstrumentEnum.MAGED,
        mfm=MfmEnum.T89,
    )


def test_init_accepts_string_inputs():
    """Test that the class can be initialized with string inputs."""
    ds = RBMDataSet(satellite="GOESSecondary", instrument="MAGED", mfm=MfmEnum.T89)
    assert ds.satellite.sat_name == "secondary"
    assert ds.instrument == InstrumentEnum.MAGED
    assert ds.mfm == MfmEnum.T89


def test_variable_mapping_exposed(dataset):
    """Test that ep_variables contains the expected variable names"""
    assert isinstance(dataset.ep_variables, list)
    assert "Flux" in dataset.ep_variables
    assert "energy_channels" in dataset.ep_variables


def test_repr_and_str(dataset):
    assert "RBMDataSet" in repr(dataset)
    assert str(dataset.satellite) in repr(dataset)
    assert str(dataset.instrument) in repr(dataset)
    assert str(dataset.mfm) in repr(dataset)

    assert "RBMDataSet" in str(dataset)
    assert str(dataset.satellite) in str(dataset)
    assert str(dataset.instrument) in str(dataset)
    assert str(dataset.mfm) in str(dataset)


def test_update_from_dict_sets_variables(dataset):
    """Test that the correct variable is set with direct key"""
    flux_data = np.array([[1.0, 2.0]])

    source_dict = {"Flux": flux_data}

    dataset.update_from_dict(source_dict)
    np.testing.assert_array_equal(dataset.Flux, flux_data)


def test_update_from_dict_sets_time(dataset):
    """Test that the correct variable is set with direct time key"""
    time_data = np.array([738000.0])  # MATLAB datenum format

    source_dict = {"time": time_data}

    dataset.update_from_dict(source_dict)
    assert hasattr(dataset, "time")
    np.testing.assert_array_equal(dataset.time, time_data)


def test_update_from_dict_with_multiple_variables(dataset):
    """Test that multiple variables can be set at once"""
    lstar_data = np.array([4.5, 5.0, 5.5])
    energy_data = np.array([100.0, 200.0, 300.0])

    source_dict = {"Lstar": lstar_data, "energy_channels": energy_data}

    dataset.update_from_dict(source_dict)
    np.testing.assert_array_equal(dataset.Lstar, lstar_data)
    np.testing.assert_array_equal(dataset.energy_channels, energy_data)


def test_computed_p_property(dataset):
    """Test P property with correct dimensions"""
    dataset.MLT = np.array([0.0, 6.0, 12.0])

    expected_p = ((dataset.MLT + 12) / 12 * np.pi) % (2 * np.pi)
    np.testing.assert_allclose(dataset.P, expected_p)


def test_computed_invv_property(dataset):
    """Test InvV property with correct dimensions"""
    dataset.InvMu = np.array([[0.1, 0.2]])
    dataset.InvK = np.array([[1.0]])

    inv_K_repeated = np.repeat(dataset.InvK[:, np.newaxis, :], dataset.InvMu.shape[1], axis=1)
    expected_invv = dataset.InvMu * (inv_K_repeated + 0.5) ** 2

    np.testing.assert_allclose(dataset.InvV, expected_invv)


def test_getattr_errors(dataset):
    with pytest.raises(AttributeError, match="exists in `VariableLiteral` but has not been set"):
        _ = dataset.Flux

    with pytest.raises(AttributeError, match="no attribute"):
        _ = dataset.NonExistent


def test_dir_contains_variable_names(dataset):
    variable_names = [var.var_name for var in VariableEnum]
    for name in variable_names:
        assert name in dir(dataset)


def test_update_from_dict_invalid_key(dataset):
    """Test that invalid keys raise VariableNotFoundError"""
    from swvo.io.exceptions import VariableNotFoundError

    source_dict = {"InvalidKey": np.array([1.0, 2.0])}

    with pytest.raises(VariableNotFoundError, match="not a valid `VariableLiteral`"):
        dataset.update_from_dict(source_dict)


def test_update_from_dict_similar_key(dataset):
    """Test that similar keys suggest the correct variable"""
    from swvo.io.exceptions import VariableNotFoundError

    source_dict = {"Flx": np.array([1.0, 2.0])}  # Typo: should be "Flux"

    with pytest.raises(VariableNotFoundError, match="Maybe you meant 'Flux'"):
        dataset.update_from_dict(source_dict)


def test_all_variable_literals(dataset):
    """Test that all VariableLiteral values can be set and retrieved."""

    # Test with a subset of common variables
    test_variables = {
        "time": np.array([738000.0, 738001.0]),
        "energy_channels": np.array([100.0, 200.0, 300.0]),
        "alpha_local": np.array([0.1, 0.2, 0.3]),
        "alpha_eq_model": np.array([45.0, 60.0, 90.0]),
        "InvMu": np.array([[0.1, 0.2]]),
        "InvK": np.array([[1.0, 2.0]]),
        "Lstar": np.array([4.5, 5.0, 5.5]),
        "Flux": np.array([[1.0, 2.0, 3.0]]),
        "PSD": np.array([[0.1, 0.2, 0.3]]),
        "MLT": np.array([0.0, 6.0, 12.0]),
        "B_SM": np.array([100.0, 200.0, 300.0]),
        "B_total": np.array([50.0, 60.0, 70.0]),
        "B_sat": np.array([45.0, 55.0, 65.0]),
        "xGEO": np.array([6.6, 6.7, 6.8]),
        "R0": np.array([5.0, 5.5, 6.0]),
        "density": np.array([100.0, 200.0, 300.0]),
    }

    dataset.update_from_dict(test_variables)

    for var_name, expected_data in test_variables.items():
        assert hasattr(dataset, var_name), f"Attribute {var_name} not set"
        np.testing.assert_array_equal(
            getattr(dataset, var_name),
            expected_data,
            err_msg=f"Data mismatch for {var_name}",
        )


@pytest.mark.parametrize("satellite, expected", [("goessecondary", "secondary"), ("goesprimary", "primary")])
def test_goes_lowercase(satellite, expected):
    goes_dataset = RBMDataSet(
        satellite=satellite,
        instrument=InstrumentEnum.MAGED,
        mfm=MfmEnum.T89,
    )
    assert goes_dataset.satellite.sat_name == expected


uppercase_satellites = set(get_args(SatelliteLiteral)) - set(["GOESPrimary", "GOESSecondary"])


@pytest.mark.parametrize("satellite", [i.lower() for i in uppercase_satellites])
def test_satellite_lowercase(satellite):
    dataset = RBMDataSet(
        satellite=satellite,
        instrument=InstrumentEnum.MAGED,
        mfm=MfmEnum.T89,
    )
    assert dataset.satellite.sat_name == satellite.lower()
