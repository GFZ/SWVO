# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
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
from swvo.io.RBMDataSet.utils import python2matlab


class MockVariable:
    """Create a mock Elpaso Variable class for testing"""

    def __init__(self, standard_name, data=None):
        self.standard_name = standard_name
        self.data = data


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
    assert isinstance(dataset.variable_mapping, dict)
    assert "FEDU" in dataset.variable_mapping
    assert dataset.variable_mapping["FEDU"] == "Flux"


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
    """Test that the correct variable is set with the standard name"""
    fedu_data = np.array([[1.0, 2.0]])

    source_dict = {"FEDU": MockVariable(standard_name="FEDU", data=fedu_data)}

    dataset.update_from_dict(source_dict)
    np.testing.assert_array_equal(dataset.Flux, fedu_data)


def test_update_from_dict_sets_time(dataset):
    """Test that the correct variable is set with the time standard name"""
    ts = [datetime(2025, 4, 1, tzinfo=timezone.utc).timestamp()]

    source_dict = {"Epoch": MockVariable(standard_name="Epoch_posixtime", data=ts)}

    dataset.update_from_dict(source_dict)
    assert hasattr(dataset, "time")
    assert hasattr(dataset, "datetime")
    assert dataset.time[0] == python2matlab(datetime(2025, 4, 1, tzinfo=timezone.utc))


def test_update_from_dict_with_mfm_suffix(dataset):
    """Test that the correct variable is set with the MFM suffix"""
    mfm_suffix = "_" + dataset._mfm_prefix
    lstar_data = np.array([4.5, 5.0, 5.5])

    source_dict = {"Lstar": MockVariable(standard_name=f"Lstar{mfm_suffix}", data=lstar_data)}

    dataset.update_from_dict(source_dict)
    np.testing.assert_array_equal(dataset.Lstar, lstar_data)


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
    with pytest.raises(AttributeError, match="mapped but has not been set"):
        _ = dataset.Flux

    with pytest.raises(AttributeError, match="no attribute"):
        _ = dataset.NonExistent


def test_dir_contains_variable_names(dataset):
    variable_names = [var.var_name for var in VariableEnum]
    for name in variable_names:
        assert name in dir(dataset)


def test_all_variable_mappings(dataset):
    """Test that all variable mappings work correctly."""

    expected_mappings = {
        "Epoch_posixtime": "time",
        "Energy_FEDU": "energy_channels",
        "PA_local": "alpha_local",
        "PA_eq_": "alpha_eq_model",
        "alpha_eq_real": "alpha_eq_real",
        "invMu_": "InvMu",
        "InvMu_real": "InvMu_real",
        "invK_": "InvK",
        "Lstar_": "Lstar",
        "FEDU": "Flux",
        "PSD_FEDU": "PSD",
        "MLT_": "MLT",
        "B_SM": "B_SM",
        "B_eq_": "B_total",
        "B_local_": "B_sat",
        "xGEO": "xGEO",
        "R_eq_": "R0",
        "density": "density",
    }

    for source, target in expected_mappings.items():
        assert source in dataset.variable_mapping
        assert dataset.variable_mapping[source] == target

    test_data = {}
    for source, target in expected_mappings.items():
        if source == "Epoch_posixtime":
            data = [datetime(2025, 4, 1, tzinfo=timezone.utc).timestamp()]
        else:
            data = np.array([float(hash(source) % 1000) / 10.0])

        test_data[source] = MockVariable(standard_name=source, data=data)

    dataset.update_from_dict(test_data)

    for source, target in expected_mappings.items():
        if source == "Epoch_posixtime":
            assert hasattr(dataset, "time")
            assert hasattr(dataset, "datetime")
            assert isinstance(dataset.datetime[0], datetime)
            assert dataset.time[0] == python2matlab(dataset.datetime[0])
        elif source in ["P", "InvV"]:
            pass
        else:
            assert hasattr(dataset, target), f"Attribute {target} not set from {source}"

            np.testing.assert_array_equal(
                dataset.__getattribute__(target),
                test_data[source].data,
                err_msg=f"Data mismatch for {target} from {source}",
            )


def test_all_mfm_specific_mappings(dataset):
    """Test that all MFM-specific variable mappings work correctly."""

    mfm_variables = [
        ("PA_eq", "alpha_eq_model"),
        ("invMu", "InvMu"),
        ("invK", "InvK"),
        ("Lstar", "Lstar"),
        ("B_eq", "B_total"),
        ("B_local", "B_sat"),
        ("R_eq", "R0"),
    ]

    mfm_suffix = "_" + dataset._mfm_prefix

    test_data = {}
    for source_base, target in mfm_variables:
        source = f"{source_base}{mfm_suffix}"

        data = np.array([float(hash(source) % 1000) / 10.0])
        test_data[source] = MockVariable(standard_name=source, data=data)

    dataset.update_from_dict(test_data)

    for source_base, target in mfm_variables:
        source = f"{source_base}{mfm_suffix}"
        assert hasattr(dataset, target), f"Attribute {target} not set from {source}"
        np.testing.assert_array_equal(
            dataset.__getattribute__(target),
            test_data[source].data,
            err_msg=f"Data mismatch for {target} from {source}",
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
