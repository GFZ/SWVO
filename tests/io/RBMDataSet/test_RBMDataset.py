# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
from datetime import timezone
from pathlib import Path
from typing import get_args
from unittest import mock

import numpy as np
import pytest

from swvo.io.RBMDataSet import (
    FileCadenceEnum,
    InstrumentEnum,
    MfmEnum,
    RBMDataSet,
    SatelliteEnum,
    SatelliteLiteral,
    VariableEnum,
)


@pytest.fixture
def mock_module_string():
    return "swvo.io.RBMDataSet.RBMDataSet.RBMDataSet"


@pytest.fixture
def mock_dataset():
    start_time = dt.datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_time = dt.datetime(2023, 1, 31, tzinfo=timezone.utc)

    with mock.patch("swvo.io.RBMDataSet.utils.get_file_path_any_format") as mock_get_path:
        with mock.patch("swvo.io.RBMDataSet.utils.load_file_any_format") as mock_load_file:
            mock_get_path.return_value = Path("/mock/path/file.pickle")
            mock_load_file.return_value = {
                "time": np.array([dt.datetime(2023, 1, 15).timestamp()]),
                "datetime": np.array([dt.datetime(2023, 1, 15, tzinfo=timezone.utc)]),
                "energy_channels": np.array([100, 200, 300]),
                "alpha_local": np.array([0.1, 0.2, 0.3]),
                "Flux": np.array([[1.0, 2.0, 3.0]]),
            }

            dataset = RBMDataSet(
                satellite=SatelliteEnum.RBSPA,
                instrument=InstrumentEnum.MAGEIS,
                mfm=MfmEnum.T89,
                start_time=start_time,
                end_time=end_time,
                folder_path=Path("/mock/path"),
                preferred_extension="pickle",
                verbose=False,
            )

            return dataset


def test_init_datetime_timezone(mock_module_string):
    """Test timezone handling for input datetimes."""

    start_time = dt.datetime(2023, 1, 1)
    end_time = dt.datetime(2023, 1, 31)

    with (
        mock.patch(f"{mock_module_string}._create_date_list"),
        mock.patch(f"{mock_module_string}._create_file_path_stem"),
        mock.patch(f"{mock_module_string}._create_file_name_stem"),
    ):
        dataset = RBMDataSet(
            satellite=SatelliteEnum.RBSPA,
            instrument=InstrumentEnum.MAGEIS,
            mfm=MfmEnum.T89,
            start_time=start_time,
            end_time=end_time,
            folder_path=Path("/mock/path"),
            preferred_extension="pickle",
        )

        assert dataset._start_time.tzinfo == timezone.utc
        assert dataset._end_time.tzinfo == timezone.utc


def test_get_satellite_name(mock_dataset: RBMDataSet):
    """Test get_satellite_name method."""
    assert mock_dataset.get_satellite_name() == "rbspa"


def test_get_satellite_and_instrument_name(mock_dataset: RBMDataSet):
    """Test get_satellite_and_instrument_name method."""
    assert mock_dataset.get_satellite_and_instrument_name() == "rbspa_mageis"


def test_get_print_name(mock_dataset: RBMDataSet):
    """Test get_print_name method."""
    assert mock_dataset.get_print_name() == "rbspa mageis"


def test_satellite_string_input(mock_module_string):
    """Test that satellite can be provided as string."""
    with mock.patch(f"{mock_module_string}._create_date_list"):
        with mock.patch(f"{mock_module_string}._create_file_path_stem"):
            with mock.patch(f"{mock_module_string}._create_file_name_stem"):
                dataset = RBMDataSet(
                    satellite="RBSPA",
                    instrument=InstrumentEnum.MAGEIS,
                    mfm=MfmEnum.T89,
                    start_time=dt.datetime(2023, 1, 1, tzinfo=timezone.utc),
                    end_time=dt.datetime(2023, 1, 31, tzinfo=timezone.utc),
                    folder_path=Path("/mock/path"),
                    preferred_extension="pickle",
                )

                assert dataset._satellite == SatelliteEnum.RBSPA


def test_getattr_with_valid_variable(mock_dataset: RBMDataSet):
    """Test __getattr__ with a valid variable."""
    with mock.patch.object(mock_dataset, "_load_variable") as _:
        mock_dataset.Flux = np.array([1.0, 2.0, 3.0])
        result = mock_dataset.Flux
        assert isinstance(result, np.ndarray)
        assert (result == np.array([1.0, 2.0, 3.0])).all()


def test_getattr_with_invalid_variable(mock_dataset: RBMDataSet):
    """Test __getattr__ with an invalid variable."""
    with pytest.raises(AttributeError):
        _ = mock_dataset.NonExistentAttribute


def test_getattr_with_similar_variable(mock_dataset: RBMDataSet):
    """Test __getattr__ suggests similar variable name."""
    with pytest.raises(AttributeError) as e:
        _ = mock_dataset.Flx

    assert "Maybe you meant Flux?" in str(e.value)


def test_computed_invv_variable(mock_dataset: RBMDataSet):
    """Test computed InvV variable."""

    mock_dataset.InvK = np.array([[1.0, 2.0]])
    mock_dataset.InvMu = np.array([[0.1, 0.2], [0.3, 0.4]])

    mock_dataset._load_variable(VariableEnum.INV_V)

    expected = (
        mock_dataset.InvMu
        * (np.repeat(mock_dataset.InvK[:, np.newaxis, :], mock_dataset.InvMu.shape[1], axis=1) + 0.5) ** 2
    )
    np.testing.assert_array_equal(mock_dataset.InvV, expected)


def test_computed_p_variable(mock_dataset: RBMDataSet):
    """Test computed P variable."""

    mock_dataset.MLT = np.array([0.0, 6.0, 12.0, 18.0])

    mock_dataset._load_variable(VariableEnum.P)

    expected = ((mock_dataset.MLT + 12) / 12 * np.pi) % (2 * np.pi)
    np.testing.assert_array_equal(mock_dataset.P, expected)


@pytest.mark.parametrize("satellite", list(SatelliteEnum))
def test_all_satellites_work(satellite, mock_module_string):
    """Ensure all SatelliteEnum values initialize without error."""
    with mock.patch(f"{mock_module_string}._create_date_list"):
        with mock.patch(f"{mock_module_string}._create_file_path_stem"):
            with mock.patch(f"{mock_module_string}._create_file_name_stem"):
                dataset = RBMDataSet(
                    satellite=satellite,
                    instrument=InstrumentEnum.HOPE,
                    mfm=MfmEnum.T89,
                    start_time=dt.datetime(2023, 1, 1, tzinfo=timezone.utc),
                    end_time=dt.datetime(2023, 1, 31, tzinfo=timezone.utc),
                    folder_path=Path("/mock/path"),
                )
                assert dataset._satellite == satellite


@pytest.mark.parametrize("instrument", list(InstrumentEnum))
def test_all_instruments_work(instrument, mock_module_string):
    """Ensure all InstrumentEnum values initialize without error."""
    with mock.patch(f"{mock_module_string}._create_date_list"):
        with mock.patch(f"{mock_module_string}._create_file_path_stem"):
            with mock.patch(f"{mock_module_string}._create_file_name_stem"):
                dataset = RBMDataSet(
                    satellite=SatelliteEnum.RBSPA,
                    instrument=instrument,
                    mfm=MfmEnum.T89,
                    start_time=dt.datetime(2023, 1, 1, tzinfo=timezone.utc),
                    end_time=dt.datetime(2023, 1, 31, tzinfo=timezone.utc),
                    folder_path=Path("/mock/path"),
                )
                assert dataset._instrument == instrument


def test_create_date_list_monthly(mock_dataset: RBMDataSet):
    """Test monthly cadence date generation."""
    mock_dataset.set_file_cadence(FileCadenceEnum.Monthly)
    date_list = mock_dataset._create_date_list()
    assert date_list[0].month == 1
    assert all(date.tzinfo == timezone.utc for date in date_list)


def test_create_date_list_daily(mock_dataset: RBMDataSet):
    """Test daily cadence date generation."""
    mock_dataset.set_file_cadence(FileCadenceEnum.Daily)
    date_list = mock_dataset._create_date_list()
    assert len(date_list) > 20
    assert all(date.tzinfo == timezone.utc for date in date_list)


def test_file_name_stem_generation(mock_dataset: RBMDataSet):
    """Test that file name stem is generated correctly."""
    assert mock_dataset._create_file_name_stem() == "rbspa_mageis_"


def test_file_path_stem_dataserver(mock_dataset: RBMDataSet):
    """Test correct file path stem for DataServer folder type."""
    expected_path = Path("/mock/path/RBSP/rbspa/Processed_Mat_Files")
    assert mock_dataset._create_file_path_stem() == expected_path


def test_invalid_cadence_raises(mock_dataset: RBMDataSet):
    """Invalid cadence should raise ValueError."""
    mock_dataset._file_cadence = None
    with pytest.raises(ValueError):
        mock_dataset._create_date_list()


def test_invalid_folder_type_raises(mock_dataset: RBMDataSet):
    """Invalid folder type should raise ValueError."""
    mock_dataset._folder_type = None
    with pytest.raises(ValueError):
        mock_dataset._create_file_path_stem()


def test_get_var_method(mock_dataset: RBMDataSet):
    """Test get_var returns correct variable."""
    mock_dataset.Flux = np.array([4.0, 5.0])
    result = mock_dataset.get_var(VariableEnum.FLUX)
    assert isinstance(result, np.ndarray)
    assert (result == np.array([4.0, 5.0])).all()


def test_load_variable_real_file():
    start_time = dt.datetime(2025, 4, 1, tzinfo=dt.timezone.utc)
    end_time = dt.datetime(2025, 4, 30, tzinfo=dt.timezone.utc)

    dataset = RBMDataSet(
        satellite=SatelliteEnum.GOESSecondary,
        instrument=InstrumentEnum.MAGED,
        mfm=MfmEnum.T89,
        start_time=start_time,
        end_time=end_time,
        folder_path=Path("path/to/real/files"),  # this does not matter for the test
        preferred_extension="pickle",
        verbose=True,
    )

    dataset._load_variable(VariableEnum.ALPHA_LOCAL)

    assert hasattr(dataset, "alpha_local"), "Dataset should have 'alpha_local' attribute after loading."
    assert isinstance(dataset.alpha_local, np.ndarray), "'alpha_local' should be a NumPy array."


def test_all_variables_in_dir(mock_dataset: RBMDataSet):
    vars = [
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

    for var in vars:
        assert var in mock_dataset.__dir__()


def test_disable_dict_loading_mode(mock_dataset: RBMDataSet):
    """Test that default file loading mode raises RuntimeError on update_from_dict."""
    with pytest.raises(RuntimeError):
        mock_dataset.update_from_dict({"Flux": np.array([[1.0, 2.0, 3.0]])})


def test_enable_dict_loading_mode(mock_dataset: RBMDataSet):
    """Test that default file loading mode raises RuntimeError on update_from_dict."""
    mock_dataset._enable_dict_loading = True
    mock_dataset.update_from_dict({"Flux": np.array([[1.0, 2.0, 3.0]])})

    assert hasattr(mock_dataset, "Flux")


@pytest.fixture
def dict_dataset():
    """Fixture for dictionary-based loading mode (no file parameters)"""
    return RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )


def test_dict_mode_init_accepts_string_inputs():
    """Test that the class can be initialized with string inputs in dict mode."""
    ds = RBMDataSet(satellite="GOESSecondary", instrument="MAGED", mfm=MfmEnum.T89)
    assert ds.satellite.sat_name == "secondary"
    assert ds.instrument == InstrumentEnum.MAGED
    assert ds.mfm == MfmEnum.T89


def test_dict_mode_repr_and_str(dict_dataset):
    """Test repr and str for dict mode"""
    assert "RBMDataSet" in repr(dict_dataset)
    assert str(dict_dataset.satellite) in repr(dict_dataset)
    assert str(dict_dataset.instrument) in repr(dict_dataset)
    assert str(dict_dataset.mfm) in repr(dict_dataset)

    assert "RBMDataSet" in str(dict_dataset)
    assert str(dict_dataset.satellite) in str(dict_dataset)
    assert str(dict_dataset.instrument) in str(dict_dataset)
    assert str(dict_dataset.mfm) in str(dict_dataset)


def test_update_from_dict_returns_self(dict_dataset):
    """Test that update_from_dict returns self for method chaining"""
    source_dict = {"Flux": np.array([[1.0, 2.0]])}

    result = dict_dataset.update_from_dict(source_dict)
    assert isinstance(result, RBMDataSet)


def test_update_from_dict_sets_variables(dict_dataset):
    """Test that the correct variable is set with direct key"""
    flux_data = np.array([[1.0, 2.0]])

    source_dict = {"Flux": flux_data}

    dict_dataset.update_from_dict(source_dict)
    np.testing.assert_array_equal(dict_dataset.Flux, flux_data)


def test_update_from_dict_sets_time(dict_dataset):
    """Test that the correct variable is set with direct time key"""
    time_data = np.array([738000.0])

    source_dict = {"time": time_data}

    dict_dataset.update_from_dict(source_dict)
    assert hasattr(dict_dataset, "time")
    np.testing.assert_array_equal(dict_dataset.time, time_data)


def test_update_from_dict_with_multiple_variables(dict_dataset):
    """Test that multiple variables can be set at once"""
    lstar_data = np.array([4.5, 5.0, 5.5])
    energy_data = np.array([100.0, 200.0, 300.0])

    source_dict = {"Lstar": lstar_data, "energy_channels": energy_data}

    dict_dataset.update_from_dict(source_dict)
    np.testing.assert_array_equal(dict_dataset.Lstar, lstar_data)
    np.testing.assert_array_equal(dict_dataset.energy_channels, energy_data)


def test_dict_mode_computed_p_property(dict_dataset):
    """Test P property with correct dimensions in dict mode"""
    dict_dataset.MLT = np.array([0.0, 6.0, 12.0])

    expected_p = ((dict_dataset.MLT + 12) / 12 * np.pi) % (2 * np.pi)
    np.testing.assert_allclose(dict_dataset.P, expected_p)


def test_dict_mode_computed_invv_property(dict_dataset):
    """Test InvV property with correct dimensions in dict mode"""
    dict_dataset.InvMu = np.array([[0.1, 0.2]])
    dict_dataset.InvK = np.array([[1.0]])

    inv_K_repeated = np.repeat(dict_dataset.InvK[:, np.newaxis, :], dict_dataset.InvMu.shape[1], axis=1)
    expected_invv = dict_dataset.InvMu * (inv_K_repeated + 0.5) ** 2

    np.testing.assert_allclose(dict_dataset.InvV, expected_invv)


def test_dict_mode_getattr_errors(dict_dataset):
    """Test error handling for unset attributes in dict mode"""
    with pytest.raises(AttributeError, match="exists in `VariableLiteral` but has not been set"):
        _ = dict_dataset.Flux

    with pytest.raises(AttributeError, match="no attribute"):
        _ = dict_dataset.NonExistent


def test_dict_mode_dir_contains_variable_names(dict_dataset):
    """Test that dir() includes variable names in dict mode"""
    variable_names = [var.var_name for var in VariableEnum]
    for name in variable_names:
        assert name in dir(dict_dataset)


def test_update_from_dict_invalid_key(dict_dataset):
    """Test that invalid keys raise VariableNotFoundError"""
    from swvo.io.exceptions import VariableNotFoundError

    source_dict = {"InvalidKey": np.array([1.0, 2.0])}

    with pytest.raises(VariableNotFoundError, match="not a valid `VariableLiteral`"):
        dict_dataset.update_from_dict(source_dict)


def test_update_from_dict_similar_key(dict_dataset):
    """Test that similar keys suggest the correct variable"""
    from swvo.io.exceptions import VariableNotFoundError

    source_dict = {"Flx": np.array([1.0, 2.0])}  # Typo: should be "Flux"

    with pytest.raises(VariableNotFoundError, match="Maybe you meant 'Flux'"):
        dict_dataset.update_from_dict(source_dict)


def test_all_variable_literals(dict_dataset):
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

    dict_dataset.update_from_dict(test_variables)

    for var_name, expected_data in test_variables.items():
        assert hasattr(dict_dataset, var_name), f"Attribute {var_name} not set"
        np.testing.assert_array_equal(
            getattr(dict_dataset, var_name),
            expected_data,
            err_msg=f"Data mismatch for {var_name}",
        )


@pytest.mark.parametrize("satellite, expected", [("goessecondary", "secondary"), ("goesprimary", "primary")])
def test_dict_mode_goes_lowercase(satellite, expected):
    """Test GOES satellite lowercase string handling in dict mode"""
    goes_dataset = RBMDataSet(
        satellite=satellite,
        instrument=InstrumentEnum.MAGED,
        mfm=MfmEnum.T89,
    )
    assert goes_dataset.satellite.sat_name == expected


uppercase_satellites = set(get_args(SatelliteLiteral)) - set(["GOESPrimary", "GOESSecondary"])


@pytest.mark.parametrize("satellite", [i.lower() for i in uppercase_satellites])
def test_dict_mode_satellite_lowercase(satellite):
    """Test satellite lowercase string handling in dict mode"""
    dataset = RBMDataSet(
        satellite=satellite,
        instrument=InstrumentEnum.MAGED,
        mfm=MfmEnum.T89,
    )
    assert dataset.satellite.sat_name == satellite.lower()


def test_eq_file_loading_mode_identical(mock_module_string):
    """Test equality for identical file loading mode datasets."""
    start_time = dt.datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_time = dt.datetime(2023, 1, 31, tzinfo=timezone.utc)
    folder_path = Path("/mock/path")

    with mock.patch(f"{mock_module_string}._create_date_list"):
        with mock.patch(f"{mock_module_string}._create_file_path_stem"):
            with mock.patch(f"{mock_module_string}._create_file_name_stem"):
                dataset1 = RBMDataSet(
                    satellite=SatelliteEnum.RBSPA,
                    instrument=InstrumentEnum.MAGEIS,
                    mfm=MfmEnum.T89,
                    start_time=start_time,
                    end_time=end_time,
                    folder_path=folder_path,
                    verbose=False,
                )

                dataset2 = RBMDataSet(
                    satellite=SatelliteEnum.RBSPA,
                    instrument=InstrumentEnum.MAGEIS,
                    mfm=MfmEnum.T89,
                    start_time=start_time,
                    end_time=end_time,
                    folder_path=folder_path,
                    verbose=False,
                )

                dataset1.Flux = np.array([[1.0, 2.0, 3.0]])
                dataset1.time = np.array([738000.0])
                dataset1.datetime = [dt.datetime(2023, 1, 15, tzinfo=timezone.utc)]

                dataset2.Flux = np.array([[1.0, 2.0, 3.0]])
                dataset2.time = np.array([738000.0])
                dataset2.datetime = [dt.datetime(2023, 1, 15, tzinfo=timezone.utc)]

                assert dataset1 == dataset2


def test_eq_file_loading_mode_different_satellite(mock_module_string):
    """Test inequality for file loading mode datasets with different satellites."""
    start_time = dt.datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_time = dt.datetime(2023, 1, 31, tzinfo=timezone.utc)
    folder_path = Path("/mock/path")

    with mock.patch(f"{mock_module_string}._create_date_list"):
        with mock.patch(f"{mock_module_string}._create_file_path_stem"):
            with mock.patch(f"{mock_module_string}._create_file_name_stem"):
                dataset1 = RBMDataSet(
                    satellite=SatelliteEnum.RBSPA,
                    instrument=InstrumentEnum.MAGEIS,
                    mfm=MfmEnum.T89,
                    start_time=start_time,
                    end_time=end_time,
                    folder_path=folder_path,
                    verbose=False,
                )

                dataset2 = RBMDataSet(
                    satellite=SatelliteEnum.RBSPB,  # Different satellite
                    instrument=InstrumentEnum.MAGEIS,
                    mfm=MfmEnum.T89,
                    start_time=start_time,
                    end_time=end_time,
                    folder_path=folder_path,
                    verbose=False,
                )

                print("sahil")

                assert dataset1 != dataset2


def test_eq_file_loading_mode_different_instrument(mock_module_string):
    """Test inequality for file loading mode datasets with different instruments."""
    start_time = dt.datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_time = dt.datetime(2023, 1, 31, tzinfo=timezone.utc)
    folder_path = Path("/mock/path")

    with mock.patch(f"{mock_module_string}._create_date_list"):
        with mock.patch(f"{mock_module_string}._create_file_path_stem"):
            with mock.patch(f"{mock_module_string}._create_file_name_stem"):
                dataset1 = RBMDataSet(
                    satellite=SatelliteEnum.RBSPA,
                    instrument=InstrumentEnum.MAGEIS,
                    mfm=MfmEnum.T89,
                    start_time=start_time,
                    end_time=end_time,
                    folder_path=folder_path,
                    verbose=False,
                )

                dataset2 = RBMDataSet(
                    satellite=SatelliteEnum.RBSPA,
                    instrument=InstrumentEnum.HOPE,  # Different instrument
                    mfm=MfmEnum.T89,
                    start_time=start_time,
                    end_time=end_time,
                    folder_path=folder_path,
                    verbose=False,
                )

                assert dataset1 != dataset2


def test_eq_file_loading_mode_different_mfm(mock_module_string):
    """Test inequality for file loading mode datasets with different MFM."""
    start_time = dt.datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_time = dt.datetime(2023, 1, 31, tzinfo=timezone.utc)
    folder_path = Path("/mock/path")

    with mock.patch(f"{mock_module_string}._create_date_list"):
        with mock.patch(f"{mock_module_string}._create_file_path_stem"):
            with mock.patch(f"{mock_module_string}._create_file_name_stem"):
                dataset1 = RBMDataSet(
                    satellite=SatelliteEnum.RBSPA,
                    instrument=InstrumentEnum.MAGEIS,
                    mfm=MfmEnum.T89,
                    start_time=start_time,
                    end_time=end_time,
                    folder_path=folder_path,
                    verbose=False,
                )

                dataset2 = RBMDataSet(
                    satellite=SatelliteEnum.RBSPA,
                    instrument=InstrumentEnum.MAGEIS,
                    mfm=MfmEnum.T96,  # Different MFM
                    start_time=start_time,
                    end_time=end_time,
                    folder_path=folder_path,
                    verbose=False,
                )

                assert dataset1 != dataset2


def test_eq_dict_mode_identical():
    """Test equality for identical dict mode datasets."""
    dataset1 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    dataset2 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    test_data = {
        "Flux": np.array([[1.0, 2.0, 3.0]]),
        "time": np.array([738000.0]),
        "energy_channels": np.array([100.0, 200.0, 300.0]),
        "Lstar": np.array([4.5, 5.0, 5.5]),
    }

    dataset1.update_from_dict(test_data)
    dataset2.update_from_dict(test_data.copy())

    assert dataset1 == dataset2


def test_eq_dict_mode_different_variables():
    """Test inequality for dict mode datasets with different variables."""
    dataset1 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    dataset2 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    dataset1.update_from_dict({"Flux": np.array([[1.0, 2.0, 3.0]])})
    dataset2.update_from_dict({"time": np.array([738000.0])})

    assert dataset1 != dataset2


def test_eq_dict_mode_same_variables_different_values():
    """Test inequality for dict mode datasets with same variables but different values."""
    dataset1 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    dataset2 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    dataset1.update_from_dict({"Flux": np.array([[1.0, 2.0, 3.0]])})
    dataset2.update_from_dict({"Flux": np.array([[4.0, 5.0, 6.0]])})

    assert dataset1 != dataset2


def test_eq_different_modes(mock_module_string):
    """Test inequality between file loading and dict mode datasets."""
    # File loading mode dataset
    start_time = dt.datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_time = dt.datetime(2023, 1, 31, tzinfo=timezone.utc)
    folder_path = Path("/mock/path")

    with mock.patch(f"{mock_module_string}._create_date_list"):
        with mock.patch(f"{mock_module_string}._create_file_path_stem"):
            with mock.patch(f"{mock_module_string}._create_file_name_stem"):
                file_dataset = RBMDataSet(
                    satellite=SatelliteEnum.RBSPA,
                    instrument=InstrumentEnum.MAGEIS,
                    mfm=MfmEnum.T89,
                    start_time=start_time,
                    end_time=end_time,
                    folder_path=folder_path,
                    verbose=False,
                )

    # Dict mode dataset
    dict_dataset = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    assert file_dataset != dict_dataset


def test_eq_array_comparison_with_nan():
    """Test equality with NaN values in arrays."""
    dataset1 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    dataset2 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    test_array = np.array([[1.0, np.nan, 3.0]])
    dataset1.update_from_dict({"Flux": test_array})
    dataset2.update_from_dict({"Flux": test_array.copy()})

    assert dataset1 == dataset2


def test_eq_list_comparison():
    """Test equality with list variables."""
    dataset1 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    dataset2 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    test_datetime_list = [dt.datetime(2023, 1, 15, tzinfo=timezone.utc)]
    dataset1.datetime = test_datetime_list
    dataset2.datetime = test_datetime_list.copy()

    assert dataset1 == dataset2


def test_eq_list_different_lengths():
    """Test inequality with lists of different lengths."""
    dataset1 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    dataset2 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    dataset1.datetime = [dt.datetime(2023, 1, 15, tzinfo=timezone.utc)]
    dataset2.datetime = [
        dt.datetime(2023, 1, 15, tzinfo=timezone.utc),
        dt.datetime(2023, 1, 16, tzinfo=timezone.utc),
    ]

    assert dataset1 != dataset2


def test_eq_array_different_shapes():
    """Test inequality with arrays of different shapes."""
    dataset1 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    dataset2 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    dataset1.update_from_dict({"Flux": np.array([[1.0, 2.0]])})
    dataset2.update_from_dict({"Flux": np.array([1.0, 2.0])})

    assert dataset1 != dataset2


def test_eq_different_types():
    """Test inequality when same variable has different types."""
    dataset1 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    dataset2 = RBMDataSet(
        satellite="RBSPA",
        instrument="hope",
        mfm="T89",
    )

    dataset1.time = np.array([738000.0])
    dataset2.time = [738000.0]

    assert dataset1 != dataset2
