# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
from datetime import timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from swvo.io.RBMDataSet import (
    FileCadenceEnum,
    InstrumentEnum,
    MfmEnum,
    RBMDataSet,
    SatelliteEnum,
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
