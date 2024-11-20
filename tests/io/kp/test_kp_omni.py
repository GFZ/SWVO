import os
import pytest
from pathlib import Path
from datetime import datetime, timezone
from data_management.io.kp import KpOMNI
import pandas as pd

from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))


@pytest.fixture
def kp_omni():
    os.environ["OMNI_LOW_RES_STREAM_DIR"] = str(DATA_DIR)
    yield KpOMNI()


@pytest.fixture
def mock_kp_omni_data():
    test_dates = pd.date_range(
        start=datetime(2020, 1, 1), end=datetime(2020, 12, 31), freq="h"
    )
    test_data = pd.DataFrame(
        {
            "t": test_dates,
            "kp": [150.0] * len(test_dates),
            "timestamp": test_dates.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    test_data.index = test_dates
    return test_data


def test_initialization_with_env_var(kp_omni):
    assert kp_omni.data_dir.exists()


def test_initialization_with_data_dir():
    kp_omni = KpOMNI(data_dir=DATA_DIR)
    assert kp_omni.data_dir == DATA_DIR


def test_initialization_without_env_var():
    if "OMNI_LOW_RES_STREAM_DIR" in os.environ:
        del os.environ["OMNI_LOW_RES_STREAM_DIR"]
    with pytest.raises(ValueError):
        KpOMNI()


def test_download_and_process(kp_omni, mocker):
    mocker.patch("wget.download")
    mocker.patch.object(
        kp_omni,
        "_process_single_file",
        return_value=kp_omni._process_single_file(
            Path(TEST_DIR) / "data/omni2_2020.dat"
        ),
    )

    start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

    kp_omni.download_and_process(start_time, end_time, verbose=True)

    assert (TEST_DIR / Path("data/omni2_2020.dat")).exists()


def test_read_without_download(kp_omni, mocker):
    start_time = datetime(2021, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2021, 12, 31, tzinfo=timezone.utc)
    with pytest.raises(ValueError): #value error is raised when no files are found hence no concatenation is possible
        kp_omni.read(start_time, end_time, download=False)


def test_read_with_download(kp_omni, mock_kp_omni_data, mocker):

    mocker.patch("pathlib.Path.exists", return_value=False)
    mocker.patch.object(kp_omni, "_read_single_file", return_value=mock_kp_omni_data)
    mocker.patch.object(kp_omni, "download_and_process")

    start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

    df = kp_omni.read(start_time, end_time, download=True)
    kp_omni.download_and_process.assert_called_once()

    assert not df.empty
    assert "kp" in df.columns
    assert all(df["kp"] == 150.0)
    assert all(idx.hour % 3== 0 for idx in df.index)
    assert all(idx.tzinfo is not None for idx in df.index)


def test_process_single_file(kp_omni):
    file = Path(TEST_DIR) / "data/omni2_2020.dat"
    df = kp_omni._process_single_file(file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "kp" in df.columns


def test_read_single_file(kp_omni):

    csv_file = Path(TEST_DIR) / "data/OMNI_LOW_RES_2020.csv"
    df = kp_omni._read_single_file(csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "kp" in df.columns


def test_start_year_behind(kp_omni, mocker, mock_kp_omni_data):
    start_time = datetime(1920, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)
    
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch.object(
        kp_omni, "_get_processed_file_list", return_value=([Path("dummy.csv")], [])
    )
    mocker.patch.object(kp_omni, "_read_single_file", return_value=mock_kp_omni_data)

    df = pd.DataFrame(
        {
            "kp": [],
        }
    )
    df.index = pd.DatetimeIndex([])

    mocker.patch("pandas.concat", return_value=df)
    mocker.patch.object(pd.DataFrame, "truncate", return_value=df)

    with patch("logging.Logger.warning") as mock_warning:
        result_df = kp_omni.read(start_time, end_time)

        mock_warning.assert_any_call(
            "Start date chosen falls behind the existing data. Moving start date to first" " available mission files..."
        )

    assert result_df.empty, "Expected resulting DataFrame to be empty"





def test_remove_processed_file():

    os.remove(Path(TEST_DIR) / "data/OMNI_LOW_RES_2020.csv")
