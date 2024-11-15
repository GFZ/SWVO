import os
import pytest
from pathlib import Path
from datetime import datetime, timezone
from data_management.io.omni.omni_high_res import OMNIHighRes
import pandas as pd


TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))


@pytest.fixture
def omni_high_res():
    os.environ["OMNI_HIGH_RES_STREAM_DIR"] = str(DATA_DIR)
    yield OMNIHighRes()


def test_initialization_with_env_var(omni_high_res):
    assert omni_high_res.data_dir.exists()


def test_initialization_with_data_dir():
    omni_high_res = OMNIHighRes(data_dir=DATA_DIR)
    assert omni_high_res.data_dir == DATA_DIR


def test_initialization_without_env_var():
    if "OMNI_HIGH_RES_STREAM_DIR" in os.environ:
        del os.environ["OMNI_HIGH_RES_STREAM_DIR"]
    with pytest.raises(ValueError):
        OMNIHighRes()


def test_download_and_process(omni_high_res, mocker):
    mocker.patch("wget.download")
    mocker.patch.object(
        omni_high_res,
        "_process_single_file",
        return_value=omni_high_res._process_single_file(
            Path(TEST_DIR) / "data/omni_min2020.asc"
        ),
    )

    start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

    omni_high_res.download_and_process(start_time, end_time, verbose=True)

    assert (TEST_DIR / Path("data/omni_min2020.asc")).exists()


def test_read_without_download(omni_high_res, mocker):
    start_time = datetime(2021, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2021, 12, 31, tzinfo=timezone.utc)
    with pytest.raises(FileNotFoundError):
        omni_high_res.read(start_time, end_time, download=False)


def test_read_with_download(omni_high_res, mocker):
    mocker.patch.object(omni_high_res, "download_and_process")
    mocker.patch.object(omni_high_res, "_read_single_file", return_value=pd.DataFrame())
    start_time = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2022, 12, 31, tzinfo=timezone.utc)
    omni_high_res.read(start_time, end_time, download=True)
    omni_high_res.download_and_process.assert_called_once()


def test_process_single_file(omni_high_res):
    file = Path(TEST_DIR) / "data/omni_min2020.asc"
    df = omni_high_res._process_single_file(file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_read_single_file(omni_high_res):

    csv_file = Path(TEST_DIR) / "data/OMNI_HIGH_RES_1min_2020.csv"
    df = omni_high_res._read_single_file(csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_invalid_cadence(omni_high_res):
    start_time = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2022, 12, 31, tzinfo=timezone.utc)

    with pytest.raises(AssertionError):
        omni_high_res.read(start_time, end_time, cadence_min=2)

    with pytest.raises(AssertionError):
        omni_high_res.download_and_process(start_time, end_time, cadence_min=10)


def test_start_year_behind(omni_high_res, mocker):

    start_time = datetime(1920, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

    mock_print = mocker.patch("builtins.print")

    mocker.patch.object(
        omni_high_res, "_get_processed_file_list", return_value=([], [])
    )
    mocker.patch.object(omni_high_res, "_read_single_file", return_value=pd.DataFrame())

    mocker.patch("pandas.concat", return_value=pd.DataFrame())

    mocker.patch.object(pd.DataFrame, "truncate", return_value=pd.DataFrame())

    dfs = omni_high_res.read(start_time, end_time)

    mock_print.assert_called_once_with(
        "Start date chosen falls behind the existing data. Moving start date to first available mission files..."
    )

    assert len(dfs) == 0, "Expected dfs list to be empty since no files are found."


def test_remove_processed_file():

    os.remove(Path(TEST_DIR) / "data/OMNI_HIGH_RES_1min_2020.csv")
