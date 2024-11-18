import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import pytest
import numpy as np

from data_management.io.kp import KpNiemegk

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_kp_niemegk"


@pytest.fixture(scope="session", autouse=True)
def setup_and_cleanup():

    TEST_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    yield

    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)


@pytest.fixture
def kp_niemegk_instance():

    return KpNiemegk(data_dir=DATA_DIR)


def test_initialization_with_data_dir():

    instance = KpNiemegk(data_dir=DATA_DIR)
    assert instance.data_dir == DATA_DIR
    assert instance.data_dir.exists()


def test_initialization_without_env_var():

    if KpNiemegk.ENV_VAR_NAME in os.environ:
        del os.environ[KpNiemegk.ENV_VAR_NAME]
    with pytest.raises(
        ValueError,
        match=f"Necessary environment variable {KpNiemegk.ENV_VAR_NAME} not set!",
    ):
        KpNiemegk()


def test_initialization_with_env_var():

    os.environ[KpNiemegk.ENV_VAR_NAME] = str(DATA_DIR)
    instance = KpNiemegk()
    assert instance.data_dir == DATA_DIR


def test_get_processed_file_list():

    instance = KpNiemegk(data_dir=DATA_DIR)
    start_time = datetime(2024, 1, 1)
    end_time = datetime(2024, 1, 3)

    file_paths, time_intervals = instance._get_processed_file_list(start_time, end_time)

    assert len(file_paths) == 4
    assert len(time_intervals) == 4

    assert file_paths[0].name == "NIEMEGK_KP_NOWCAST_20240101.csv"
    assert file_paths[-1].name == "NIEMEGK_KP_NOWCAST_20240104.csv"

    assert time_intervals[0][0] == datetime(2023, 12, 30)
    assert time_intervals[0][1] == datetime(2024, 1, 1, 23, 59, 59)


def test_download_and_process_current_month(kp_niemegk_instance):

    current_time = datetime.now(timezone.utc)
    end_time = current_time + timedelta(days=2)

    kp_niemegk_instance.download_and_process(
        current_time, end_time, reprocess_files=True, verbose=True
    )

    file_paths, _ = kp_niemegk_instance._get_processed_file_list(current_time, end_time)

    for file_path in file_paths:
        if file_path.exists():
            df = pd.read_csv(file_path, names=["t", "kp"])
            assert len(df) > 0
            assert "t" in df.columns
            assert "kp" in df.columns

            valid_kps = df["kp"].dropna()
            assert valid_kps.min() >= 0
            assert valid_kps.max() <= 9


def test_download_past_month(kp_niemegk_instance):

    past_time = datetime.now() - timedelta(days=32)
    end_time = past_time + timedelta(days=2)

    kp_niemegk_instance.download_and_process(past_time, end_time, verbose=True)

    file_paths, _ = kp_niemegk_instance._get_processed_file_list(past_time, end_time)

    for file_path in file_paths:
        assert not file_path.exists()


def test_read_with_download(kp_niemegk_instance):

    current_time = datetime.now()
    end_time = current_time + timedelta(days=1)

    data = kp_niemegk_instance.read(current_time, end_time, download=True)

    assert isinstance(data, pd.DataFrame)
    assert "kp" in data.columns
    assert "file_name" in data.columns
    assert isinstance(data.index, pd.DatetimeIndex)

    assert data.index[0] >= current_time.replace(tzinfo=timezone.utc) - timedelta(
        hours=3
    )
    assert data.index[-1] <= end_time.replace(tzinfo=timezone.utc) + timedelta(hours=3)

    valid_kps = data["kp"].dropna()
    assert valid_kps.min() >= 0
    assert valid_kps.max() <= 9


def test_read_without_download_no_file(kp_niemegk_instance):

    current_time = datetime.now()
    end_time = current_time + timedelta(days=1)

    if DATA_DIR.exists():
        for file in DATA_DIR.glob("NIEMEGK_KP_NOWCAST_*.csv"):
            file.unlink()

    data = kp_niemegk_instance.read(current_time, end_time, download=False)
    assert isinstance(data, pd.DataFrame)
    assert data["kp"].isna().all()


def test_process_single_file():

    instance = KpNiemegk(data_dir=DATA_DIR)
    temp_dir = Path("./temp_test")
    temp_dir.mkdir(exist_ok=True)

    try:

        sample_data = """24021 2o 2+ 3- 3o 3+ 4- 4o 4+\n"""

        with open(temp_dir / "qlyymm.tab", "w") as f:
            f.write(sample_data)

        df = instance._process_single_file(temp_dir)

        assert isinstance(df, pd.DataFrame)
        assert "kp" in df.columns
        assert len(df) == 8

        expected_values = [
            2.0,
            2.333,
            2.667,
            3.0,
            3.333,
            3.667,
            4.0,
            4.333,
        ]

        for actual, expected in zip(df["kp"].values, expected_values):
            assert np.isclose(actual, expected, atol=0.001)

    finally:
        shutil.rmtree(temp_dir)


def test_reprocess_files_flag(kp_niemegk_instance):

    current_time = datetime.now()
    end_time = current_time + timedelta(days=1)

    kp_niemegk_instance.download_and_process(
        current_time, end_time, reprocess_files=True, verbose=True
    )

    file_paths, _ = kp_niemegk_instance._get_processed_file_list(current_time, end_time)

    initial_data = None
    for file_path in file_paths:
        if file_path.exists():
            initial_data = pd.read_csv(file_path, names=["t", "kp"])
            break

    assert initial_data is not None

    kp_niemegk_instance.download_and_process(
        current_time, end_time, reprocess_files=False, verbose=True
    )

    unchanged_data = pd.read_csv(file_path, names=["t", "kp"])
    pd.testing.assert_frame_equal(initial_data, unchanged_data)
