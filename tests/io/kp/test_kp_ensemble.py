import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import pytest
import numpy as np
from data_management.io.kp import KpEnsemble

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_kp_ensemble"


@pytest.fixture(scope="session", autouse=True)
def setup_and_cleanup():

    TEST_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    yield
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

@pytest.fixture
def kp_ensemble_instance():

    return KpEnsemble(data_dir=DATA_DIR)


def test_initialization_with_data_dir():

    instance = KpEnsemble(data_dir=DATA_DIR)
    assert instance.data_dir == DATA_DIR


def test_initialization_without_env_var():

    if KpEnsemble.ENV_VAR_NAME in os.environ:
        del os.environ[KpEnsemble.ENV_VAR_NAME]
    with pytest.raises(ValueError, match=f"Necessary environment variable {KpEnsemble.ENV_VAR_NAME} not set!"):
        KpEnsemble()


def test_initialization_with_env_var():

    os.environ[KpEnsemble.ENV_VAR_NAME] = str(DATA_DIR)
    instance = KpEnsemble()
    assert instance.data_dir == DATA_DIR


def test_initialization_with_nonexistent_directory():

    with pytest.raises(FileNotFoundError):
        KpEnsemble(data_dir="nonexistent_directory")


def test_read_with_ensemble_data(kp_ensemble_instance):

    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    str_date = current_time.strftime("%Y%m%dT%H0000")

    for i in range(3):
        test_dates = pd.date_range(start=current_time, end=current_time + timedelta(days=3), freq="3h")

        df = pd.DataFrame(
            {
                "t": test_dates.strftime("%Y-%m-%d %H:%M:%S"),
                "kp": np.random.uniform(0, 9, size=len(test_dates)),
            }
        )

        filename = f"FORECAST_PAGER_SWIFT_swift_{str_date}_ensemble_{i+1}.csv"
        file_path = kp_ensemble_instance.data_dir / filename
        df.to_csv(file_path, index=False, header=False)

    data = kp_ensemble_instance.read(current_time, current_time + timedelta(days=1))

    assert isinstance(data, list)
    assert len(data) == 3
    for df in data:
        assert isinstance(df, pd.DataFrame)
        assert "kp" in df.columns
        assert "file_name" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        assert not df.empty

        assert df.index[0] >= current_time.replace(tzinfo=timezone.utc) - timedelta(hours=3)
        assert df.index[-1] <= current_time.replace(tzinfo=timezone.utc) + timedelta(days=1) + timedelta(hours=3)


def test_read_with_default_times(kp_ensemble_instance):

    current_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    str_date = current_time.strftime("%Y%m%dT%H0000")

    test_dates = pd.date_range(start=current_time, end=current_time + timedelta(days=3), freq="3h")
    df = pd.DataFrame({"t": test_dates.strftime("%Y-%m-%d %H:%M:%S"), "kp": np.random.uniform(0, 9, size=len(test_dates))})
    filename = f"FORECAST_PAGER_SWIFT_swift_{str_date}_ensemble_1.csv"
    file_path = kp_ensemble_instance.data_dir / filename
    df.to_csv(file_path, index=False, header=False)

    data = kp_ensemble_instance.read(None, None)

    assert isinstance(data, list)
    assert len(data) > 0
    assert isinstance(data[0], pd.DataFrame)
    assert not data[0].empty


def test_read_empty_directory(kp_ensemble_instance):
    for files in DATA_DIR.glob("*"):
        files.unlink()
    
    current_time = datetime.now()

    with pytest.raises(FileNotFoundError):
        _ = kp_ensemble_instance.read(current_time, current_time + timedelta(days=1))