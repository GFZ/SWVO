import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import pytest
import numpy as np

from data_management.io.hp.ensemble import HpEnsemble, Hp30Ensemble, Hp60Ensemble

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_hp_ensemble"


@pytest.fixture(scope="session",autouse=True)
def setup_and_cleanup():
    TEST_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    yield

    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)


@pytest.fixture
def hp30ensemble_instance():

    (DATA_DIR / "hp30").mkdir(exist_ok=True)
    return Hp30Ensemble(data_dir=DATA_DIR / "hp30")



@pytest.mark.parametrize("instance_type,index_name", [("hp30", "hp30"), ("hp60", "hp60")])
def test_initialization(instance_type, index_name):

    ensemble_dir = DATA_DIR / instance_type
    ensemble_dir.mkdir(exist_ok=True)

    ensemble_class = Hp30Ensemble if instance_type == "hp30" else Hp60Ensemble

    instance = ensemble_class(data_dir=ensemble_dir)
    assert instance.index == index_name
    assert instance.index_number == index_name[2:]
    assert instance.data_dir == ensemble_dir


def test_initialization_without_env_var():
    if Hp30Ensemble.ENV_VAR_NAME in os.environ:
        del os.environ[Hp30Ensemble.ENV_VAR_NAME]
    with pytest.raises(ValueError, match=f"Necessary environment variable {Hp30Ensemble.ENV_VAR_NAME} not set!"):
        Hp30Ensemble()


def test_invalid_index():
    with pytest.raises(AssertionError):
        HpEnsemble("hp45", data_dir=DATA_DIR)


@pytest.mark.parametrize("instance_type,index_name", [("hp30", "hp30"), ("hp60", "hp60")])
def test_read_with_ensemble_data(instance_type, index_name):

    ensemble_dir = DATA_DIR / instance_type
    ensemble_dir.mkdir(exist_ok=True)
    instance_class = Hp30Ensemble if instance_type == "hp30" else Hp60Ensemble
    instance = instance_class(data_dir=ensemble_dir)

    current_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    str_date = current_time.strftime("%Y%m%dT%H%M%S")

    for i in range(3):

        test_dates = pd.date_range(
            start=current_time, end=current_time + timedelta(days=3), freq=f"{instance.index_number}min"
        )
        df = pd.DataFrame(
            {"t": test_dates.strftime("%Y-%m-%d %H:%M:%S"), index_name: np.random.uniform(10, 20, size=len(test_dates))}
        )

        filename = f"FORECAST_{index_name.upper()}_SWIFT_DRIVEN_swift_{str_date}_ensemble_{i+1}.csv"
        file_path = instance.data_dir / filename
        df.to_csv(file_path, index=False, header=False)

    data = instance.read(current_time, current_time + timedelta(days=1))

    assert isinstance(data, list)
    assert len(data) == 3
    for df in data:
        assert isinstance(df, pd.DataFrame)
        assert index_name in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz == timezone.utc
        assert not df.empty
        assert df.index[0] >= current_time.replace(tzinfo=timezone.utc) - timedelta(minutes=float(instance.index_number) - 0.01)
        assert df.index[-1] <= current_time.replace(tzinfo=timezone.utc) + timedelta(days=1) + timedelta(minutes=float(instance.index_number) + 0.01)


def test_read_with_default_times(instance_type="hp30"):

    ensemble_dir = DATA_DIR / instance_type
    ensemble_dir.mkdir(exist_ok=True)
    instance_class = Hp30Ensemble if instance_type == "hp30" else Hp60Ensemble
    instance = instance_class(data_dir=ensemble_dir)
    data = instance.read(None, None)

    assert isinstance(data, list)
    assert len(data) == 3
    assert isinstance(data[0], pd.DataFrame)
    assert all("hp30" in i.columns for i in data)
    assert data[0].index.tz == timezone.utc