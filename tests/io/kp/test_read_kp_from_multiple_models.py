import pytest
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import os
from data_management.io.kp import KpEnsemble, KpNiemegk, KpOMNI, KpSWPC, read_kp_from_multiple_models
from pathlib import Path

from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))


@pytest.fixture(scope="session", autouse=True)
def set_env_var():
    ENV_VAR_NAMES = {
        "OMNI_LOW_RES_STREAM_DIR": f"{str(DATA_DIR)}",
        "KP_ENSEMBLE_OUTPUT_DIR": f"{str(DATA_DIR)}/ensemble",
        "RT_KP_NIEMEGK_STREAM_DIR": f"{str(DATA_DIR)}/Niemegk",
        "RT_KP_SWPC_STREAM_DIR": f"{str(DATA_DIR)}/SWPC",
    }


    for key, var in ENV_VAR_NAMES.items():
        os.environ[key] = ENV_VAR_NAMES[key]


@pytest.fixture
def sample_times():
    now = datetime(2024, 11, 25).replace(tzinfo=timezone.utc, minute=0, second=0, microsecond=0)
    return {
        "past_start": now - timedelta(days=5),
        "past_end": now - timedelta(days=2),
        "future_start": now + timedelta(days=1),
        "future_end": now + timedelta(days=3),
        "now": now,
    }


def test_basic_historical_read(sample_times):

    data = read_kp_from_multiple_models(
        start_time=sample_times["past_start"],
        end_time=sample_times["past_end"],
        model_order=[KpOMNI(), KpNiemegk()],
        synthetic_now_time=sample_times["now"],
    )

    assert isinstance(data, pd.DataFrame)
    assert "kp" in data.columns
    assert "model" in data.columns
    assert data.loc["2024-11-22 18:00:00+00:00"].model == "omni"
    assert data.loc["2024-11-22 21:00:00+00:00"].model == "niemegk"
    assert not data["kp"].isna().all()
    assert not data["file_name"].isna().all()


def test_basic_forecast_read(sample_times):

    data = read_kp_from_multiple_models(
        start_time=sample_times["future_start"],
        end_time=sample_times["future_end"],
        model_order=[KpEnsemble(), KpSWPC()],
        synthetic_now_time=sample_times["now"],
    )

    assert all(isinstance(d, pd.DataFrame) for d in data)
    assert all("kp" in d.columns for d in data)
    assert all("model" in d.columns for d in data)
    assert all(not d["file_name"].isna().all() for d in data)
    assert all(all(d.model == "ensemble") for d in data)


def test_ensemble_reduce_mean(sample_times):

    data = read_kp_from_multiple_models(
        start_time=sample_times["future_start"],
        end_time=sample_times["future_end"],
        model_order=[KpEnsemble()],
        reduce_ensemble="mean",
        synthetic_now_time=sample_times["now"],
    )

    assert isinstance(data, pd.DataFrame)
    assert "kp" in data.columns
    assert not data["kp"].isna().all()
    assert "ensemble" in data["model"].unique()


def test_full_ensemble(sample_times):

    data = read_kp_from_multiple_models(
        start_time=sample_times["future_start"],
        end_time=sample_times["future_end"],
        model_order=[KpEnsemble()],
        reduce_ensemble=None,
        synthetic_now_time=sample_times["now"],
    )

    assert isinstance(data, list)
    assert len(data) > 1
    assert all(isinstance(d, pd.DataFrame) for d in data)
    assert all("kp" in d.columns for d in data)
    assert all("model" in d.columns for d in data)
    assert all(not d["file_name"].isna().all() for d in data)


def test_time_ordering_and_transition(sample_times):

    data = read_kp_from_multiple_models(
        start_time=sample_times["past_start"],
        end_time=sample_times["future_end"],
        model_order=[KpOMNI(), KpNiemegk(), KpEnsemble(), KpSWPC()],
        synthetic_now_time=sample_times["now"],
    )

    for d in data:
        assert d.index.is_monotonic_increasing
        assert d.loc["2024-11-25 00:00:00+00:00"].model == "ensemble"
        assert d.loc["2024-11-24 21:00:00+00:00"].model == "niemegk"


def test_time_boundaries(sample_times):

    start = sample_times["past_start"]
    end = sample_times["future_end"]

    data = read_kp_from_multiple_models(
        start_time=start,
        end_time=end,
        model_order=[KpOMNI(), KpNiemegk(), KpEnsemble(), KpSWPC()],
        synthetic_now_time=sample_times["now"],
    )

    for d in data:
        assert d.index.min() >= start
        assert d.index.max() <= end


def test_invalid_time_range(sample_times):

    with pytest.raises(AssertionError):
        read_kp_from_multiple_models(
            start_time=sample_times["future_end"], end_time=sample_times["past_start"], model_order=[KpOMNI()]
        )


def test_date_more_than_3_days_in_future(sample_times):
    with pytest.raises(ValueError, match="We can only read 3 days at a time of Kp SWPC!"):
        read_kp_from_multiple_models(
            start_time=sample_times["now"] - timedelta(days=6),
            end_time=sample_times["now"] + timedelta(days=4),
            synthetic_now_time=sample_times["now"],
        )


def test_kp_value_range(sample_times):

    data = read_kp_from_multiple_models(
        start_time=sample_times["past_start"],
        end_time=sample_times["future_end"],
        model_order=[KpOMNI(), KpNiemegk(), KpEnsemble(), KpSWPC()],
        synthetic_now_time=sample_times["now"],
    )

    def check_kp_range(df):
        valid_kp = df["kp"].dropna()
        assert (valid_kp >= 0).all() and (valid_kp <= 9).all()

    for d in data:
        check_kp_range(d)


def test_model_transition(sample_times):

    data = read_kp_from_multiple_models(
        start_time=sample_times["past_start"],
        end_time=sample_times["future_end"],
        model_order=[KpOMNI(), KpNiemegk(), KpEnsemble(), KpSWPC()],
        synthetic_now_time=sample_times["now"],
    )

    assert all([df.loc["2024-11-25 00:00:00+00:00"].model == "ensemble" for df in data])
    assert all([df.loc["2024-11-24 21:00:00+00:00"].model == "niemegk" for df in data])
    assert not all([df.loc["2024-11-24 21:00:00+00:00"].model == "omni" for df in data])



def test_data_consistency(sample_times):

    params = {
        "start_time": sample_times["past_start"],
        "end_time": sample_times["past_end"],
        "model_order": [KpOMNI()],
        "synthetic_now_time": sample_times["now"],
    }

    data1 = read_kp_from_multiple_models(**params)
    data2 = read_kp_from_multiple_models(**params)

    pd.testing.assert_frame_equal(data1, data2)
