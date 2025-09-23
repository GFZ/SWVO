# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from swvo.io.dst.read_dst_from_multiple_models import ModelError
from swvo.io.hp import (
    Hp30Ensemble,
    Hp30GFZ,
    Hp60Ensemble,
    Hp60GFZ,
    read_hp_from_multiple_models,
)

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))


@pytest.fixture(scope="session", autouse=True)
def set_env_var():
    ENV_VAR_NAMES = {
        "RT_HP_GFZ_STREAM_DIR": f"{str(DATA_DIR)}/gfz",
        "HP30_ENSEMBLE_FORECAST_DIR": f"{str(DATA_DIR)}/ensemble/hp30",
        "HP60_ENSEMBLE_FORECAST_DIR": f"{str(DATA_DIR)}/ensemble/hp60",
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
        "test_time_now": now,
    }


@pytest.mark.parametrize(
    "hp_index,models",
    [("hp30", [Hp30GFZ, Hp30Ensemble]), ("hp60", [Hp60GFZ, Hp60Ensemble])],
)
class TestHpFromMultipleModels:
    def test_basic_historical_read(self, sample_times, hp_index, models):
        data = read_hp_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["past_end"],
            hp_index=hp_index,
            model_order=[models[0]()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert isinstance(data, pd.DataFrame)
        assert hp_index in data.columns
        assert "model" in data.columns
        assert not data[hp_index].isna().all()
        assert data["model"].unique() == ["gfz"]

    def test_basic_forecast_read(self, sample_times, hp_index, models):
        data = read_hp_from_multiple_models(
            start_time=sample_times["future_start"],
            end_time=sample_times["future_end"],
            hp_index=hp_index,
            model_order=[models[1]()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert all(isinstance(d, pd.DataFrame) for d in data)
        assert all(hp_index in d.columns for d in data)
        assert all("model" in d.columns for d in data)
        assert all(all(d.model == "ensemble") for d in data)

    def test_ensemble_reduce_mean(self, sample_times, hp_index, models):
        data = read_hp_from_multiple_models(
            start_time=sample_times["future_start"],
            end_time=sample_times["future_end"],
            hp_index=hp_index,
            model_order=[models[1]()],
            reduce_ensemble="mean",
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert isinstance(data, pd.DataFrame)
        assert hp_index in data.columns
        assert not data[hp_index].isna().all()
        assert "ensemble" in data["model"].unique()

    def test_full_ensemble(self, sample_times, hp_index, models):
        data = read_hp_from_multiple_models(
            start_time=sample_times["future_start"],
            end_time=sample_times["future_end"],
            hp_index=hp_index,
            model_order=[models[1]()],
            reduce_ensemble=None,
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert isinstance(data, list)
        assert len(data) > 1
        assert all(isinstance(d, pd.DataFrame) for d in data)
        assert all(hp_index in d.columns for d in data)
        assert all("model" in d.columns for d in data)

    def test_time_ordering(self, sample_times, hp_index, models):
        data = read_hp_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            hp_index=hp_index,
            model_order=[models[0](), models[1]()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        for d in data:
            assert d.index.is_monotonic_increasing

    def test_invalid_time_range(self, sample_times, hp_index, models):
        with pytest.raises(AssertionError):
            read_hp_from_multiple_models(
                start_time=sample_times["future_end"],
                end_time=sample_times["past_start"],
                hp_index=hp_index,
                model_order=[models[0]()],
            )

    def test_model_transition(self, sample_times, hp_index, models):
        data = read_hp_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            hp_index=hp_index,
            model_order=[models[0](), models[1]()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )
        assert all([df.loc["2024-11-25 00:00:00+00:00"].model == "ensemble" for df in data])
        assert all([df.loc["2024-11-24 21:00:00+00:00"].model == "gfz" for df in data])

    def test_data_consistency(self, sample_times, hp_index, models):
        params = {
            "start_time": sample_times["past_start"],
            "end_time": sample_times["past_end"],
            "hp_index": hp_index,
            "model_order": [models[0]()],
            "historical_data_cutoff_time": sample_times["test_time_now"],
        }

        data1 = read_hp_from_multiple_models(**params)
        data2 = read_hp_from_multiple_models(**params)

        pd.testing.assert_frame_equal(data1, data2)

    def test_model_check_with_wrong_class(self, sample_times, hp_index, models):
        _ = models

        class FakeModel:
            pass

        fake = FakeModel()
        with pytest.raises(ModelError, match="Unknown or incompatible model"):
            read_hp_from_multiple_models(
                start_time=sample_times["past_start"],
                end_time=sample_times["future_end"],
                hp_index=hp_index,
                model_order=[fake],
            )


def test_invalid_hp_index():
    with pytest.raises(
        ValueError,
        match="Requested invalid_index index does not exist! Possible options: hp30, hp60",
    ):
        read_hp_from_multiple_models(
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc) + timedelta(days=1),
            hp_index="invalid_index",
        )
