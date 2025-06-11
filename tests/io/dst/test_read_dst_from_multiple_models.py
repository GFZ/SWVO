import pytest
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import os
from data_management.io.dst import DSTOMNI, DSTWDC, read_dst_from_multiple_models
from pathlib import Path

from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))


ENV_VAR_NAMES = {
    "OMNI_LOW_RES_STREAM_DIR": f"{str(DATA_DIR)}",
    "WDC_STREAM_DIR": f"{str(DATA_DIR)}/wdc",
}


for key, var in ENV_VAR_NAMES.items():
    os.environ[key] = ENV_VAR_NAMES[key]


class TestReadDSTFromMultipleModels:
    @pytest.fixture
    def sample_times(self):
        now = datetime(2024, 11, 25).replace(
            tzinfo=timezone.utc, minute=0, second=0, microsecond=0
        )
        return {
            "past_start": now - timedelta(days=5),
            "past_end": now - timedelta(days=2),
            "future_start": now + timedelta(days=1),
            "future_end": now + timedelta(days=3),
            "test_time_now": now,
        }

    def test_basic_historical_read(self, sample_times):
        data = read_dst_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["past_end"],
            model_order=[DSTOMNI()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert isinstance(data, pd.DataFrame)
        assert "dst" in data.columns
        assert "model" in data.columns
        assert "file_name" in data.columns
        assert not data["dst"].isna().all()

    def test_basic_forecast_read(self, sample_times):
        data = read_dst_from_multiple_models(
            start_time=sample_times["future_start"],
            end_time=sample_times["future_end"],
            model_order=[DSTWDC()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert isinstance(data, pd.DataFrame)
        assert "dst" in data.columns
        assert "model" in data.columns

    def test_time_ordering(self, sample_times):
        start = sample_times["past_start"]
        end = sample_times["future_end"]

        data = read_dst_from_multiple_models(
            start_time=start,
            end_time=end,
            model_order=[DSTOMNI(), DSTWDC()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert data.index.is_monotonic_increasing
        assert data.index.min() >= start
        assert data.index.max() <= end

    def test_invalid_time_range(self, sample_times):
        with pytest.raises(AssertionError):
            read_dst_from_multiple_models(
                start_time=sample_times["future_end"],
                end_time=sample_times["past_start"],
                model_order=[DSTOMNI()],
            )

    def test_model_transition_1(self, sample_times):
        """Test that the model transition works correctly if the omni data is only available in the past and the swpc data is only available in the future and there is gap in data between the two models."""
        data = read_dst_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            model_order=[DSTOMNI(), DSTWDC()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )


        assert data.loc["2024-11-22 20:00:00+00:00"].model == "omni"
        assert data.loc["2024-11-22 21:00:00+00:00"].model == "wdc"

    def test_model_transition_2(self, sample_times):
        """Test that the model transition works correctly if the omni data is only available in the past and the swpc data is only available in the future and there is no gap in data between the two models."""
        data = read_dst_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            model_order=[DSTOMNI(), DSTWDC()],
            historical_data_cutoff_time=sample_times["test_time_now"] + timedelta(days=-2),
        )

        assert data.loc["2024-11-22 00:00:00+00:00"].model == "omni"
        assert data.loc["2024-11-23 00:00:00+00:00"].model == "wdc"

    def test_data_consistency(self, sample_times):
        params = {
            "start_time": sample_times["past_start"],
            "end_time": sample_times["past_end"],
            "model_order": [DSTOMNI()],
            "historical_data_cutoff_time": sample_times["test_time_now"],
        }

        data1 = read_dst_from_multiple_models(**params)
        data2 = read_dst_from_multiple_models(**params)

        pd.testing.assert_frame_equal(data1, data2)


    def test_synthetic_now_time_deprecation_with_message(self, sample_times):
        with pytest.warns(DeprecationWarning, match="synthetic_now_time.*deprecated"):
            read_dst_from_multiple_models(
                start_time=sample_times["past_start"],
                end_time=sample_times["future_end"],
                synthetic_now_time= sample_times["test_time_now"],
            )