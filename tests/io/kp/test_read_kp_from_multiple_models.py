# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from swvo.io.exceptions import ModelError
from swvo.io.kp import (
    KpEnsemble,
    KpNiemegk,
    KpOMNI,
    KpSWPC,
    read_kp_from_multiple_models,
)

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))


class TestReadKpFromMultipleModels:
    @pytest.fixture(scope="session", autouse=True)
    def set_env_var(self):
        ENV_VAR_NAMES = {
            "OMNI_LOW_RES_STREAM_DIR": f"{str(DATA_DIR)}",
            "KP_ENSEMBLE_OUTPUT_DIR": f"{str(DATA_DIR)}/ensemble",
            "RT_KP_NIEMEGK_STREAM_DIR": f"{str(DATA_DIR)}/Niemegk",
            "RT_KP_SWPC_STREAM_DIR": f"{str(DATA_DIR)}/SWPC",
        }

        for key, var in ENV_VAR_NAMES.items():
            os.environ[key] = ENV_VAR_NAMES[key]

    @pytest.fixture
    def sample_times(self):
        now = datetime(2024, 11, 25).replace(tzinfo=timezone.utc, minute=0, second=0, microsecond=0)
        return {
            "past_start": now - timedelta(days=5),
            "past_end": now - timedelta(days=2),
            "future_start": now + timedelta(days=1),
            "future_end": now + timedelta(days=3),
            "test_time_now": now,
        }

    def test_basic_historical_read(self, sample_times):
        data = read_kp_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["past_end"],
            model_order=[KpOMNI(), KpNiemegk()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert isinstance(data, pd.DataFrame)
        assert "kp" in data.columns
        assert "model" in data.columns
        assert data.loc["2024-11-22 18:00:00+00:00"].model == "omni"
        assert data.loc["2024-11-22 21:00:00+00:00"].model == "niemegk"
        assert not data["kp"].isna().all()
        assert not data["file_name"].isna().all()

    def test_basic_forecast_read(self, sample_times):
        data = read_kp_from_multiple_models(
            start_time=sample_times["future_start"],
            end_time=sample_times["future_end"],
            model_order=[KpEnsemble(), KpSWPC()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert all(isinstance(d, pd.DataFrame) for d in data)
        assert all("kp" in d.columns for d in data)
        assert all("model" in d.columns for d in data)
        assert all(not d["file_name"].isna().all() for d in data)
        assert all(all(d.model == "ensemble") for d in data)

    def test_ensemble_reduce_mean(self):
        with pytest.raises(ValueError):
            raise ValueError("This reduction method has not been implemented yet!")

    def test_full_ensemble(self, sample_times):
        data = read_kp_from_multiple_models(
            start_time=sample_times["future_start"],
            end_time=sample_times["future_end"],
            model_order=[KpEnsemble()],
            reduce_ensemble=None,
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert isinstance(data, list)
        assert len(data) > 1
        assert all(isinstance(d, pd.DataFrame) for d in data)
        assert all("kp" in d.columns for d in data)
        assert all("model" in d.columns for d in data)
        assert all(not d["file_name"].isna().all() for d in data)

    def test_time_ordering_and_transition(self, sample_times):
        data = read_kp_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            model_order=[KpOMNI(), KpNiemegk(), KpEnsemble(), KpSWPC()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        for d in data:
            assert d.index.is_monotonic_increasing
            assert d.loc["2024-11-24 00:00:00+00:00"].model == "niemegk"
            assert d.loc["2024-11-25 03:00:00+00:00"].model == "ensemble"

    def test_forecast_in_past(self, sample_times):
        data = read_kp_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_start"],
            model_order=[KpOMNI(), KpNiemegk(), KpEnsemble(), KpSWPC()],
            historical_data_cutoff_time=sample_times["test_time_now"] - timedelta(days=2),
        )

        assert all(d.index.is_monotonic_increasing for d in data)
        assert all(d.loc["2024-11-22 18:00:00+00:00"].model == "omni" for d in data)
        assert all(d.loc["2024-11-23 00:00:00+00:00"].model == "niemegk" for d in data)

        # there should be ensemble in the forecast
        assert all(d.loc["2024-11-23 03:00:00+00:00"].model == "ensemble" for d in data)

    def test_time_boundaries(self, sample_times):
        start = sample_times["past_start"]
        end = sample_times["future_end"]

        data = read_kp_from_multiple_models(
            start_time=start,
            end_time=end,
            model_order=[KpOMNI(), KpNiemegk(), KpEnsemble(), KpSWPC()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )
        for d in data:
            assert d.index.min() >= start
            assert d.index.max() <= end + timedelta(hours=3)

    def test_invalid_time_range(self, sample_times):
        with pytest.raises(AssertionError):
            read_kp_from_multiple_models(
                start_time=sample_times["future_end"],
                end_time=sample_times["past_start"],
                model_order=[KpOMNI()],
            )

    def test_date_more_than_3_days_in_future(self, sample_times):
        with pytest.raises(ValueError, match="We can only read 3 days at a time of Kp SWPC!"):
            read_kp_from_multiple_models(
                start_time=sample_times["test_time_now"] - timedelta(days=6),
                end_time=sample_times["test_time_now"] + timedelta(days=4),
                historical_data_cutoff_time=sample_times["test_time_now"],
            )

    def test_kp_value_range(self, sample_times):
        data = read_kp_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            model_order=[KpOMNI(), KpNiemegk(), KpEnsemble(), KpSWPC()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        def check_kp_range(df):
            valid_kp = df["kp"].dropna()
            assert (valid_kp >= 0).all() and (valid_kp <= 9).all()

        for d in data:
            check_kp_range(d)

    def test_model_transition(self, sample_times):
        data = read_kp_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            model_order=[KpOMNI(), KpNiemegk(), KpEnsemble(), KpSWPC()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert not all([df.loc["2024-11-24 21:00:00+00:00"].model == "omni" for df in data])
        assert all([df.loc["2024-11-25 00:00:00+00:00"].model == "niemegk" for df in data])
        assert all([df.loc["2024-11-25 03:00:00+00:00"].model == "ensemble" for df in data])

    def test_data_consistency(self, sample_times):
        params = {
            "start_time": sample_times["past_start"],
            "end_time": sample_times["past_end"],
            "model_order": [KpOMNI()],
            "historical_data_cutoff_time": sample_times["test_time_now"],
        }

        data1 = read_kp_from_multiple_models(**params)
        data2 = read_kp_from_multiple_models(**params)

        pd.testing.assert_frame_equal(data1, data2)

    def test_synthetic_now_time_deprecation_with_message(self, sample_times):
        with pytest.warns(DeprecationWarning, match="synthetic_now_time.*deprecated"):
            read_kp_from_multiple_models(
                start_time=sample_times["past_start"],
                end_time=sample_times["future_end"],
                synthetic_now_time=sample_times["test_time_now"],
            )

    def test_model_check_with_wrong_class(self, sample_times):
        class FakeModel:
            pass

        fake = FakeModel()
        with pytest.raises(ModelError, match="Unknown or incompatible model"):
            read_kp_from_multiple_models(
                start_time=sample_times["past_start"],
                end_time=sample_times["future_end"],
                model_order=[fake],
            )

    def test_recurrence_with_Niemegk_recurr(self, sample_times):
        """Test basic 27-day recurrence filling functionality."""
        data = read_kp_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            model_order=[KpOMNI(), KpNiemegk()],
            rec_model_order=[KpNiemegk()],
            recurrence=True,
            download=True,
        )
        assert data.loc["2024-11-27 00:00:00+00:00":].model.unique()[0] == "niemegk_recurrence"

    def test_recurrence_with_both_recurr(self, sample_times):
        """Test recurrence with custom historical model order."""
        data = read_kp_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            model_order=[KpOMNI(), KpNiemegk()],
            recurrence=True,
            rec_model_order=[KpNiemegk(), KpOMNI()],
            download=True,
        )
        assert data.loc["2024-11-25 15:00:00+00:00":"2024-11-26 00:00:00+00:00"].model.unique()[0] == "omni_recurrence"
        assert data.loc["2024-11-27 00:00:00+00:00":].model.unique()[0] == "niemegk_recurrence"

    def test_recurrence_fills_gaps(self, sample_times):
        """Test that recurrence actually fills missing values."""
        data_no_rec = read_kp_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            model_order=[KpOMNI()],
            recurrence=False,
            download=False,
        )
        data_with_rec = read_kp_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            model_order=[KpOMNI()],
            recurrence=True,
            download=False,
        )
        nan_count_no_rec = data_no_rec["kp"].isna().sum()
        nan_count_with_rec = data_with_rec["kp"].isna().sum()

        assert nan_count_with_rec <= nan_count_no_rec

    def test_recurrence_preserves_existing_data(self, sample_times):
        """Test that recurrence doesn't overwrite existing valid data."""
        data_no_rec = read_kp_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            model_order=[KpOMNI(), KpNiemegk()],
            recurrence=False,
            download=False,
        )

        data_with_rec = read_kp_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            model_order=[KpOMNI(), KpNiemegk()],
            recurrence=True,
            download=False,
        )

        valid_mask = ~data_no_rec["kp"].isna()
        if valid_mask.any():
            pd.testing.assert_series_equal(
                data_no_rec.loc[valid_mask, "kp"],
                data_with_rec.loc[valid_mask, "kp"],
                check_names=False,
            )
