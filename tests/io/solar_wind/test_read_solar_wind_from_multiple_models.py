# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swvo.io.exceptions import ModelError
from swvo.io.solar_wind import (
    AVERAGE_VALUES_TO_FILL,
    DSCOVR,
    SWACE,
    SWOMNI,
    SWSWIFTEnsemble,
    read_solar_wind_from_multiple_models,
)
from swvo.io.solar_wind.read_solar_wind_from_multiple_models import _interpolate_short_gaps

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))
READ_SW_MODULE = importlib.import_module("swvo.io.solar_wind.read_solar_wind_from_multiple_models")


class TestReadSolarWindFromMultipleModels:
    @pytest.fixture(scope="session", autouse=True)
    def set_env_var(self):
        ENV_VAR_NAMES = {
            "OMNI_HIGH_RES_STREAM_DIR": f"{str(DATA_DIR)}/OMNI",
            "SWIFT_ENSEMBLE_OUTPUT_DIR": f"{str(DATA_DIR)}/ensemble",
            "RT_SW_ACE_STREAM_DIR": f"{str(DATA_DIR)}/ACE_RT",
            "SW_DSCOVR_STREAM_DIR": f"{str(DATA_DIR)}/DSCOVR",
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

    @pytest.fixture
    def expected_columns(self):
        return [
            "proton_density",
            "speed",
            "bavg",
            "temperature",
            "bx_gsm",
            "by_gsm",
            "bz_gsm",
            "file_name",
        ]

    def test_basic_historical_read(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["past_end"],
            model_order=[SWOMNI(), DSCOVR(), SWACE()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert isinstance(data, pd.DataFrame)
        assert all(col in data.columns for col in expected_columns)
        assert data.loc["2024-11-20 13:23:00+00:00"].model == "dscovr"
        assert data.loc["2024-11-22 18:00:00+00:00"].model == "omni"
        # no ace since dscovr and ace files are same in the test data and dscovr is before ace in the model_order
        assert not data["file_name"].isna().all()

    def test_basic_forecast_read(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["future_start"],
            end_time=sample_times["future_end"],
            model_order=[SWSWIFTEnsemble()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )
        assert all(isinstance(d, pd.DataFrame) for d in data)
        assert all(not d["file_name"].isna().all() for d in data)

        assert all(all(d.model == "swift") for d in data)
        for d in data:
            assert all(col in d.columns for col in expected_columns)

    def test_full_ensemble(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["future_start"],
            end_time=sample_times["future_end"],
            model_order=[SWSWIFTEnsemble()],
            reduce_ensemble=None,
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        assert isinstance(data, list)
        assert len(data) > 1
        assert all(isinstance(d, pd.DataFrame) for d in data)
        assert all(not d["file_name"].isna().all() for d in data)
        for d in data:
            assert all(col in d.columns for col in expected_columns)

    def test_time_ordering_and_transition(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["future_end"],
            model_order=[SWOMNI(), DSCOVR(), SWACE(), SWSWIFTEnsemble()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        for d in data:
            assert d.index.is_monotonic_increasing
            assert d.loc["2024-11-20 00:00:00+00:00"].model == "omni"
            assert d.loc["2024-11-25 00:01:00+00:00"].model == "swift"
            assert all(col in d.columns for col in expected_columns)

    def test_forecast_in_past(self, sample_times, expected_columns):
        data = read_solar_wind_from_multiple_models(
            start_time=sample_times["past_start"],
            end_time=sample_times["past_end"],
            model_order=[SWOMNI(), DSCOVR(), SWACE(), SWSWIFTEnsemble()],
            historical_data_cutoff_time=sample_times["test_time_now"] - timedelta(days=3),
        )

        for d in data:
            assert d.index.is_monotonic_increasing
            assert d.loc["2024-11-22 00:01:00+00:00"].model == "swift"
            assert d.loc["2024-11-22 00:00:00+00:00"].model == "omni"
            assert all(col in d.columns for col in expected_columns)

    def test_time_boundaries(self, sample_times):
        start = sample_times["past_start"]
        end = sample_times["future_end"]

        data = read_solar_wind_from_multiple_models(
            start_time=start,
            end_time=end,
            model_order=[SWOMNI(), SWSWIFTEnsemble()],
            historical_data_cutoff_time=sample_times["test_time_now"],
        )

        for d in data:
            assert d.index.min() >= start
            assert d.index.max() <= end

    def test_invalid_time_range(self, sample_times):
        with pytest.raises(ValueError):
            read_solar_wind_from_multiple_models(
                start_time=sample_times["future_end"],
                end_time=sample_times["past_start"],
                model_order=[SWOMNI()],
            )

    def test_data_consistency(self, sample_times):
        params = {
            "start_time": sample_times["past_start"],
            "end_time": sample_times["future_start"],
            "model_order": [SWOMNI(), DSCOVR(), SWACE(), SWSWIFTEnsemble()],
            "historical_data_cutoff_time": sample_times["test_time_now"],
        }

        data1 = read_solar_wind_from_multiple_models(**params)
        data2 = read_solar_wind_from_multiple_models(**params)

        for d1, d2 in zip(data1, data2):
            pd.testing.assert_frame_equal(d1, d2)

    def test_model_check_with_wrong_class(self, sample_times):
        class FakeModel:
            pass

        fake = FakeModel()
        with pytest.raises(ModelError, match="Unknown or incompatible model"):
            read_solar_wind_from_multiple_models(
                start_time=sample_times["past_start"],
                end_time=sample_times["future_end"],
                model_order=[fake],
            )

    @pytest.mark.parametrize(("fill_average"), [False, True])
    def test_fill_modes_do_not_truncate_final_dataframe(self, monkeypatch, fill_average):
        start_time = datetime(2024, 11, 25, 0, 0, tzinfo=timezone.utc)
        historical_data_cutoff_time = start_time + timedelta(minutes=5)
        end_time = start_time + timedelta(minutes=10)
        index = pd.date_range(start_time, end_time, freq="1min", tz="UTC")
        data = pd.DataFrame(
            {
                "speed": [400.0] * 6 + [np.nan] * 5,
                "model": ["omni"] * 6 + [None] * 5,
                "file_name": ["test_file.txt"] * 6 + [None] * 5,
            },
            index=index,
        )

        monkeypatch.setattr(READ_SW_MODULE, "_read_from_model", lambda *args, **kwargs: data)

        result = read_solar_wind_from_multiple_models(
            start_time=start_time,
            end_time=end_time,
            model_order=[SWOMNI(), SWSWIFTEnsemble()],
            historical_data_cutoff_time=historical_data_cutoff_time,
            fill_average=fill_average,
        )

        assert result.index.max() == end_time

    def test_average_fill_uses_expected_values(self, monkeypatch):
        start_time = datetime(2024, 11, 25, 0, 0, tzinfo=timezone.utc)
        historical_data_cutoff_time = start_time + timedelta(minutes=5)
        end_time = start_time + timedelta(minutes=10)
        index = pd.date_range(start_time, end_time, freq="1min", tz="UTC")
        average_values = AVERAGE_VALUES_TO_FILL
        data = pd.DataFrame(
            {
                **{col: [1.0] * 6 + [np.nan] * 5 for col in average_values},
                "model": ["omni"] * 6 + [None] * 5,
                "file_name": ["test_file.txt"] * 6 + [None] * 5,
            },
            index=index,
        )

        monkeypatch.setattr(READ_SW_MODULE, "_read_from_model", lambda *args, **kwargs: data)

        result = read_solar_wind_from_multiple_models(
            start_time=start_time,
            end_time=end_time,
            model_order=[SWOMNI()],
            historical_data_cutoff_time=historical_data_cutoff_time,
            fill_average=True,
        )

        future_mask = result.index > historical_data_cutoff_time
        assert result.index.max() == end_time
        for col, avg_value in average_values.items():
            assert result.loc[historical_data_cutoff_time, col] == 1.0
            np.testing.assert_allclose(result.loc[future_mask, col].to_numpy(), avg_value)
        assert (result.loc[future_mask, "model"] == "10_year_average_filled").all()
        assert (result.loc[future_mask, "file_name"] == "10_year_average_filled").all()

    def test_3_hour_interpolation(self, sample_times, expected_columns):
        # Use a longer time range to increase chances of gaps that need interpolation
        extended_start = sample_times["past_start"] - timedelta(days=2)
        extended_end = sample_times["past_end"] + timedelta(days=1)

        data_no_rec = read_solar_wind_from_multiple_models(
            start_time=extended_start,
            end_time=extended_end,
            model_order=[SWOMNI(), DSCOVR(), SWACE()],
            historical_data_cutoff_time=sample_times["test_time_now"],
            download=False,
            do_interpolation=True,
        )

        data_with_rec = read_solar_wind_from_multiple_models(
            start_time=extended_start,
            end_time=extended_end,
            model_order=[SWOMNI(), DSCOVR(), SWACE()],
            historical_data_cutoff_time=sample_times["test_time_now"],
            download=True,
            do_interpolation=True,
        )

        nan_count_no_rec = data_no_rec.isna().sum()
        nan_count_with_rec = data_with_rec.isna().sum()

        for i in expected_columns:
            assert nan_count_with_rec[i] <= nan_count_no_rec[i]

    def test_ensemble_reduction_methods(self, sample_times, expected_columns):
        reduction_methods = [None, "mean", "median"]

        for method in reduction_methods:
            data = read_solar_wind_from_multiple_models(
                start_time=sample_times["future_start"],
                end_time=sample_times["future_end"],
                model_order=[SWSWIFTEnsemble()],
                reduce_ensemble=method,
                historical_data_cutoff_time=sample_times["test_time_now"],
            )

            if method is None:
                assert isinstance(data, list)
                assert len(data) > 1
                for d in data:
                    assert all(col in d.columns for col in expected_columns)
            else:
                assert isinstance(data, pd.DataFrame)
                assert all(col in data.columns for col in expected_columns)


class TestInterpolateShortGaps:
    @pytest.fixture
    def sample_dataframe_with_short_gap(self):
        np.random.seed(42)
        times = pd.date_range("2024-01-01 00:00:00", periods=480, freq="1min", tz=timezone.utc)
        n_points = len(times)
        data = {
            "proton_density": np.linspace(5.0, 15.0, n_points) + np.random.normal(0, 0.5, n_points),
            "speed": np.linspace(400.0, 600.0, n_points) + np.random.normal(0, 20, n_points),
            "bavg": np.linspace(8.0, 18.0, n_points) + np.random.normal(0, 1, n_points),
            "temperature": np.linspace(80000, 120000, n_points) + np.random.normal(0, 5000, n_points),
            "bx_gsm": np.sin(np.linspace(0, 4 * np.pi, n_points)) * 5 + np.random.normal(0, 0.5, n_points),
            "by_gsm": np.cos(np.linspace(0, 3 * np.pi, n_points)) * 3 + np.random.normal(0, 0.3, n_points),
            "bz_gsm": np.linspace(-5, 5, n_points) + np.random.normal(0, 1, n_points),
            "model": ["omni"] * n_points,
            "file_name": ["test_file.txt"] * n_points,
        }
        df = pd.DataFrame(data, index=times)

        gap_start = 150
        gap_end = 270
        numeric_cols = ["proton_density", "speed", "bavg", "temperature", "bx_gsm", "by_gsm", "bz_gsm"]
        df.loc[df.index[gap_start:gap_end], numeric_cols] = np.nan

        return df

    @pytest.fixture
    def sample_dataframe_with_long_gap(self):
        np.random.seed(123)
        times = pd.date_range("2024-01-01 00:00:00", periods=600, freq="1min", tz=timezone.utc)

        n_points = len(times)
        data = {
            "proton_density": np.linspace(3.0, 12.0, n_points) + np.random.normal(0, 0.8, n_points),
            "speed": np.linspace(350.0, 550.0, n_points) + np.random.normal(0, 25, n_points),
            "bavg": np.linspace(6.0, 16.0, n_points) + np.random.normal(0, 1.2, n_points),
            "temperature": np.linspace(70000, 130000, n_points) + np.random.normal(0, 8000, n_points),
            "bx_gsm": np.sin(np.linspace(0, 6 * np.pi, n_points)) * 4 + np.random.normal(0, 0.7, n_points),
            "by_gsm": np.cos(np.linspace(0, 4 * np.pi, n_points)) * 2.5 + np.random.normal(0, 0.4, n_points),
            "bz_gsm": np.linspace(-8, 8, n_points) + np.random.normal(0, 1.5, n_points),
            "model": ["omni"] * n_points,
            "file_name": ["test_file.txt"] * n_points,
        }
        df = pd.DataFrame(data, index=times)

        gap_start = 180
        gap_end = 420
        numeric_cols = ["proton_density", "speed", "bavg", "temperature", "bx_gsm", "by_gsm", "bz_gsm"]
        df.loc[df.index[gap_start:gap_end], numeric_cols] = np.nan

        return df

    @pytest.fixture
    def sample_dataframe_no_gaps(self):
        np.random.seed(456)

        times = pd.date_range("2024-01-01 00:00:00", periods=120, freq="1min", tz=timezone.utc)

        n_points = len(times)
        data = {
            "proton_density": np.linspace(4.0, 8.0, n_points) + np.random.normal(0, 0.3, n_points),
            "speed": np.linspace(380.0, 480.0, n_points) + np.random.normal(0, 15, n_points),
            "bavg": np.linspace(9.0, 13.0, n_points) + np.random.normal(0, 0.8, n_points),
            "temperature": np.linspace(85000, 105000, n_points) + np.random.normal(0, 3000, n_points),
            "bx_gsm": np.sin(np.linspace(0, 2 * np.pi, n_points)) * 3 + np.random.normal(0, 0.4, n_points),
            "by_gsm": np.cos(np.linspace(0, np.pi, n_points)) * 2 + np.random.normal(0, 0.2, n_points),
            "bz_gsm": np.linspace(-3, 3, n_points) + np.random.normal(0, 0.8, n_points),
            "model": ["omni"] * n_points,
            "file_name": ["test_file.txt"] * n_points,
        }
        df = pd.DataFrame(data, index=times)
        return df

    def test_interpolate_short_gap_success(self, sample_dataframe_with_short_gap):
        gap_start, gap_end = 150, 270
        original_before = sample_dataframe_with_short_gap["proton_density"].iloc[gap_start - 1]
        original_after = sample_dataframe_with_short_gap["proton_density"].iloc[gap_end]

        result = _interpolate_short_gaps(sample_dataframe_with_short_gap, max_gap_minutes=180)

        gap_slice = slice(gap_start, gap_end)
        assert not result["proton_density"].iloc[gap_slice].isna().any()
        assert not result["speed"].iloc[gap_slice].isna().any()
        assert not result["bavg"].iloc[gap_slice].isna().any()

        assert all(result["file_name"].iloc[gap_slice] == "interpolated")
        assert all(result["model"].iloc[gap_slice] == "interpolated")

        assert result["proton_density"].iloc[0] == sample_dataframe_with_short_gap["proton_density"].iloc[0]
        assert result["proton_density"].iloc[gap_start - 1] == original_before
        assert result["proton_density"].iloc[gap_end] == original_after

    def test_long_gap_not_interpolated(self, sample_dataframe_with_long_gap):
        result = _interpolate_short_gaps(sample_dataframe_with_long_gap, max_gap_minutes=180)

        gap_start, gap_end = 180, 420
        gap_slice = slice(gap_start, gap_end)
        assert result["proton_density"].iloc[gap_slice].isna().all()
        assert result["speed"].iloc[gap_slice].isna().all()
        assert result["bavg"].iloc[gap_slice].isna().all()

        assert result["file_name"].iloc[gap_start] == "test_file.txt"

    def test_no_gaps_unchanged(self, sample_dataframe_no_gaps):
        result = _interpolate_short_gaps(sample_dataframe_no_gaps)

        pd.testing.assert_frame_equal(result, sample_dataframe_no_gaps)

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        result = _interpolate_short_gaps(empty_df)

        assert result.empty
        pd.testing.assert_frame_equal(result, empty_df)


class TestRecursiveFill27dHistorical:
    @pytest.fixture(scope="session", autouse=True)
    def set_env_var(self):
        ENV_VAR_NAMES = {
            "OMNI_HIGH_RES_STREAM_DIR": f"{str(DATA_DIR)}",
            "RT_SW_ACE_STREAM_DIR": f"{str(DATA_DIR)}/ACE_RT",
            "SW_DSCOVR_STREAM_DIR": f"{str(DATA_DIR)}/DSCOVR",
        }

        for key, var in ENV_VAR_NAMES.items():
            os.environ[key] = ENV_VAR_NAMES[key]

    @pytest.fixture
    def sample_dataframe_with_gaps(self):
        current_time = datetime(2024, 11, 25, tzinfo=timezone.utc)
        times = pd.date_range(current_time, periods=1440, freq="1min", tz=timezone.utc)

        data = {
            "proton_density": [np.nan] * 1440,
            "speed": [np.nan] * 1440,
            "bavg": [np.nan] * 1440,
            "temperature": [np.nan] * 1440,
            "bx_gsm": [np.nan] * 1440,
            "by_gsm": [np.nan] * 1440,
            "bz_gsm": [np.nan] * 1440,
            "model": [None] * 1440,
            "file_name": [None] * 1440,
        }
        df = pd.DataFrame(data, index=times)
        return df

    @pytest.fixture
    def sample_dataframe_no_gaps(self):
        current_time = datetime(2024, 11, 25, tzinfo=timezone.utc)
        times = pd.date_range(current_time, periods=5, freq="1min", tz=timezone.utc)

        data = {
            "proton_density": [5.0, 6.0, 7.0, 8.0, 9.0],
            "speed": [400.0, 410.0, 420.0, 430.0, 440.0],
            "bavg": [10.0, 11.0, 12.0, 13.0, 14.0],
            "model": ["omni"] * 5,
            "file_name": ["test_file.txt"] * 5,
        }
        df = pd.DataFrame(data, index=times)
        return df
