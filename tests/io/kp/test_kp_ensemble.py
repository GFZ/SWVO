# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swvo.io.kp import KpEnsemble

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_kp_ensemble"


class TestKpEnsemble:
    @pytest.fixture(scope="session", autouse=True)
    def setup_and_cleanup(self):
        TEST_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        yield
        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR, ignore_errors=True)

    @pytest.fixture
    def kp_ensemble_instance(self):
        return KpEnsemble(data_dir=DATA_DIR)

    def test_initialization_with_data_dir(self):
        instance = KpEnsemble(data_dir=DATA_DIR)
        assert instance.data_dir == DATA_DIR

    def test_initialization_without_env_var(self):
        if KpEnsemble.ENV_VAR_NAME in os.environ:
            del os.environ[KpEnsemble.ENV_VAR_NAME]
        with pytest.raises(
            ValueError,
            match=f"Necessary environment variable {KpEnsemble.ENV_VAR_NAME} not set!",
        ):
            KpEnsemble()

    def test_initialization_with_env_var(self):
        os.environ[KpEnsemble.ENV_VAR_NAME] = str(DATA_DIR)
        instance = KpEnsemble()
        assert instance.data_dir == DATA_DIR

    def test_initialization_with_nonexistent_directory(self):
        with pytest.raises(FileNotFoundError):
            KpEnsemble(data_dir="nonexistent_directory")

    def test_read_with_ensemble_data(self, kp_ensemble_instance):
        current_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        str_date = current_time.strftime("%Y%m%dT%H0000")

        for i in range(3):
            test_dates = pd.date_range(start=current_time, end=current_time + timedelta(days=3), freq="3h")

            df = pd.DataFrame(
                {
                    "t": test_dates.strftime("%Y-%m-%d %H:%M:%S"),
                    "kp": np.random.uniform(0, 9, size=len(test_dates)),
                }
            )

            filename = f"FORECAST_Kp_{str_date}_ensemble_{i + 1}.csv"
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

    def test_read_with_default_times(self, kp_ensemble_instance):
        current_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        str_date = current_time.strftime("%Y%m%dT%H0000")

        test_dates = pd.date_range(start=current_time, end=current_time + timedelta(days=3), freq="3h")
        df = pd.DataFrame(
            {
                "t": test_dates.strftime("%Y-%m-%d %H:%M:%S"),
                "kp": np.random.uniform(0, 9, size=len(test_dates)),
            }
        )
        filename = f"FORECAST_Kp_{str_date}_ensemble_1.csv"
        file_path = kp_ensemble_instance.data_dir / filename
        df.to_csv(file_path, index=False, header=False)

        data = kp_ensemble_instance.read(None, None)

        assert isinstance(data, list)
        assert len(data) > 0
        assert isinstance(data[0], pd.DataFrame)
        assert all("kp" in i.columns for i in data)
        assert not data[0].empty

    def make_csv_file(self, path, filename, times, values):
        """Helper to create a CSV file with time-value pairs."""
        df = pd.DataFrame({"Forecast Time": times, "kp": values})
        file = path / filename
        df.to_csv(file, header=False, index=False)
        return file

    def test_read_with_horizon_single_file(self, kp_ensemble_instance, tmp_path):
        kp_ensemble_instance.data_dir = tmp_path

        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=3)
        horizon = 3

        str_date = start.strftime("%Y%m%dT%H0000")
        times = pd.date_range(start.replace(tzinfo=None), periods=10, freq="3h")
        values = np.arange(10)

        self.make_csv_file(
            tmp_path,
            f"FORECAST_Kp_{str_date}_ensemble_0.csv",
            times,
            values,
        )
        result = kp_ensemble_instance.read_with_horizon(start, end, horizon)

        assert isinstance(result, list)
        assert all(isinstance(df, pd.DataFrame) for df in result)
        assert not result[0]["kp"].isna().all()
        assert (result[0]["horizon"] == horizon).all()
        assert "Forecast Time" in result[0].columns
        assert "kp" in result[0].columns
        assert "source" in result[0].columns

    def test_read_with_horizon_multiple_ensembles(self, kp_ensemble_instance, tmp_path):
        kp_ensemble_instance.data_dir = tmp_path

        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=6)
        horizon = 6

        str_date = (start + timedelta(hours=-(horizon % 3))).strftime("%Y%m%dT%H0000")
        times = pd.date_range(start.replace(tzinfo=None), periods=10, freq="3h")
        values1 = np.arange(10)
        values2 = np.arange(10, 20)

        self.make_csv_file(tmp_path, f"FORECAST_Kp_{str_date}_ensemble_0.csv", times, values1)
        self.make_csv_file(tmp_path, f"FORECAST_Kp_{str_date}_ensemble_1.csv", times, values2)

        result = kp_ensemble_instance.read_with_horizon(start, end, horizon)

        assert len(result) == 2
        assert all("kp" in df.columns for df in result)
        for df in result:
            assert set(df.index) == set(pd.date_range(start, end, freq="3h", tz=timezone.utc))

    def test_read_with_horizon_no_files(self, kp_ensemble_instance, tmp_path):
        kp_ensemble_instance.data_dir = tmp_path

        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=3)
        horizon = 3

        result = kp_ensemble_instance.read_with_horizon(start, end, horizon)

        assert result == []

    def test_read_with_horizon_nan_fill_for_missing_files(self, kp_ensemble_instance, tmp_path):
        kp_ensemble_instance.data_dir = tmp_path

        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=6)
        horizon = 3

        str_date = start.strftime("%Y%m%dT%H0000")
        times = pd.date_range(start.replace(tzinfo=None), periods=10, freq="3h")
        values = np.arange(10)
        self.make_csv_file(tmp_path, f"FORECAST_Kp_{str_date}_ensemble_0.csv", times, values)

        result = kp_ensemble_instance.read_with_horizon(start, end, horizon)

        assert len(result) == 1
        assert not result[0]["kp"].isna().all()
        assert (result[0]["horizon"] == horizon).all()

    def test_read_with_horizon_correct_horizon_selection(self, kp_ensemble_instance, tmp_path):
        kp_ensemble_instance.data_dir = tmp_path

        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=6)

        horizons_to_test = [3, 25, 35]
        for horizon in horizons_to_test:
            file_time = start
            file_offset = -(horizon % 3)
            str_date = (file_time + timedelta(hours=file_offset)).strftime("%Y%m%dT%H0000")

            times = pd.date_range(start.replace(tzinfo=None), periods=50, freq="3h")
            values = np.arange(len(times)) + horizon

            self.make_csv_file(
                tmp_path,
                f"FORECAST_Kp_{str_date}_ensemble_0.csv",
                times,
                values,
            )

            result = kp_ensemble_instance.read_with_horizon(start, end, horizon)
            assert len(result) >= 1
            df = result[0]

            file_index = (horizon + 2) // 3
            expected_value = values[file_index]

            actual_value = df.loc[start, "kp"]
            assert actual_value == expected_value, (
                f"Expected {expected_value} but got {actual_value} for horizon {horizon}"
            )

    def test_read_with_horizon_invalid_horizon(self, kp_ensemble_instance):
        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=3)
        invalid_horizons = [-1, 73, 100]

        for horizon in invalid_horizons:
            with pytest.raises(ValueError, match="Horizon must be between 0 and 72 hours"):
                kp_ensemble_instance.read_with_horizon(start, end, horizon)
