# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: S101,PLR2004

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swvo.io.hp.ensemble import Hp30Ensemble, Hp60Ensemble, HpEnsemble

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_hp_ensemble"


class TestHpEnsemble:
    @pytest.fixture(scope="session", autouse=True)
    def setup_and_cleanup(self):
        TEST_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        yield

        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR, ignore_errors=True)

    @pytest.fixture
    def hp30ensemble_instance(self):
        (DATA_DIR / "hp30").mkdir(exist_ok=True)
        return Hp30Ensemble(data_dir=DATA_DIR / "hp30")

    @pytest.mark.parametrize("instance_type,index_name", [("hp30", "hp30"), ("hp60", "hp60")])
    def test_initialization(self, instance_type, index_name):
        ensemble_dir = DATA_DIR / instance_type
        ensemble_dir.mkdir(exist_ok=True)

        ensemble_class = Hp30Ensemble if instance_type == "hp30" else Hp60Ensemble

        instance = ensemble_class(data_dir=ensemble_dir)
        assert instance.index == index_name
        assert instance.index_number == index_name[2:]
        assert instance.data_dir == ensemble_dir

    def test_initialization_without_env_var(self):
        if Hp30Ensemble.ENV_VAR_NAME in os.environ:
            del os.environ[Hp30Ensemble.ENV_VAR_NAME]
        with pytest.raises(
            ValueError,
            match=f"Necessary environment variable {Hp30Ensemble.ENV_VAR_NAME} not set!",
        ):
            Hp30Ensemble()

    def test_abc_instantiation(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class*"):
            HpEnsemble("hp45", data_dir=DATA_DIR)

    @pytest.mark.parametrize("instance_type,index_name", [("hp30", "hp30"), ("hp60", "hp60")])
    def test_read_with_ensemble_data(self, instance_type, index_name):
        ensemble_dir = DATA_DIR / instance_type
        ensemble_dir.mkdir(exist_ok=True)
        instance_class = Hp30Ensemble if instance_type == "hp30" else Hp60Ensemble
        instance = instance_class(data_dir=ensemble_dir)

        current_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        str_date = current_time.strftime("%Y%m%dT%H%M%S")

        for i in range(3):
            test_dates = pd.date_range(
                start=current_time,
                end=current_time + timedelta(days=3),
                freq=f"{instance.index_number}min",
            )
            df = pd.DataFrame(
                {
                    "t": test_dates.strftime("%Y-%m-%d %H:%M:%S"),
                    index_name: np.random.uniform(10, 20, size=len(test_dates)),
                }
            )

            filename = f"FORECAST_{index_name.title()}_{str_date}_ensemble_{i + 1}.csv"
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
            assert df.index[0] >= current_time.replace(tzinfo=timezone.utc) - timedelta(
                minutes=float(instance.index_number) - 0.01,
            )
            assert df.index[-1] <= current_time.replace(tzinfo=timezone.utc) + timedelta(days=1) + timedelta(
                minutes=float(instance.index_number) + 0.01,
            )

    def test_read_with_default_times(self, instance_type="hp30"):
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

    def make_csv_file(self, path, filename, times, values, index_name):
        """Helper to create a CSV file with time-value pairs."""
        df = pd.DataFrame({"Forecast Time": times, index_name: values})
        file = path / filename
        df.to_csv(file, header=False, index=False)
        return file

    @pytest.mark.parametrize("instance_type,index_name", [("hp30", "hp30"), ("hp60", "hp60")])
    def test_read_with_horizon_single_file(self, instance_type, index_name):
        ensemble_dir = DATA_DIR / instance_type
        ensemble_dir.mkdir(exist_ok=True)
        instance_class = Hp30Ensemble if instance_type == "hp30" else Hp60Ensemble
        instance = instance_class(data_dir=ensemble_dir)

        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=1)
        horizon = 1.0 if instance_type == "hp30" else 1

        str_date = start.strftime("%Y%m%dT%H0000")
        times = pd.date_range(start.replace(tzinfo=None), periods=10, freq=f"{instance.index_number}min")
        values = np.arange(10)

        self.make_csv_file(
            ensemble_dir,
            f"FORECAST_{index_name.title()}_{str_date}_ensemble_0.csv",
            times,
            values,
            index_name,
        )
        result = instance.read_with_horizon(start, end, horizon)

        assert isinstance(result, list)
        assert all(isinstance(df, pd.DataFrame) for df in result)
        assert not result[0][index_name].isna().all()
        assert (result[0]["horizon"] == horizon).all()
        assert "Forecast Time" in result[0].columns
        assert index_name in result[0].columns
        assert "source" in result[0].columns

    @pytest.mark.parametrize("instance_type,index_name", [("hp30", "hp30"), ("hp60", "hp60")])
    def test_read_with_horizon_multiple_ensembles(self, instance_type, index_name):
        ensemble_dir = DATA_DIR / instance_type
        ensemble_dir.mkdir(exist_ok=True)
        instance_class = Hp30Ensemble if instance_type == "hp30" else Hp60Ensemble
        instance = instance_class(data_dir=ensemble_dir)

        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=2)
        horizon = 2.0 if instance_type == "hp30" else 2

        str_date = start.strftime("%Y%m%dT%H0000")
        times = pd.date_range(start.replace(tzinfo=None), periods=10, freq=f"{instance.index_number}min")
        values1 = np.arange(10)
        values2 = np.arange(10, 20)

        self.make_csv_file(
            ensemble_dir, f"FORECAST_{index_name.title()}_{str_date}_ensemble_0.csv", times, values1, index_name
        )
        self.make_csv_file(
            ensemble_dir, f"FORECAST_{index_name.title()}_{str_date}_ensemble_1.csv", times, values2, index_name
        )

        result = instance.read_with_horizon(start, end, horizon)

        assert len(result) == 2
        assert all(index_name in df.columns for df in result)
        freq = "0.5h" if instance_type == "hp30" else "1h"
        expected_range = pd.date_range(start, end, freq=freq, tz=timezone.utc)
        for df in result:
            assert set(df.index) == set(expected_range)

    @pytest.mark.parametrize("instance_type,index_name", [("hp30", "hp30"), ("hp60", "hp60")])
    def test_read_with_horizon_nan_fill_for_missing_files(self, instance_type, index_name):
        ensemble_dir = DATA_DIR / instance_type
        ensemble_dir.mkdir(exist_ok=True)
        instance_class = Hp30Ensemble if instance_type == "hp30" else Hp60Ensemble
        instance = instance_class(data_dir=ensemble_dir)

        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=2)
        horizon = 1.0 if instance_type == "hp30" else 1

        str_date = start.strftime("%Y%m%dT%H0000")
        times = pd.date_range(start.replace(tzinfo=None), periods=10, freq=f"{instance.index_number}min")
        values = np.arange(10)
        self.make_csv_file(
            ensemble_dir, f"FORECAST_{index_name.title()}_{str_date}_ensemble_0.csv", times, values, index_name
        )

        result = instance.read_with_horizon(start, end, horizon)

        assert len(result) >= 1
        assert not result[0][index_name].isna().all()
        assert (result[0]["horizon"] == horizon).all()

    @pytest.mark.parametrize("instance_type,index_name", [("hp30", "hp30"), ("hp60", "hp60")])
    def test_read_with_horizon_correct_horizon_selection(self, instance_type, index_name):
        ensemble_dir = DATA_DIR / instance_type
        ensemble_dir.mkdir(exist_ok=True)
        instance_class = Hp30Ensemble if instance_type == "hp30" else Hp60Ensemble
        instance = instance_class(data_dir=ensemble_dir)

        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=2)

        horizons_to_test = [0.5, 1.0, 1.5] if instance_type == "hp30" else [1, 2]
        for horizon in horizons_to_test:
            for existing_file in ensemble_dir.glob("FORECAST_*.csv"):
                existing_file.unlink()

            str_date = start.strftime("%Y%m%dT%H0000")

            times = pd.date_range(start.replace(tzinfo=None), periods=50, freq=f"{instance.index_number}min")
            values = np.arange(len(times)) + horizon

            self.make_csv_file(
                ensemble_dir,
                f"FORECAST_{index_name.title()}_{str_date}_ensemble_0.csv",
                times,
                values,
                index_name,
            )

            result = instance.read_with_horizon(start, end, horizon)
            assert len(result) >= 1
            df = result[0]

            assert not df[index_name].isna().all()
            assert (df["horizon"] == horizon).all()

    @pytest.mark.parametrize("instance_type", ["hp30", "hp60"])
    def test_read_with_horizon_invalid_horizon(self, instance_type):
        ensemble_dir = DATA_DIR / instance_type
        ensemble_dir.mkdir(exist_ok=True)
        instance_class = Hp30Ensemble if instance_type == "hp30" else Hp60Ensemble
        instance = instance_class(data_dir=ensemble_dir)

        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=1)

        invalid_horizons = [-1, 73, 100]
        for horizon in invalid_horizons:
            with pytest.raises(ValueError, match="Horizon must be between 0 and 72 hours"):
                instance.read_with_horizon(start, end, horizon)

        if instance_type == "hp30":
            invalid_increments = [0.3, 1.7, 2.2]
            for horizon in invalid_increments:
                with pytest.raises(ValueError, match="Horizon for hp30 must be in 0.5 hour increments"):
                    instance.read_with_horizon(start, end, horizon)

        if instance_type == "hp60":
            invalid_increments = [0.5, 1.5, 2.3]
            for horizon in invalid_increments:
                with pytest.raises(ValueError, match="Horizon for hp60 must be in 1 hour increments"):
                    instance.read_with_horizon(start, end, horizon)

    @pytest.mark.parametrize("instance_type", ["hp30", "hp60"])
    def test_read_with_horizon_no_files(self, instance_type):
        ensemble_dir = DATA_DIR / instance_type
        ensemble_dir.mkdir(exist_ok=True)
        instance_class = Hp30Ensemble if instance_type == "hp30" else Hp60Ensemble
        instance = instance_class(data_dir=ensemble_dir)

        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=1)
        horizon = 1.0 if instance_type == "hp30" else 1

        for existing_file in ensemble_dir.glob("FORECAST_*.csv"):
            existing_file.unlink()

        with pytest.raises(FileNotFoundError, match="No ensemble data found"):
            instance.read_with_horizon(start, end, horizon)
