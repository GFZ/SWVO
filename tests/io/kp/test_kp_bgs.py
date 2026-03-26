# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swvo.io.kp import KpBGS

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_kp_bgs"


class TestKpBGS:
    @pytest.fixture(scope="session", autouse=True)
    def setup_and_cleanup(self):
        TEST_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        yield

        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR, ignore_errors=True)

    @pytest.fixture
    def kp_bgs_instance(self):
        return KpBGS(data_dir=DATA_DIR)

    def test_initialization_with_data_dir(self):
        instance = KpBGS(data_dir=DATA_DIR)
        assert instance.data_dir == DATA_DIR
        assert instance.data_dir.exists()

    def test_initialization_without_env_var(self):
        if KpBGS.ENV_VAR_NAME in os.environ:
            del os.environ[KpBGS.ENV_VAR_NAME]
        with pytest.raises(
            ValueError,
            match=f"Necessary environment variable {KpBGS.ENV_VAR_NAME} not set!",
        ):
            KpBGS()

    def test_initialization_with_env_var(self):
        os.environ[KpBGS.ENV_VAR_NAME] = str(DATA_DIR)
        instance = KpBGS()
        assert instance.data_dir == DATA_DIR

    def test_get_processed_file_list(self):
        instance = KpBGS(data_dir=DATA_DIR)
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 5, 3)

        file_paths, time_intervals = instance._get_processed_file_list(start_time, end_time)

        assert len(file_paths) == 5
        assert len(time_intervals) == 5

        assert file_paths[0].name == "BGS_KP_FORECAST_202401.csv"
        assert file_paths[-1].name == "BGS_KP_FORECAST_202405.csv"

        assert time_intervals[0][0] == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert time_intervals[0][1] == datetime(2024, 1, 31, 23, 59, 59, tzinfo=timezone.utc)

    def test_download_and_process_current_month(self, kp_bgs_instance):
        current_time = datetime.now(timezone.utc)
        end_time = current_time + timedelta(days=2)

        kp_bgs_instance.download_and_process(current_time, reprocess_files=True)

        file_paths, _ = kp_bgs_instance._get_processed_file_list(current_time, end_time)

        for file_path in file_paths:
            if file_path.exists():
                df = pd.read_csv(file_path, names=["t", "kp"])
                assert len(df) > 0
                assert "t" in df.columns
                assert "kp" in df.columns

                valid_kps = df["kp"].dropna()
                assert valid_kps.min() >= 0
                assert valid_kps.max() <= 9

    def test_download_past_month(self, kp_bgs_instance):
        past_time = datetime.now(timezone.utc) - timedelta(days=32)
        end_time = past_time + timedelta(days=2)

        kp_bgs_instance.download_and_process(past_time)

        file_paths, _ = kp_bgs_instance._get_processed_file_list(past_time, end_time)

        for file_path in file_paths:
            assert not file_path.exists()  # Files are only available for 2 months past, so these should not exist

    def test_read_with_download(self, kp_bgs_instance):
        current_time = datetime.now()
        end_time = current_time + timedelta(days=1)

        data = kp_bgs_instance.read(current_time, end_time, download=True)

        assert isinstance(data, pd.DataFrame)
        assert "kp" in data.columns
        assert "file_name" in data.columns
        assert isinstance(data.index, pd.DatetimeIndex)

        assert data.index[0] >= current_time.replace(tzinfo=timezone.utc) - timedelta(hours=3)
        assert data.index[-1] <= end_time.replace(tzinfo=timezone.utc) + timedelta(hours=3)

    def test_read_without_download_no_file(self, kp_bgs_instance):
        current_time = datetime.now()
        end_time = current_time + timedelta(days=1)

        if DATA_DIR.exists():
            for file in DATA_DIR.glob("**/BGS_KP_FORECAST_*.csv"):
                file.unlink()

        data = kp_bgs_instance.read(current_time, end_time, download=False)
        assert isinstance(data, pd.DataFrame)
        assert data["kp"].isna().all()

    def test_process_single_file(self):
        instance = KpBGS(data_dir=DATA_DIR)
        df = instance._process_single_file("tests/io/kp/data/kp.html")

        assert isinstance(df, pd.DataFrame)
        assert "Kp" in df.columns
        assert len(df) == 248

        # fmt: off
        expected_values = [4.0, 5.333, 5.0, 6.0, 6.667, 8.0, 6.333, 4.0, 3.333, 2.0, 3.333, 4.667, 4.333, 3.667, 3.333, 3.0, 3.0, 3.333, 3.333, 2.0, 0.667, 1.333, 1.0, 1.667, 2.0, 4.667, 5.0, 4.333, 3.333, 4.333, 5.333, 3.0, 3.333, 2.333, 2.333, 3.333, 3.0, 4.333, 4.0, 2.667, 3.0, 3.667, 2.667, 3.0, 2.667, 3.0, 2.333, 2.333, 2.333, 2.667, 2.333, 2.667, 2.333, 3.333, 2.0, 2.0, 3.667, 2.333, 1.667, 0.0, 1.0, 2.333, 1.0, 0.667, 2.0, 1.0, 1.667, 2.0, 3.667, 2.667, 1.667, 1.333, 1.0, 3.333, 2.0, 2.667, 3.0, 1.333, 2.333, 4.0, 2.333, 1.0, 1.333, 1.0, 1.667, 3.0, 1.333, 0.667, 1.333, 1.667, 1.667, 1.667, 2.333, 1.333, 1.333, 1.667, 4.0, 2.333, 1.0, 2.333, 2.0, 2.333, 2.333, 2.0, 2.667, 2.333, 2.333, 2.0, 2.0, 2.667, 3.0, 2.667, 3.333, 3.667, 3.667, 2.667, 3.0, 3.333, 2.667, 1.667, 2.667, 3.333, 1.667, 2.333, 3.0, 3.0, 3.667, 2.667, 3.333, 4.0, 3.667, 3.0, 3.333, 3.0, 4.0, 4.0, 2.0, 2.0, 2.333, 2.0, 2.333, 2.0, 2.0, 2.0, 1.0, 2.333, 3.667, 2.667, 3.333, 3.0, 3.0, 3.667, 3.667, 3.333, 3.0, 3.0, 4.333, 3.0, 2.667, 3.667, 2.333, 3.0, 2.0, 1.667, 1.0, 2.667, 3.0, 2.333, 2.667, 2.0, 3.0, 2.333, 1.667, 1.333, 1.0, 3.0, 3.333, 1.667, 0.667, 1.0, 1.333, 3.667, 3.667, 2.333, 1.333, 1.333, 2.333, 1.333, 2.0, 1.0, 1.0, 2.333, 0.333, 0.0, 0.333, 0.333, 0.667, 1.0, 0.0, 0.667, 0.333, 0.667, 0.333, 0.333, 0.333, 0.0, 0.0, 0.333, 0.0, 0.0, 1.333, 1.667, 1.333, 2.667, 2.0, 2.0, 2.0, 1.333, 0.667, 2.333, 2.667, 2.667, 3.667, 2.667, 2.0, 1.0, 1.0, 2.0, 2.0, 1.667, 1.333, 1.333, 0.667, 0.667, 0.667, 0.333, 0.667, 1.333, 2.0, 2.667, 1.667] 
        # fmt: on

        for actual, expected in zip(df["Kp"].values, expected_values):
            assert np.isclose(actual, expected, atol=0.0)

    def test_reprocess_files_flag(self, kp_bgs_instance):
        current_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end_time = current_time + timedelta(days=60)

        kp_bgs_instance.download_and_process(current_time, reprocess_files=True)

        file_paths, _ = kp_bgs_instance._get_processed_file_list(current_time, end_time)

        initial_data = None
        for file_path in file_paths:
            if file_path.exists():
                initial_data = pd.read_csv(file_path, names=["t", "kp"])
                break

        assert initial_data is not None

        kp_bgs_instance.download_and_process(current_time, reprocess_files=False)

        unchanged_data = pd.read_csv(file_path, names=["t", "kp"])
        pd.testing.assert_frame_equal(initial_data, unchanged_data)
