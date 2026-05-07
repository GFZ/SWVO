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

from swvo.io.kp import KpSIDC

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_kp_sidc"


@pytest.mark.skip(reason="SIDC data source is currently unavailable, so these tests cannot be run.")
class TestKpSIDC:
    @pytest.fixture(scope="session", autouse=True)
    def setup_and_cleanup(self):
        TEST_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        yield

        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR, ignore_errors=True)

    @pytest.fixture
    def kp_sidc_instance(self):
        return KpSIDC(data_dir=DATA_DIR)

    def test_initialization_with_data_dir(self):
        instance = KpSIDC(data_dir=DATA_DIR)
        assert instance.data_dir == DATA_DIR
        assert instance.data_dir.exists()

    def test_initialization_without_env_var(self):
        if KpSIDC.ENV_VAR_NAME in os.environ:
            del os.environ[KpSIDC.ENV_VAR_NAME]
        with pytest.raises(
            ValueError,
            match=f"Necessary environment variable {KpSIDC.ENV_VAR_NAME} not set!",
        ):
            KpSIDC()

    def test_initialization_with_env_var(self):
        os.environ[KpSIDC.ENV_VAR_NAME] = str(DATA_DIR)
        instance = KpSIDC()
        assert instance.data_dir == DATA_DIR

    def test_get_processed_file_list(self):
        instance = KpSIDC(data_dir=DATA_DIR)
        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 5, 3)

        file_paths, time_intervals = instance._get_processed_file_list(start_time, end_time)

        assert len(file_paths) == 5
        assert len(time_intervals) == 5

        assert file_paths[0].name == "SIDC_KP_FORECAST_202401.csv"
        assert file_paths[-1].name == "SIDC_KP_FORECAST_202405.csv"

        assert time_intervals[0][0] == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert time_intervals[0][1] == datetime(2024, 1, 31, 23, 59, 59, tzinfo=timezone.utc)

    def test_download_and_process_current_month(self, kp_sidc_instance):
        current_time = datetime.now(timezone.utc)
        end_time = current_time + timedelta(days=2)

        kp_sidc_instance.download_and_process(current_time, end_time, reprocess_files=True)

        file_paths, _ = kp_sidc_instance._get_processed_file_list(current_time, end_time)

        for file_path in file_paths:
            if file_path.exists():
                df = pd.read_csv(file_path, names=["t", "kp"])
                assert len(df) > 0
                assert "t" in df.columns
                assert "kp" in df.columns

                valid_kps = df["kp"].dropna()
                assert valid_kps.min() >= 0
                assert valid_kps.max() <= 9

    def test_download_past_month(self, kp_sidc_instance):
        past_time = datetime.now(timezone.utc) - timedelta(days=32)
        end_time = past_time + timedelta(days=2)

        kp_sidc_instance.download_and_process(past_time, end_time)

        file_paths, _ = kp_sidc_instance._get_processed_file_list(past_time, end_time)

        for file_path in file_paths:
            assert file_path.exists()

    def test_read_with_download(self, kp_sidc_instance):
        current_time = datetime.now()
        end_time = current_time + timedelta(days=1)

        data = kp_sidc_instance.read(current_time, end_time, download=True)

        assert isinstance(data, pd.DataFrame)
        assert "kp" in data.columns
        assert "file_name" in data.columns
        assert isinstance(data.index, pd.DatetimeIndex)

        assert data.index[0] >= current_time.replace(tzinfo=timezone.utc) - timedelta(hours=3)
        assert data.index[-1] <= end_time.replace(tzinfo=timezone.utc) + timedelta(hours=3)

    def test_read_without_download_no_file(self, kp_sidc_instance):
        current_time = datetime.now()
        end_time = current_time + timedelta(days=1)

        if DATA_DIR.exists():
            for file in DATA_DIR.glob("**/SIDC_KP_FORECAST_*.csv"):
                file.unlink()

        data = kp_sidc_instance.read(current_time, end_time, download=False)
        assert isinstance(data, pd.DataFrame)
        assert data["kp"].isna().all()

    def test_process_single_file(self):
        instance = KpSIDC(data_dir=DATA_DIR)
        temp_dir = Path("./temp_test")
        temp_dir.mkdir(exist_ok=True)

        try:
            sample_data = """[
                        {
                            "data": {
                                "end_time": "2024-01-31T00:00:00Z",
                                "issue_time": "2024-01-30T12:39:59Z",
                                "start_time": "2024-01-30T21:00:00Z",
                                "value": 2
                            }
                        },
                        {
                            "data": {
                                "end_time": "2024-01-31T00:00:00Z",
                                "issue_time": "2024-01-29T12:34:56Z",
                                "start_time": "2024-01-30T21:00:00Z",
                                "value": 2
                            }
                        },
                        {
                            "data": {
                                "end_time": "2024-01-31T00:00:00Z",
                                "issue_time": "2024-01-28T12:30:03Z",
                                "start_time": "2024-01-30T21:00:00Z",
                                "value": 3
                            }
                        },
                        {
                            "data": {
                                "end_time": "2024-01-30T21:00:00Z",
                                "issue_time": "2024-01-30T12:39:59Z",
                                "start_time": "2024-01-30T18:00:00Z",
                                "value": 6
                            }
                        },
                        {
                            "data": {
                                "end_time": "2024-01-30T21:00:00Z",
                                "issue_time": "2024-01-29T12:34:56Z",
                                "start_time": "2024-01-30T18:00:00Z",
                                "value": 3
                            }
                        }]"""

            temp_file = temp_dir / "kp.json"
            with open(temp_file, "w") as f:
                f.write(sample_data)

            df = instance._process_single_file(temp_file)

            assert isinstance(df, pd.DataFrame)
            assert "Kp" in df.columns
            assert len(df) == 2

            expected_values = [
                6,
                2,
            ]

            for actual, expected in zip(df["Kp"].values, expected_values):
                assert np.isclose(actual, expected, atol=0.0)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_reprocess_files_flag(self, kp_sidc_instance):
        current_time = datetime.now(timezone.utc)
        end_time = current_time + timedelta(days=1)

        kp_sidc_instance.download_and_process(current_time, end_time, reprocess_files=True)

        file_paths, _ = kp_sidc_instance._get_processed_file_list(current_time, end_time)

        initial_data = None
        for file_path in file_paths:
            if file_path.exists():
                initial_data = pd.read_csv(file_path, names=["t", "kp"])
                break

        assert initial_data is not None

        kp_sidc_instance.download_and_process(current_time, end_time, reprocess_files=False)

        unchanged_data = pd.read_csv(file_path, names=["t", "kp"])
        pd.testing.assert_frame_equal(initial_data, unchanged_data)
