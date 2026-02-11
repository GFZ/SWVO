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

from swvo.io.kp import KpSWPC

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_kp_swpc"


class TestKpSWPC:
    @pytest.fixture(scope="session", autouse=True)
    def setup_and_cleanup(self):
        TEST_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        yield

        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR, ignore_errors=True)

    @pytest.fixture
    def kp_swpc_instance(self):
        return KpSWPC(data_dir=DATA_DIR)

    def test_initialization_with_data_dir(self):
        instance = KpSWPC(data_dir=DATA_DIR)
        assert instance.data_dir == DATA_DIR
        assert instance.data_dir.exists()

    def test_initialization_without_env_var(self):
        if KpSWPC.ENV_VAR_NAME in os.environ:
            del os.environ[KpSWPC.ENV_VAR_NAME]
        with pytest.raises(ValueError, match=f"Necessary environment variable {KpSWPC.ENV_VAR_NAME} not set!"):
            KpSWPC()

    def test_initialization_with_env_var(self):
        os.environ[KpSWPC.ENV_VAR_NAME] = str(DATA_DIR)
        instance = KpSWPC()
        assert instance.data_dir == DATA_DIR

    def test_download_and_process_current(self, kp_swpc_instance):
        current_date = datetime.now(timezone.utc)

        kp_swpc_instance.download_and_process(current_date, reprocess_files=True)

        yyyy = current_date.strftime("%Y")
        mm = current_date.strftime("%m")

        expected_file = (
            kp_swpc_instance.data_dir / yyyy / mm / f"SWPC_KP_FORECAST_{current_date.strftime('%Y%m%d')}.csv"
        )
        assert expected_file.exists()

        df = pd.read_csv(expected_file, names=["t", "kp"])
        assert len(df) > 0
        assert len(df) == 24
        assert "t" in df.columns
        assert "kp" in df.columns
        assert df["kp"].min() >= 0
        assert df["kp"].max() <= 9

        pd.to_datetime(df["t"])

    def test_download_past_date(self, kp_swpc_instance):
        past_date = datetime.now(timezone.utc) - timedelta(days=1)
        with pytest.raises(ValueError, match="We can only download and progress a Kp SWPC file for the current day!"):
            kp_swpc_instance.download_and_process(past_date)

    def test_read_with_download(self, kp_swpc_instance):
        current_time = datetime.now(timezone.utc)
        end_time = current_time + timedelta(days=1)

        data = kp_swpc_instance.read(current_time + timedelta(days=1), end_time, download=True)

        assert isinstance(data, pd.DataFrame)
        assert "kp" in data.columns
        assert "file_name" in data.columns
        assert isinstance(data.index, pd.DatetimeIndex)
        assert data.index[0] >= current_time - timedelta(hours=3)
        assert data.index[-1] <= end_time + timedelta(hours=3)
        assert data["kp"].min() >= 0
        assert data["kp"].max() <= 9
        assert data.index.tzinfo == timezone.utc

        decimals = (data["kp"] % 1).unique()
        for decimal in decimals:
            assert np.isclose(decimal, 0) or np.isclose(decimal, 1 / 3) or np.isclose(decimal, 2 / 3)

    def test_read_exceeding_three_days(self, kp_swpc_instance):
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(days=4)
        with pytest.raises(ValueError, match="We can only read 3 days at a time of Kp SWPC!"):
            kp_swpc_instance.read(start_time, end_time)

    def test_read_without_download_no_file(self, kp_swpc_instance):
        current_time = datetime.now(timezone.utc)
        end_time = current_time + timedelta(days=1)

        yyyy = current_time.strftime("%Y")
        mm = current_time.strftime("%m")

        expected_file = (
            kp_swpc_instance.data_dir / yyyy / mm / f"SWPC_KP_FORECAST_{current_time.strftime('%Y%m%d')}.csv"
        )
        if expected_file.exists():
            expected_file.unlink()

        data = kp_swpc_instance.read(current_time, end_time, download=False)
        assert data["kp"].isnull().all()

    def test_reprocess_files_flag(self, kp_swpc_instance):
        current_time = datetime.now(timezone.utc)
        yyyy = current_time.strftime("%Y")
        mm = current_time.strftime("%m")
        file_path = kp_swpc_instance.data_dir / yyyy / mm / f"SWPC_KP_FORECAST_{current_time.strftime('%Y%m%d')}.csv"

        kp_swpc_instance.download_and_process(current_time, reprocess_files=True)
        assert file_path.exists()

        initial_data = pd.read_csv(file_path, names=["t", "kp"])

        kp_swpc_instance.download_and_process(current_time, reprocess_files=False)
        assert file_path.exists()
        unchanged_data = pd.read_csv(file_path, names=["t", "kp"])
        pd.testing.assert_frame_equal(initial_data, unchanged_data)

    def test_read_default_end_time(self, kp_swpc_instance):
        current_time = datetime.now(timezone.utc)

        kp_swpc_instance.download_and_process(current_time, reprocess_files=True)

        data = kp_swpc_instance.read(current_time)

        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert "kp" in data.columns
        assert data.index[-1] <= current_time + timedelta(days=3) + timedelta(hours=3)
