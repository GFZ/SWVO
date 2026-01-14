# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from swvo.io.solar_wind import SWOMNI

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))


class TestSWOMNI:
    @pytest.fixture
    def swomni(self):
        os.environ["OMNI_HIGH_RES_STREAM_DIR"] = str(DATA_DIR)
        yield SWOMNI()

    @pytest.fixture
    def mock_swomni_data(self):
        """Create mock solar wind data with all expected columns"""
        test_dates = pd.date_range(
            start=datetime(2020, 1, 1),
            end=datetime(2020, 12, 31, 23, 59, 0),
            freq="min"
        )
        test_data = pd.DataFrame(
            {
                "bavg": [5.0] * len(test_dates),
                "bx_gsm": [1.0] * len(test_dates),
                "by_gsm": [2.0] * len(test_dates),
                "bz_gsm": [-3.0] * len(test_dates),
                "speed": [400.0] * len(test_dates),
                "proton_density": [5.0] * len(test_dates),
                "temperature": [100000.0] * len(test_dates),
                "pdyn": [2.0] * len(test_dates),
                "file_name": "some_file",
            }
        )
        test_data.index = test_dates.tz_localize("UTC")
        return test_data

    def test_initialization_with_env_var(self, swomni):
        assert swomni.data_dir.exists()

    def test_initialization_with_data_dir(self):
        swomni = SWOMNI(data_dir=DATA_DIR)
        assert swomni.data_dir == DATA_DIR

    def test_initialization_without_env_var(self):
        if "OMNI_HIGH_RES_STREAM_DIR" in os.environ:
            del os.environ["OMNI_HIGH_RES_STREAM_DIR"]
        with pytest.raises(ValueError):
            SWOMNI()

    def test_download_and_process(self, swomni):
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)
        # download this file without mocking
        swomni.download_and_process(start_time, end_time)

        assert (DATA_DIR / "OMNI_HIGH_RES_1min_2020.csv").exists()

    def test_read_without_download(self, swomni):
        start_time = datetime(2021, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2021, 2, 28, tzinfo=timezone.utc)
        with warnings.catch_warnings(record=True) as w:
            df = swomni.read(start_time, end_time, download=False)
            # Check for warning about missing file
            warning_messages = [str(warning.message) for warning in w]
            assert any("not found" in msg for msg in warning_messages), "Expected 'not found' warning"
            
            assert not df.empty
            
            # Check that solar wind columns exist (not sym-h, that's separate)
            expected_cols = ["bavg", "bx_gsm", "by_gsm", "bz_gsm", "speed", 
                           "proton_density", "temperature", "pdyn"]
            for col in expected_cols:
                assert col in df.columns, f"Column {col} missing from solar wind data"
            
            # All values should be NaN when file doesn't exist
            for col in expected_cols:
                assert all(df[col].isna())
            assert all(df["file_name"].isnull())

    def test_read_with_download(self, swomni, mock_swomni_data, mocker):
        mocker.patch("pathlib.Path.exists", return_value=False)
        mocker.patch.object(swomni, "_read_single_file", return_value=mock_swomni_data)
        mocker.patch.object(swomni, "download_and_process")

        start_time = datetime(2020, 1, 1)
        end_time = datetime(2020, 12, 31)

        df = swomni.read(start_time, end_time, download=True)
        swomni.download_and_process.assert_called_once()

        assert not df.empty
        
        # Check all expected columns exist
        expected_cols = ["bavg", "bx_gsm", "by_gsm", "bz_gsm", "speed", 
                        "proton_density", "temperature", "pdyn"]
        for col in expected_cols:
            assert col in df.columns, f"Column {col} missing"
        
        # Check values are correct
        assert all(df["bavg"] == 5.0)
        assert all(df["bz_gsm"] == -3.0)
        assert all(df["speed"] == 400.0)
        
        # Check timezone
        assert all(idx.tzinfo is not None for idx in df.index)
        assert all(idx.tzinfo is timezone.utc for idx in df.index)

    def test_invalid_cadence(self, swomni):
        start_time = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2022, 12, 31, tzinfo=timezone.utc)

        with pytest.raises(AssertionError):
            swomni.read(start_time, end_time, cadence_min=2)

        with pytest.raises(AssertionError):
            swomni.download_and_process(start_time, end_time, cadence_min=10)

    def test_read_single_file(self, swomni):
        """Test reading actual CSV file"""
        csv_file = DATA_DIR / "OMNI_HIGH_RES_1min_2020.csv"
        df = swomni._read_single_file(csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Check all expected solar wind columns exist
        expected_cols = ["bavg", "bx_gsm", "by_gsm", "bz_gsm", "speed", 
                        "proton_density", "temperature", "pdyn"]
        for col in expected_cols:
            assert col in df.columns, f"Column {col} missing from solar wind data"

    def test_year_transition(self, swomni):
        start_time = datetime(2012, 12, 31, 23, 50, 0, tzinfo=timezone.utc)
        end_time = datetime(2013, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        result_df = swomni.read(start_time, end_time, download=False)

        assert result_df.index.min() == pd.Timestamp("2012-12-31 23:50:00+00:00")
        assert result_df.index.max() == pd.Timestamp("2013-01-01 00:00:00+00:00")
        
        # Verify solar wind columns exist
        expected_cols = ["bavg", "bx_gsm", "by_gsm", "bz_gsm", "speed", 
                        "proton_density", "temperature", "pdyn"]
        for col in expected_cols:
            assert col in result_df.columns

    def test_remove_processed_file(self):
        test_file = DATA_DIR / "OMNI_HIGH_RES_1min_2020.csv"
        if test_file.exists():
            os.remove(test_file)