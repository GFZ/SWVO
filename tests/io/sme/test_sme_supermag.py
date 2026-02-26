# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from swvo.io.sme import SMESuperMAG

TEST_DATA_DIR = Path("test_data")
MOCK_DATA_PATH = TEST_DATA_DIR / "mock_sme"
TEST_USERNAME = "swvo_test"

class TestSMESuperMAG:
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        TEST_DATA_DIR.mkdir(exist_ok=True)
        MOCK_DATA_PATH.mkdir(exist_ok=True)

        yield

        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)

    @pytest.fixture
    def sme_instance(self):
        with patch.dict("os.environ", {SMESuperMAG.ENV_VAR_NAME: str(MOCK_DATA_PATH)}):
            instance = SMESuperMAG(TEST_USERNAME)
            return instance

    @pytest.fixture
    def mock_sme_supermag_data(self):
        return """ [{"tval": 1668816000.000000, "SME": 263.221710},
{"tval": 1668816060.000000, "SME": 250.118866},
{"tval": 1668816120.000000, "SME": 234.960663},
{"tval": 1668816180.000000, "SME": 227.111343},
{"tval": 1668816240.000000, "SME": 220.567047},
{"tval": 1668816300.000000, "SME": 213.856384}]
    """
    
    def test_initialization_with_env_var(self):
        with patch.dict("os.environ", {SMESuperMAG.ENV_VAR_NAME: str(MOCK_DATA_PATH)}):
            sme = SMESuperMAG(TEST_USERNAME)
            assert sme.data_dir == MOCK_DATA_PATH

    def test_initialization_without_env_var(self):
        if SMESuperMAG.ENV_VAR_NAME in os.environ:
            del os.environ[SMESuperMAG.ENV_VAR_NAME]
        with pytest.raises(ValueError):
            SMESuperMAG(TEST_USERNAME)

    def test_get_processed_file_list(self, sme_instance):
        start_time = datetime(2020, 1, 1)
        end_time = datetime(2020, 2, 1)

        file_paths, time_intervals = sme_instance._get_processed_file_list(start_time, end_time)

        assert len(file_paths) == 32
        assert all(str(path).startswith(str(MOCK_DATA_PATH)) for path in file_paths)
        assert all(path.name.startswith("SuperMAG_SME_") for path in file_paths)
        assert len(time_intervals) == 32

    def test_download_and_process(self, sme_instance):
        sme_instance.download_and_process(datetime(2020, 1, 1), datetime(2020, 1, 2))

        expected_files = list(MOCK_DATA_PATH.glob("**/SuperMAG_SME_*.csv"))
        print(expected_files)

        assert 1 <= len(expected_files) & len(expected_files) <= 2

        data = pd.read_csv(expected_files[0])
        assert "sme" in data.columns

    def test_process_single_file(self, sme_instance, mock_sme_supermag_data):
        test_file = MOCK_DATA_PATH / "test_sme.txt"
        test_file.parent.mkdir(exist_ok=True)

        with open(test_file, "w") as f:
            f.write(mock_sme_supermag_data)

        data = sme_instance._process_single_file(test_file)

        assert isinstance(data, pd.DataFrame)
        assert "sme" in data.columns
        assert len(data) == 6

    def test_read_with_no_data(self, sme_instance):
        start_time = datetime(2020, 1, 1)
        end_time = datetime(2020, 1, 10)

        with warnings.catch_warnings(record=True) as w:
            df = sme_instance.read(start_time, end_time, download=False)

            assert "SuperMAG_SME_20200110.csv not found" in str(w[-1].message)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 9 * 24 * 60 + 1
            assert all(df["sme"].isna())
            assert all(df["file_name"].isnull())

    def test_read_invalid_time_range(self, sme_instance):
        start_time = datetime(2020, 12, 31)
        end_time = datetime(2020, 1, 1)

        with pytest.raises(AssertionError, match="Start time must be before end time!"):
            sme_instance.read(start_time, end_time)

    def test_read_with_existing_data(self, sme_instance):
        sample_data = pd.DataFrame(
            {"sme": range(1441)}, index=pd.date_range(start="2020-01-01", end="2020-01-02", freq="min")
        )
        sample_data.index.name = "timestamp"

        file_path = MOCK_DATA_PATH / "SuperMAG_SME_20200101.csv"
        sample_data.to_csv(file_path, index=True)

        start_time = datetime(2020, 1, 1, 12)
        end_time = datetime(2020, 1, 1, 18)

        data = sme_instance.read(start_time, end_time)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(col in data.columns for col in ["sme"])