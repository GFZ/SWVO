# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: S101

import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from swvo.io.dst.wdc import DSTWDC

TEST_DATA_DIR = Path("test_data")
MOCK_DATA_PATH = TEST_DATA_DIR / "mock_dst"


class TestDSTWDC:
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        TEST_DATA_DIR.mkdir(exist_ok=True)
        MOCK_DATA_PATH.mkdir(exist_ok=True)

        yield

        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)

    @pytest.fixture
    def dst_instance(self):
        with patch.dict("os.environ", {DSTWDC.ENV_VAR_NAME: str(MOCK_DATA_PATH)}):
            instance = DSTWDC()
            return instance

    @pytest.fixture
    def sample_dst_data(self):
        return """      unit=nT                                                                                      UT
      1   2   3   4   5   6   7   8    9  10  11  12  13  14  15  16   17  18  19  20  21  22  23  24
DAY
 1    6   8   6   8   8  10  11  14   12  10  11  10   7   6   8  10   11  15  21  27  25  23  23  22
 2   19   0 -12 -11 -16 -17 -17 -17  -26 -28 -28 -23 -26 -21 -14 -14  -14  -8  -7 -11 -19 -22 -29 -32
 3  -31 -36 -34 -26 -19 -11 -10  -2   -3   2   6   3  -5 -19 -26 -21  -10  -5  -8 -25 -33 -19 -10 -12

<!-- vvvvv S yyyymm_part3.html vvvvv -->
</pre>
    """

    @pytest.fixture
    def mock_download_response(self, sample_dst_data):
        def mock_download(output):
            mock_file_path = Path(output) / DSTWDC.NAME_dst
            mock_file_path.parent.mkdir(exist_ok=True)
            with open(mock_file_path, "w") as f:
                f.write(sample_dst_data)

        return mock_download

    def test_initialization_with_env_var(self):
        with patch.dict("os.environ", {DSTWDC.ENV_VAR_NAME: str(MOCK_DATA_PATH)}):
            dst = DSTWDC()
            assert dst.data_dir == MOCK_DATA_PATH

    def test_initialization_without_env_var(self):
        if DSTWDC.ENV_VAR_NAME in os.environ:
            del os.environ[DSTWDC.ENV_VAR_NAME]
        with pytest.raises(ValueError):
            DSTWDC()

    def test_get_processed_file_list(self, dst_instance):
        start_time = datetime(2020, 1, 1)
        end_time = datetime(2022, 12, 31)

        file_paths, time_intervals = dst_instance._get_processed_file_list(start_time, end_time)

        assert len(file_paths) == 36
        assert all(str(path).startswith(str(MOCK_DATA_PATH)) for path in file_paths)
        assert all(path.name.startswith("WDC_DST_") for path in file_paths)
        assert len(time_intervals) == 36

    def test_download_and_process(self, dst_instance):
        dst_instance.download_and_process(datetime(2025, 9, 1), datetime(2025, 9, 1))

        expected_files = list(MOCK_DATA_PATH.glob("WDC_DST_*.csv"))

        assert 1 <= len(expected_files) & len(expected_files) <= 2

        data = pd.read_csv(expected_files[0])
        assert "dst" in data.columns

    def test_process_single_file(self, dst_instance, sample_dst_data):
        test_file = MOCK_DATA_PATH / "test_dst.txt"
        test_file.parent.mkdir(exist_ok=True)

        with open(test_file, "w") as f:
            f.write(sample_dst_data)

        data = dst_instance._process_single_file(test_file, 2025, 1)

        assert isinstance(data, pd.DataFrame)
        assert "dst" in data.columns
        assert len(data) == 72

    def test_read_with_no_data(self, dst_instance):
        start_time = datetime(2020, 1, 1)
        end_time = datetime(2020, 12, 31)

        with warnings.catch_warnings(record=True) as w:
            df = dst_instance.read(start_time, end_time, download=False)

            assert "WDC_DST_202012.csv not found" in str(w[-1].message)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 8761
            assert all(df["dst"].isna())
            assert all(df["file_name"].isnull())

    def test_read_invalid_time_range(self, dst_instance):
        start_time = datetime(2020, 12, 31)
        end_time = datetime(2020, 1, 1)

        with pytest.raises(AssertionError, match="Start time must be before end time!"):
            dst_instance.read(start_time, end_time)

    def test_read_with_existing_data(self, dst_instance):
        sample_data = pd.DataFrame(
            {
                "date": pd.date_range(start="2020-01-01", end="2020-12-31", freq="D"),
                "dst": range(366),
            }
        )

        file_path = MOCK_DATA_PATH / "SWPC_dst_2020.csv"
        sample_data.to_csv(file_path, index=False)

        start_time = datetime(2020, 6, 1)
        end_time = datetime(2020, 6, 30)

        data = dst_instance.read(start_time, end_time)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(col in data.columns for col in ["dst"])

    @pytest.mark.parametrize("test_year", [2019, 2025])
    def test_read_missing_years_warning(self, dst_instance, test_year):
        start_time = datetime(test_year, 1, 1)
        end_time = datetime(test_year, 1, 31)
        with warnings.catch_warnings(record=True) as w:
            dst_instance.read(start_time, end_time, download=False)
            assert f"WDC_DST_{test_year}01.csv not found" in str(w[-1].message)
