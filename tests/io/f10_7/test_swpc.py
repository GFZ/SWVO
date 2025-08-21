# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: S101

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import warnings

import pandas as pd
import pytest

from swvo.io.f10_7.swpc import F107SWPC

TEST_DATA_DIR = Path("test_data")
MOCK_DATA_PATH = TEST_DATA_DIR / "mock_f107"


class TestF107SWPC:
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        TEST_DATA_DIR.mkdir(exist_ok=True)
        MOCK_DATA_PATH.mkdir(exist_ok=True)

        yield

        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)

    @pytest.fixture
    def f107_instance(self):
        with patch.dict("os.environ", {F107SWPC.ENV_VAR_NAME: str(MOCK_DATA_PATH)}):
            instance = F107SWPC()
            return instance

    @pytest.fixture
    def sample_f107_data(self):
        return """:Product: Daily Solar Data            DSD.txt
    :Issued: 1425 UT 08 Nov 2024
    #
    #  Prepared by the U.S. Dept. of Commerce, NOAA, Space Weather Prediction Center
    #  Please send comments and suggestions to SWPC.Webmaster@noaa.gov
    #
    #                Last 30 Days Daily Solar Data
    #
    #                         Sunspot       Stanford GOES15
    #           Radio  SESC     Area          Solar  X-Ray  ------ Flares ------
    #           Flux  Sunspot  10E-6   New     Mean  Bkgd    X-Ray      Optical
    #  Date     10.7cm Number  Hemis. Regions Field  Flux   C  M  X  S  1  2  3
    #--------------------------------------------------------------------------- 
    2024 10 09  220    107     1920      0    -999      *   3  2  2  4  2  0  0
    2024 10 10  216    150     1460      2    -999      *   4  4  0  4  0  0  0
    2024 10 11  214    130     1445      0    -999      *   6  2  0  4  0  0  0
    """

    @pytest.fixture
    def mock_download_response(self, sample_f107_data):
        def mock_download(output):
            mock_file_path = Path(output) / F107SWPC.NAME_F107
            mock_file_path.parent.mkdir(exist_ok=True)
            with open(mock_file_path, "w") as f:
                f.write(sample_f107_data)

        return mock_download

    def test_initialization_with_env_var(self):
        with patch.dict("os.environ", {F107SWPC.ENV_VAR_NAME: str(MOCK_DATA_PATH)}):
            f107 = F107SWPC()
            assert f107.data_dir == MOCK_DATA_PATH

    def test_initialization_without_env_var(self):
        if F107SWPC.ENV_VAR_NAME in os.environ:
            del os.environ[F107SWPC.ENV_VAR_NAME]
        with pytest.raises(ValueError):
            F107SWPC()

    def test_get_processed_file_list(self, f107_instance):
        start_time = datetime(2020, 1, 1)
        end_time = datetime(2022, 12, 31)

        file_paths, time_intervals = f107_instance._get_processed_file_list(
            start_time, end_time
        )

        assert len(file_paths) == 3
        assert all(str(path).startswith(str(MOCK_DATA_PATH)) for path in file_paths)
        assert all(path.name.startswith("SWPC_F107_") for path in file_paths)
        assert len(time_intervals) == 3
        assert all(isinstance(interval, tuple) for interval in time_intervals)

    # @patch("wget.download")
    def test_download_and_process(self, f107_instance):
        # mock_wget.side_effect = mock_download_response

        f107_instance.download_and_process()

        date_now = datetime.now(timezone.utc)

        expected_files = list(MOCK_DATA_PATH.glob(f"SWPC_F107_*.csv"))

        assert (1<=len(expected_files) & len(expected_files)<=2)

        data = pd.read_csv(expected_files[0], parse_dates=["date"])

        assert "f107" in data.columns
        assert "date" in data.columns

    def test_process_single_file(self, f107_instance, sample_f107_data):
        test_file = MOCK_DATA_PATH / "test_f107.txt"
        test_file.parent.mkdir(exist_ok=True)

        with open(test_file, "w") as f:
            f.write(sample_f107_data)

        data = f107_instance._process_single_file(test_file)

        assert isinstance(data, pd.DataFrame)
        assert all(col in data.columns for col in ["date", "f107"])
        assert len(data) == 3

    def test_read_with_no_data(self, f107_instance):
        start_time = datetime(2020, 1, 1)
        end_time = datetime(2020, 12, 31)

        with warnings.catch_warnings(record=True) as w:
            df = f107_instance.read(start_time, end_time, download=False)

            assert "SWPC_F107_2020.csv not found" in str(w[-1].message)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 366
            assert all(df["f107"].isna())
            assert all(df["file_name"].isnull())

    def test_read_invalid_time_range(self, f107_instance):
        start_time = datetime(2020, 12, 31)
        end_time = datetime(2020, 1, 1)

        with pytest.raises(ValueError, match="start_time must be before end_time"):
            f107_instance.read(start_time, end_time)

    def test_read_with_existing_data(self, f107_instance):
        sample_data = pd.DataFrame(
            {
                "date": pd.date_range(start="2020-01-01", end="2020-12-31", freq="D"),
                "f107": range(366),
            }
        )

        file_path = MOCK_DATA_PATH / "SWPC_F107_2020.csv"
        sample_data.to_csv(file_path, index=False)

        start_time = datetime(2020, 6, 1)
        end_time = datetime(2020, 6, 30)

        data = f107_instance.read(start_time, end_time)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(col in data.columns for col in ["f107"])

    @pytest.mark.parametrize("test_year", [2019, 2025])
    def test_read_missing_years_warning(self, f107_instance, test_year):
        start_time = datetime(test_year, 1, 1)
        end_time = datetime(test_year, 12, 31)
        with warnings.catch_warnings(record=True) as w:
            f107_instance.read(start_time, end_time, download=False)
            assert f"SWPC_F107_{test_year}.csv not found" in str(w[-1].message)

    def test_data_update_with_existing_file(self, f107_instance):
        initial_data = pd.DataFrame(
            {
                "date": pd.date_range(start="2024-01-01", end="2024-01-05", freq="D"),
                "f107": range(5),
            }
        )

        file_path = MOCK_DATA_PATH / "SWPC_F107_2024.csv"
        initial_data.to_csv(file_path, index=False)

        f107_instance.download_and_process()

        updated_data = pd.read_csv(file_path, parse_dates=["date"])
        assert len(updated_data) >= len(initial_data)
        assert "f107" in updated_data.columns


    def test_read_with_partial_data(self, f107_instance):
        sample_data = pd.DataFrame(
            {
                "date": pd.date_range(start="2020-01-01", end="2020-12-31", freq="D"),
                "f107": range(366),
            }
        )

        file_path = MOCK_DATA_PATH / "SWPC_F107_2020.csv"
        sample_data.to_csv(file_path, index=False)

        start_time = datetime(2020, 12, 1)
        end_time = datetime(2021, 1, 31)

        with warnings.catch_warnings(record=True) as w:
            data = f107_instance.read(start_time, end_time)

            assert "SWPC_F107_2021.csv not found" in str(w[-1].message)
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            assert all(col in data.columns for col in ["f107"])
