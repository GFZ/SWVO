# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from pathlib import Path
from datetime import datetime, timezone
from data_management.io.f10_7 import F107OMNI
import pandas as pd
import warnings
from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))


class TestF107OMNI:
    @pytest.fixture
    def f107omni(self):
        os.environ["OMNI_LOW_RES_STREAM_DIR"] = str(DATA_DIR)
        yield F107OMNI()

    @pytest.fixture
    def mock_f107omni_data(self):
        test_dates = pd.date_range(
            start=datetime(2020, 1, 1), end=datetime(2020, 12, 31), freq="h"
        )
        test_data = pd.DataFrame(
            {
                "t": test_dates,
                "f107": [150.0] * len(test_dates),
                "file_name": "some_file",
                "timestamp": test_dates.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        test_data.index = test_dates.tz_localize("UTC")
        return test_data

    def test_initialization_with_env_var(self, f107omni):
        assert f107omni.data_dir.exists()

    def test_initialization_with_data_dir(self):
        f107omni = F107OMNI(data_dir=DATA_DIR)
        assert f107omni.data_dir == DATA_DIR

    def test_initialization_without_env_var(self):
        if "OMNI_LOW_RES_STREAM_DIR" in os.environ:
            del os.environ["OMNI_LOW_RES_STREAM_DIR"]
        with pytest.raises(ValueError):
            F107OMNI()

    def test_download_and_process(self, f107omni, mocker):
        mocker.patch("wget.download")
        mocker.patch.object(
            f107omni,
            "_process_single_file",
            return_value=f107omni._process_single_file(
                Path(TEST_DIR) / "data/omni2_2020.dat"
            ),
        )

        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

        f107omni.download_and_process(start_time, end_time)

        assert (TEST_DIR / Path("data/omni2_2020.dat")).exists()

    def test_read_without_download(self, f107omni):
        start_time = datetime(2021, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2021, 12, 31, tzinfo=timezone.utc)
        with warnings.catch_warnings(record=True) as w:
            df = f107omni.read(start_time, end_time, download=False)
            assert "OMNI_LOW_RES_2021.csv not found" in str(w[-1].message)
            assert not df.empty
            assert "f107" in df.columns
            assert all(df["f107"].isna())
            assert all(df["file_name"].isnull())


    def test_read_with_download(self, f107omni, mock_f107omni_data, mocker):
        mocker.patch("pathlib.Path.exists", return_value=False)
        mocker.patch.object(
            f107omni, "_read_single_file", return_value=mock_f107omni_data
        )
        mocker.patch.object(f107omni, "download_and_process")

        start_time = datetime(2020, 1, 1)
        end_time = datetime(2020, 12, 31)

        df = f107omni.read(start_time, end_time, download=True)
        f107omni.download_and_process.assert_called_once()

        assert not df.empty
        assert all(df["f107"] == 150.0)
        assert "f107" in df.columns
        assert all(idx.hour == 0 for idx in df.index)
        assert all(idx.tzinfo is not None for idx in df.index)
        assert all(idx.tzinfo is timezone.utc for idx in df.index)

    def test_process_single_file(self, f107omni):
        file = Path(TEST_DIR) / "data/omni2_2020.dat"
        df = f107omni._process_single_file(file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "f107" in df.columns

    def test_read_single_file(self, f107omni):
        csv_file = Path(TEST_DIR) / "data/OMNI_LOW_RES_2020.csv"
        df = f107omni._read_single_file(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "f107" in df.columns

    def test_start_year_behind(self, f107omni, mocker, mock_f107omni_data):
        start_time = datetime(1920, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

        mocker.patch("pathlib.Path.exists", return_value=True)
        mocker.patch.object(
            f107omni, "_get_processed_file_list", return_value=([Path("dummy.csv")], [])
        )
        mocker.patch.object(
            f107omni, "_read_single_file", return_value=mock_f107omni_data
        )

        df = pd.DataFrame(
            {
                "f107": [],
                "file_name": [],
            }
        )
        df.index = pd.DatetimeIndex([])

        mocker.patch("pandas.concat", return_value=df)
        mocker.patch.object(pd.DataFrame, "truncate", return_value=df)

        with patch("logging.Logger.warning") as mock_warning:
            result_df = f107omni.read(start_time, end_time)
            mock_warning.assert_any_call(
                "Start date chosen falls behind the existing data. Moving start date to first"
                " available mission files..."
            )

        assert result_df.empty, "Expected resulting DataFrame to be empty"

    def test_remove_processed_file(self):
        os.remove(Path(TEST_DIR) / "data/OMNI_LOW_RES_2020.csv")
