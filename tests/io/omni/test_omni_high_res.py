# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from pathlib import Path
from datetime import datetime, timezone
from swvo.io.omni.omni_high_res import OMNIHighRes
import pandas as pd
import numpy as np

from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))


class TestOMNIHighRes:
    @pytest.fixture
    def omni_high_res(self):
        os.environ["OMNI_HIGH_RES_STREAM_DIR"] = str(DATA_DIR)
        yield OMNIHighRes()

    def test_initialization_with_env_var(self, omni_high_res):
        assert omni_high_res.data_dir.exists()

    def test_initialization_with_data_dir(self):
        omni_high_res = OMNIHighRes(data_dir=DATA_DIR)
        assert omni_high_res.data_dir == DATA_DIR

    def test_initialization_without_env_var(self):
        if "OMNI_HIGH_RES_STREAM_DIR" in os.environ:
            del os.environ["OMNI_HIGH_RES_STREAM_DIR"]
        with pytest.raises(ValueError):
            OMNIHighRes()

    def test_download_and_process(self, omni_high_res):
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)
        # download this file without mocking
        omni_high_res.download_and_process(start_time, end_time)

        assert (TEST_DIR / Path("data/OMNI_HIGH_RES_1min_2020.csv")).exists()

    def test_read_without_download(self, omni_high_res, mocker):
        start_time = datetime(2021, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2021, 12, 31, tzinfo=timezone.utc)
        with pytest.raises(
            ValueError
        ):  # value error is raised when no files are found hence no concatenation is possible
            omni_high_res.read(start_time, end_time, download=False)

    def test_read_with_download(self, omni_high_res, mocker):
        mocker.patch.object(omni_high_res, "download_and_process")
        mocker.patch.object(
            omni_high_res, "_read_single_file", return_value=pd.DataFrame()
        )
        start_time = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2022, 12, 31, tzinfo=timezone.utc)
        omni_high_res.read(start_time, end_time, download=True)
        omni_high_res.download_and_process.assert_called_once()

    def test_invalid_cadence(self, omni_high_res):
        start_time = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2022, 12, 31, tzinfo=timezone.utc)

        with pytest.raises(AssertionError):
            omni_high_res.read(start_time, end_time, cadence_min=2)

        with pytest.raises(AssertionError):
            omni_high_res.download_and_process(start_time, end_time, cadence_min=10)

    def test_start_year_behind(self, omni_high_res, mocker):
        start_time = datetime(1920, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

        mocker.patch.object(
            omni_high_res, "_get_processed_file_list", return_value=([], [])
        )
        mocker.patch.object(
            omni_high_res, "_read_single_file", return_value=pd.DataFrame()
        )

        mocker.patch("pandas.concat", return_value=pd.DataFrame())

        mocker.patch.object(pd.DataFrame, "truncate", return_value=pd.DataFrame())

        with patch("logging.Logger.warning") as mock_warning:
            dfs = omni_high_res.read(start_time, end_time)
            mock_warning.assert_any_call(
                "Start date chosen falls behind the existing data. Moving start date to first available mission files..."
            )

            assert len(dfs) == 0, (
                "Expected dfs list to be empty since no files are found."
            )

    def test_year_transition(self, omni_high_res):
        start_time = datetime(2012, 12, 31, 23, 59, 0, tzinfo=timezone.utc)

        end_time = datetime(2012, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        result_df = omni_high_res.read(start_time, end_time, download=False)

        assert result_df.index.min() == pd.Timestamp("2012-12-31 23:59:00+00:00")
        assert result_df.index.max() == pd.Timestamp("2013-01-01 00:00:00+00:00")

    def test_remove_processed_file(self):
        os.remove(Path(TEST_DIR) / "data/OMNI_HIGH_RES_1min_2020.csv")

    def test_process_single_year_parses_data_correctly(self, omni_high_res):
        data = [
            "YYYY DOY HR MN bavg bx_gsm by_gsm bz_gsm speed proton_density temperature",
            "2020 1 0 0 5.1 1.2 2.3 3.4 400 5.5 1000000",
            "2020 1 0 1 9999.9 9999.9 9999.9 9999.9 99999.8 999.8 9999998.0",
        ]

        df = omni_high_res._process_single_year(data)
        assert isinstance(df.index[0], pd.Timestamp)
        assert len(df) >= 2
        # Check columns
        expected_cols = [
            "bavg",
            "bx_gsm",
            "by_gsm",
            "bz_gsm",
            "speed",
            "proton_density",
            "temperature",
        ]
        assert list(df.columns) == expected_cols
        assert np.isnan(df.iloc[1]["bavg"])
        assert np.isnan(df.iloc[1]["bx_gsm"])
        assert np.isnan(df.iloc[1]["by_gsm"])
        assert np.isnan(df.iloc[1]["bz_gsm"])
        assert np.isnan(df.iloc[1]["speed"])
        assert np.isnan(df.iloc[1]["proton_density"])
        assert np.isnan(df.iloc[1]["temperature"])
        assert df.iloc[0]["bavg"] == 5.1
        assert df.iloc[0]["bx_gsm"] == 1.2
        assert df.iloc[0]["by_gsm"] == 2.3
        assert df.iloc[0]["bz_gsm"] == 3.4
        assert df.iloc[0]["speed"] == 400
        assert df.iloc[0]["proton_density"] == 5.5
        assert df.iloc[0]["temperature"] == 1000000

    def test_process_single_year_handles_missing_data_lines(self, omni_high_res):
        data = [
            "YYYY DOY HR MN bavg bx_gsm by_gsm bz_gsm speed proton_density temperature"
        ]
        with pytest.raises(ValueError):
            df = omni_high_res._process_single_year(data)

    def test_process_single_year_raises_on_missing_header(self, omni_high_res):
        data = ["2020 1 0 0 5.1 1.2 2.3 3.4 400 5.5 1000000"]
        with pytest.raises(StopIteration):
            omni_high_res._process_single_year(data)
