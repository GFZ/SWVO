# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from swvo.io.omni.omni_low_res import OMNILowRes

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))


class TestOMNILowRes:
    @pytest.fixture
    def omni_low_res(self):
        os.environ["OMNI_LOW_RES_STREAM_DIR"] = str(DATA_DIR)
        yield OMNILowRes()

    def test_initialization_with_env_var(self, omni_low_res):
        assert omni_low_res.data_dir.exists()

    def test_initialization_with_data_dir(self):
        omni_low_res = OMNILowRes(data_dir=DATA_DIR)
        assert omni_low_res.data_dir == DATA_DIR

    def test_initialization_without_env_var(self):
        if "OMNI_LOW_RES_STREAM_DIR" in os.environ:
            del os.environ["OMNI_LOW_RES_STREAM_DIR"]
        with pytest.raises(ValueError):
            OMNILowRes()

    def test_download_and_process(self, omni_low_res, mocker):
        mock_response = mocker.Mock()
        mock_response.content = b"test content"
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch("requests.get", return_value=mock_response)
        mocker.patch.object(
            omni_low_res,
            "_process_single_file",
            return_value=omni_low_res._process_single_file(Path(TEST_DIR) / "data/omni2_2020.dat"),
        )

        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

        omni_low_res.download_and_process(start_time, end_time)

        assert (TEST_DIR / Path("data/omni2_2020.dat")).exists()

    def test_read_without_download(self, omni_low_res, mocker):
        start_time = datetime(2021, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2021, 12, 31, tzinfo=timezone.utc)
        with warnings.catch_warnings(record=True) as w:
            df = omni_low_res.read(start_time, end_time, download=False)
            assert "OMNI_LOW_RES_2021.csv not found" in str(w[-1].message)
            assert not df.empty
            assert "f107" in df.columns
            assert "kp" in df.columns
            assert "dst" in df.columns
            assert all(df["f107"].isna())
            assert all(df["kp"].isna())
            assert all(df["dst"].isna())
            assert all(df["file_name"].isnull())

    def test_read_with_download(self, omni_low_res, mocker):
        mocker.patch.object(omni_low_res, "download_and_process")
        mocker.patch.object(
            omni_low_res,
            "_read_single_file",
            return_value=pd.DataFrame(
                index=pd.date_range(start=datetime(2022, 1, 1), end=datetime(2022, 12, 31), tz=timezone.utc)
            ),
        )
        start_time = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2022, 12, 31, tzinfo=timezone.utc)
        omni_low_res.read(start_time, end_time, download=True)
        omni_low_res.download_and_process.assert_called_once()

    def test_process_single_file(self, omni_low_res):
        file = Path(TEST_DIR) / "data/omni2_2020.dat"
        df = omni_low_res._process_single_file(file)

        assert isinstance(df, pd.DataFrame)
        assert all(column in df.columns for column in ["kp", "dst", "f107"])
        assert len(df) > 0

    def test_read_single_file(self, omni_low_res):
        csv_file = Path(TEST_DIR) / "data/OMNI_LOW_RES_2020.csv"
        df = omni_low_res._read_single_file(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert all(column in df.columns for column in ["kp", "dst", "f107"])
        assert len(df) > 0

    def test_start_year_behind(self, omni_low_res, mocker):
        start_time = datetime(1920, 1, 1)
        end_time = datetime(2020, 12, 31)

        mocked_df = pd.DataFrame(index=pd.date_range(start_time, end_time))

        mocker.patch.object(omni_low_res, "_get_processed_file_list", return_value=([], []))
        mocker.patch.object(omni_low_res, "_read_single_file", return_value=mocked_df)

        mocker.patch("pandas.concat", return_value=pd.DataFrame())
        mocker.patch.object(pd.DataFrame, "truncate", return_value=pd.DataFrame())

        with patch("logging.Logger.warning") as mock_warning:
            df = omni_low_res.read(start_time, end_time)
            mock_warning.assert_any_call(
                "Start date chosen falls behind the existing data. Moving start date to first available mission files..."
            )

            assert "f107" in df.columns
            assert "kp" in df.columns
            assert "dst" in df.columns
            assert all(df["f107"].isna())
            assert all(df["kp"].isna())
            assert all(df["dst"].isna())
            assert all(df["file_name"].isnull())

    def test_remove_processed_file(self):
        os.remove(Path(TEST_DIR) / "data/OMNI_LOW_RES_2020.csv")
