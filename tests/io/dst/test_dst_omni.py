# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from pathlib import Path
from datetime import datetime, timezone
from data_management.io.dst import DSTOMNI
import pandas as pd
import warnings
from unittest.mock import patch

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))


class TestdstOMNI:
    @pytest.fixture
    def dstomni(self):
        os.environ["OMNI_LOW_RES_STREAM_DIR"] = str(DATA_DIR)
        yield DSTOMNI()

    @pytest.fixture
    def mock_dstomni_data(self):
        test_dates = pd.date_range(
            start=datetime(2020, 1, 1), end=datetime(2020, 12, 31, 23, 00, 00), freq="h"
        )
        test_data = pd.DataFrame(
            {
                "t": test_dates,
                "dst": [150.0] * len(test_dates),
                "file_name": "some_file",
                "timestamp": test_dates.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        test_data.index = test_dates.tz_localize("UTC")
        return test_data

    def test_initialization_with_env_var(self, dstomni):
        assert dstomni.data_dir.exists()

    def test_initialization_with_data_dir(self):
        dstomni = DSTOMNI(data_dir=DATA_DIR)
        assert dstomni.data_dir == DATA_DIR

    def test_initialization_without_env_var(self):
        if "OMNI_LOW_RES_STREAM_DIR" in os.environ:
            del os.environ["OMNI_LOW_RES_STREAM_DIR"]
        with pytest.raises(ValueError):
            DSTOMNI()

    def test_download_and_process(self, dstomni, mocker):
        mocker.patch("wget.download")
        mocker.patch.object(
            dstomni,
            "_process_single_file",
            return_value=dstomni._process_single_file(
                Path(TEST_DIR) / "data/omni2_2020.dat"
            ),
        )

        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

        dstomni.download_and_process(start_time, end_time)

        assert (TEST_DIR / Path("data/omni2_2020.dat")).exists()

    def test_read_without_download(self, dstomni):
        start_time = datetime(2021, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2021, 2, 28, tzinfo=timezone.utc)
        with warnings.catch_warnings(record=True) as w:
            df = dstomni.read(start_time, end_time, download=False)
            assert "OMNI_LOW_RES_2021.csv not found" in str(w[-1].message)
            assert not df.empty
            assert "dst" in df.columns
            assert all(df["dst"].isna())
            assert all(df["file_name"].isnull())


    def test_read_with_download(self, dstomni, mock_dstomni_data, mocker):
        mocker.patch("pathlib.Path.exists", return_value=False)
        mocker.patch.object(
            dstomni, "_read_single_file", return_value=mock_dstomni_data
        )
        mocker.patch.object(dstomni, "download_and_process")

        start_time = datetime(2020, 1, 1)
        end_time = datetime(2020, 12, 31)

        df = dstomni.read(start_time, end_time, download=True)
        dstomni.download_and_process.assert_called_once()

        assert not df.empty
        assert all(df["dst"] == 150.0)
        assert "dst" in df.columns
        assert all(idx.tzinfo is not None for idx in df.index)
        assert all(idx.tzinfo is timezone.utc for idx in df.index)

    def test_process_single_file(self, dstomni):
        file = Path(TEST_DIR) / "data/omni2_2020.dat"
        df = dstomni._process_single_file(file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "dst" in df.columns

    def test_read_single_file(self, dstomni):
        csv_file = Path(TEST_DIR) / "data/OMNI_LOW_RES_2020.csv"
        df = dstomni._read_single_file(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "dst" in df.columns

    def test_remove_processed_file(self):
        os.remove(Path(TEST_DIR) / "data/OMNI_LOW_RES_2020.csv")
