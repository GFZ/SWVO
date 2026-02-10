# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from swvo.io.symh import SymhOMNI

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "../omni/data/OMNI"))


class TestSymhOMNI:
    @pytest.fixture
    def symhomni(self):
        os.environ["OMNI_HIGH_RES_STREAM_DIR"] = str(DATA_DIR)
        yield SymhOMNI()

    @pytest.fixture
    def mock_symhomni_data(self):
        test_dates = pd.date_range(start=datetime(2020, 1, 1), end=datetime(2020, 12, 31, 23, 59, 0), freq="min")
        test_data = pd.DataFrame(
            {
                "t": test_dates,
                "sym-h": [-15.0] * len(test_dates),
                "file_name": "some_file",
                "timestamp": test_dates.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        test_data.index = test_dates.tz_localize("UTC")
        return test_data

    def test_initialization_with_env_var(self, symhomni):
        assert symhomni.data_dir.exists()

    def test_initialization_with_data_dir(self):
        symhomni = SymhOMNI(data_dir=DATA_DIR)
        assert symhomni.data_dir == DATA_DIR

    def test_initialization_without_env_var(self):
        if "OMNI_HIGH_RES_STREAM_DIR" in os.environ:
            del os.environ["OMNI_HIGH_RES_STREAM_DIR"]
        with pytest.raises(ValueError):
            SymhOMNI()

    def test_download_and_process(self, symhomni):
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)
        # download this file without mocking
        symhomni.download_and_process(start_time, end_time)
        for file in (DATA_DIR / "OMNI" / "2020").glob("OMNI_HIGH_RES_1min_2020*.csv"):
            assert file.exists()

    def test_read_without_download(self, symhomni):
        start_time = datetime(2021, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2021, 2, 28, tzinfo=timezone.utc)
        with pytest.raises(
            ValueError
        ):  # value error is raised when no files are found hence no concatenation is possible
            symhomni.read(start_time, end_time, download=False)

    def test_read_with_download(self, symhomni):
        start_time = datetime(2021, 12, 1, tzinfo=timezone.utc)
        end_time = datetime(2022, 1, 31, tzinfo=timezone.utc)
        df = symhomni.read(start_time, end_time, download=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "sym-h" in df.columns

    def test_read_single_file(self, symhomni):
        csv_file = Path(DATA_DIR) / "2012/OMNI_HIGH_RES_1min_201212.csv"
        df = symhomni._read_single_file(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "sym-h" in df.columns

    def test_year_transition(self, symhomni):
        start_time = datetime(2012, 12, 31, 23, 50, 0, tzinfo=timezone.utc)
        end_time = datetime(2012, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        result_df = symhomni.read(start_time, end_time, download=False)

        assert result_df.index.min() == pd.Timestamp("2012-12-31 23:50:00+00:00")
        assert result_df.index.max() == pd.Timestamp("2013-01-01 00:00:00+00:00")

    def test_remove_processed_file(self):
        shutil.rmtree(Path(DATA_DIR) / "2021", ignore_errors=True)
        shutil.rmtree(Path(DATA_DIR) / "2022", ignore_errors=True)
