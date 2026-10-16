# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: S101

import logging
import os
import shutil
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
        return (
            "2025-01-31T22:29:30Z,-5,20\n"
            "2025-01-31T23:29:30Z,-6,20\n"
            "2025-02-01T00:29:30Z,99999,10\n"
            "2025-02-01T01:29:30Z,3,10\n"
        )

    @staticmethod
    def _mock_hapi(data, start="1957-01-01T00:29:30Z", stop="2026-10-16T10:00:00Z", requested=None):
        class MockResponse:
            def __init__(self, text="", payload=None):
                self.text = text
                self._payload = payload

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        def mock_get(url, timeout):
            if requested is not None:
                requested.append(url)
            if "/info" in url:
                return MockResponse(payload={"startDate": start, "stopDate": stop})
            return MockResponse(text=data)

        return patch("swvo.io.dst.wdc.requests.get", side_effect=mock_get)

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
        current_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        past_month = current_month.replace(month=current_month.month - 2)
        dst_instance.download_and_process(past_month, current_month)

        expected_files = list(MOCK_DATA_PATH.glob("**/WDC_DST_*.csv"))

        assert 1 <= len(expected_files) & len(expected_files) <= 3

        data = pd.read_csv(expected_files[0])
        assert "dst" in data.columns

    def test_download_and_process_warns_when_no_data(self, dst_instance, caplog):
        with self._mock_hapi(""):
            with caplog.at_level(logging.WARNING, logger="py.warnings"):
                dst_instance.download_and_process(datetime(2025, 1, 1), datetime(2025, 1, 2))

        assert "WDC Dst data not found" in caplog.text
        assert not (MOCK_DATA_PATH / "2025" / "WDC_DST_202501.csv").exists()

    def test_download_and_process_splits_months(self, dst_instance, sample_dst_data):
        requested = []
        with self._mock_hapi(sample_dst_data, requested=requested):
            dst_instance.download_and_process(datetime(2025, 1, 1), datetime(2025, 2, 2))

        assert len(requested) == 2
        assert "id=hour_dst&" in requested[1]
        assert "start=2025-01-01T00:00:00Z&stop=2025-03-01T00:00:00Z" in requested[1]

        january = pd.read_csv(MOCK_DATA_PATH / "2025" / "WDC_DST_202501.csv")
        february = pd.read_csv(MOCK_DATA_PATH / "2025" / "WDC_DST_202502.csv")
        assert list(january["dst"]) == [-5.0, -6.0]
        assert list(february["dst"]) == [3.0]

    def test_download_and_process_clamps_to_available_range(self, dst_instance, sample_dst_data):
        requested = []
        with self._mock_hapi(
            sample_dst_data, start="1957-01-01T00:29:30Z", stop="2025-02-01T10:00:00Z", requested=requested
        ):
            dst_instance.download_and_process(datetime(1950, 1, 1), datetime(2030, 1, 1))

        assert "start=1957-01-01T00:29:30Z&stop=2025-02-01T10:00:00Z" in requested[1]

    def test_download_and_process_does_not_request_existing_files(self, dst_instance, sample_dst_data):
        with self._mock_hapi(sample_dst_data):
            dst_instance.download_and_process(datetime(2025, 1, 1), datetime(2025, 2, 2))

        requested = []
        with self._mock_hapi("", requested=requested):
            dst_instance.download_and_process(datetime(2025, 1, 1), datetime(2025, 2, 2))
        assert requested == []

        with self._mock_hapi(sample_dst_data, requested=requested):
            dst_instance.download_and_process(datetime(2025, 1, 1), datetime(2025, 2, 2), reprocess_files=True)
        assert len(requested) == 2

    def test_process_hapi_response(self, dst_instance, sample_dst_data):
        data = dst_instance._process_hapi_response(sample_dst_data)

        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ["dst"]
        assert data.index.name == "timestamp"
        # fill value dropped, timestamps moved from the center to the start of the hour
        assert list(data.index.strftime("%Y-%m-%d %H:%M")) == [
            "2025-01-31 22:00",
            "2025-01-31 23:00",
            "2025-02-01 01:00",
        ]
        assert list(data["dst"]) == [-5.0, -6.0, 3.0]

    def test_process_hapi_response_empty(self, dst_instance):
        data = dst_instance._process_hapi_response("")
        assert len(data) == 0
        assert list(data.columns) == ["dst"]

    def test_read_with_no_data(self, dst_instance, caplog):
        start_time = datetime(2020, 1, 1)
        end_time = datetime(2020, 12, 31)

        with caplog.at_level(logging.WARNING, logger="py.warnings"):
            df = dst_instance.read(start_time, end_time, download=False)

        assert "WDC_DST_202012.csv not found" in caplog.text
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 8761
        assert all(df["dst"].isna())
        assert all(df["file_name"].isnull())

    def test_read_with_download_missing_file_fills_nans(self, dst_instance, caplog):
        start_time = datetime(2026, 1, 1)
        end_time = datetime(2026, 1, 2)

        with patch.object(dst_instance, "download_and_process", return_value=None):
            with caplog.at_level(logging.WARNING, logger="py.warnings"):
                df = dst_instance.read(start_time, end_time, download=True)

        assert "filling with NaNs" in caplog.text
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 25
        assert all(df["dst"].isna())
        assert all(df["file_name"].isnull())

    def test_read_invalid_time_range(self, dst_instance):
        start_time = datetime(2020, 12, 31)
        end_time = datetime(2020, 1, 1)

        with pytest.raises(AssertionError, match="Start time must be before end time"):
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
    def test_read_missing_years_warning(self, dst_instance, test_year, caplog):
        start_time = datetime(test_year, 1, 1)
        end_time = datetime(test_year, 1, 31)
        with caplog.at_level(logging.WARNING, logger="py.warnings"):
            dst_instance.read(start_time, end_time, download=False)
        assert f"WDC_DST_{test_year}01.csv not found" in caplog.text
