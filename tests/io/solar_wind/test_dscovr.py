# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from swvo.io.solar_wind import DSCOVR

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_dscovr"
EXPECTED_COLUMNS = [*DSCOVR.MAG_FIELDS, *DSCOVR.SWEPAM_FIELDS, "pdyn"]


class TestDSCOVR:
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        TEST_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        yield

        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR, ignore_errors=True)

    @pytest.fixture
    def dscovr_instance(self):
        with patch.dict("os.environ", {DSCOVR.ENV_VAR_NAME: str(DATA_DIR)}):
            instance = DSCOVR()
            return instance

    @pytest.fixture
    def sample_portal_payload(self):
        return {
            "status": {"code": 200},
            "data": {
                "time": [
                    "2024-03-15T01:00:00Z",
                    "2024-03-15T01:02:00Z",
                    "2024-03-15T01:03:00Z",
                ],
                "m1m_dscovr.bt": [5.5, 0.0, 7.1],
                "m1m_dscovr.bx_gse": [1.1, -100000.1, 3.3],
                "m1m_dscovr.by_gse": [2.2, 4.4, 5.5],
                "m1m_dscovr.bz_gse": [-1.2, -2.3, -3.4],
                "f1m_dscovr.proton_density": [4.2, 5.0, 0.0],
                "f1m_dscovr.proton_speed": [380.0, 0.0, 400.0],
                "f1m_dscovr.proton_temperature": [145000.0, 150000.0, -100000.1],
            },
        }

    def test_initialization_with_env_var(self):
        with patch.dict("os.environ", {DSCOVR.ENV_VAR_NAME: str(DATA_DIR)}):
            dscovr = DSCOVR()
            assert dscovr.data_dir == DATA_DIR

    def test_initialization_with_explicit_path(self):
        explicit_path = DATA_DIR / "explicit"
        dscovr = DSCOVR(data_dir=explicit_path)
        assert dscovr.data_dir == explicit_path

    def test_initialization_without_env_var(self):
        if DSCOVR.ENV_VAR_NAME in os.environ:
            del os.environ[DSCOVR.ENV_VAR_NAME]
        with pytest.raises(ValueError):
            DSCOVR()

    def test_get_processed_file_list(self, dscovr_instance):
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

        file_paths, time_intervals = dscovr_instance._get_processed_file_list(start_time, end_time)

        assert len(file_paths) == 366
        assert all(str(path).startswith(str(DATA_DIR)) for path in file_paths)
        assert all(path.name.startswith("DSCOVR_SW_NOWCAST_") for path in file_paths)
        assert len(time_intervals) == 366
        assert all(isinstance(interval, tuple) for interval in time_intervals)

    def test_download_and_process(self, dscovr_instance, sample_portal_payload):
        start_time = datetime(2024, 3, 15, 1, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 3, 15, 1, 3, tzinfo=timezone.utc)
        mock_response = Mock()
        mock_response.content = json.dumps(sample_portal_payload).encode()
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            dscovr_instance.download_and_process(start_time, end_time)

        expected_file = DATA_DIR / "2024/03" / "DSCOVR_SW_NOWCAST_20240315.csv"
        assert expected_file.exists()

        data = pd.read_csv(expected_file)
        assert len(data) == 4
        assert all(field in data.columns for field in EXPECTED_COLUMNS)
        assert data["pdyn"].iloc[0] == pytest.approx(2e-6 * 4.2 * 380.0**2)
        assert not Path("./temp_sw_dscovr_wget").exists()

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["start_time"] == "2024-03-15T01:00:00"
        assert kwargs["params"]["end_time"] == "2024-03-15T01:03:00"
        assert kwargs["params"]["parameters"] == ";".join(dscovr_instance._portal_parameters())
        assert kwargs["params"]["format"] == "json"
        assert kwargs["params"]["time_format"] == "iso"
        assert kwargs["timeout"] == 30

    def test_portal_parameters(self, dscovr_instance):
        parameters = dscovr_instance._portal_parameters()

        assert parameters == [
            "DSCOVR:m1m_dscovr:bt",
            "DSCOVR:m1m_dscovr:bx_gse",
            "DSCOVR:m1m_dscovr:by_gse",
            "DSCOVR:m1m_dscovr:bz_gse",
            "DSCOVR:f1m_dscovr:proton_speed",
            "DSCOVR:f1m_dscovr:proton_density",
            "DSCOVR:f1m_dscovr:proton_temperature",
        ]

    def test_process_single_file(self, dscovr_instance, sample_portal_payload):
        test_file = DATA_DIR / DSCOVR.NAME_DATA

        with open(test_file, "w") as f:
            json.dump(sample_portal_payload, f)

        data = dscovr_instance._process_single_file(DATA_DIR)

        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == EXPECTED_COLUMNS
        assert len(data) == 4
        assert data.index[0] == pd.Timestamp("2024-03-15T01:00:00Z")
        assert data.index[-1] == pd.Timestamp("2024-03-15T01:03:00Z")
        assert data.loc["2024-03-15 01:00:00+00:00", "bavg"] == 5.5
        assert data.loc["2024-03-15 01:00:00+00:00", "bx_gsm"] == 1.1
        assert data.loc["2024-03-15 01:01:00+00:00"].isna().all()
        assert np.isnan(data.loc["2024-03-15 01:02:00+00:00", "bx_gsm"])
        assert np.isnan(data.loc["2024-03-15 01:02:00+00:00", "speed"])
        assert np.isnan(data.loc["2024-03-15 01:03:00+00:00", "proton_density"])
        assert np.isnan(data.loc["2024-03-15 01:03:00+00:00", "temperature"])
        assert data.loc["2024-03-15 01:00:00+00:00", "pdyn"] == pytest.approx(2e-6 * 4.2 * 380.0**2)

    def test_process_single_file_api_error(self, dscovr_instance):
        test_file = DATA_DIR / DSCOVR.NAME_DATA
        payload = {"status": {"code": 404, "message": "No matching data"}, "data": {"time": []}}

        with open(test_file, "w") as f:
            json.dump(payload, f)

        with pytest.raises(FileNotFoundError, match="No matching data"):
            dscovr_instance._process_single_file(DATA_DIR)

    def test_read_with_no_data(self, dscovr_instance):
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 1, 2, tzinfo=timezone.utc)

        data = dscovr_instance.read(start_time, end_time, download=False)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(col in data.columns for col in EXPECTED_COLUMNS)
        assert data.isna().all().all()

    def test_read_invalid_time_range(self, dscovr_instance):
        start_time = datetime(2020, 12, 31, tzinfo=timezone.utc)
        end_time = datetime(2020, 1, 1, tzinfo=timezone.utc)

        with pytest.raises(AssertionError, match="Start time must be before end time"):
            dscovr_instance.read(start_time, end_time)

    def test_read_with_existing_data(self, dscovr_instance):
        start_time = datetime(2024, 3, 15, tzinfo=timezone.utc)
        end_time = datetime(2024, 3, 15, 23, 59, 59, tzinfo=timezone.utc)

        t = pd.date_range(start=start_time, end=end_time, freq="min")

        sample_data = pd.DataFrame(
            {
                "t": t,
                "bavg": np.full(len(t), 5.0),
                "bx_gsm": np.full(len(t), 1.0),
                "by_gsm": np.full(len(t), 2.0),
                "bz_gsm": np.full(len(t), -1.0),
                "proton_density": np.full(len(t), 4.2),
                "speed": np.full(len(t), 380.0),
                "temperature": np.full(len(t), 145000.0),
                "pdyn": np.full(len(t), 2e-6 * 4.2 * 380.0**2),
            }
        )

        file_path = DATA_DIR / start_time.strftime("%Y/%m") / f"DSCOVR_SW_NOWCAST_{start_time.strftime('%Y%m%d')}.csv"
        file_path.parent.mkdir(parents=True)
        sample_data.to_csv(file_path, index=False)

        data = dscovr_instance.read(start_time, end_time)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1440
        assert all(col in data.columns for col in EXPECTED_COLUMNS)
        assert data.loc[start_time, "bavg"] == pytest.approx(sample_data["bavg"].iloc[0])
        assert data.loc[start_time, "file_name"] == file_path

    @pytest.mark.parametrize(
        "field,invalid_value,expected",
        [
            ("bx_gsm", -100000.1, np.nan),
            ("speed", 0.0, np.nan),
            ("temperature", -100000.1, np.nan),
            ("proton_density", 4.2, 4.2),
        ],
    )
    def test_invalid_value_handling(self, dscovr_instance, field, invalid_value, expected):
        data = pd.DataFrame({column: [1.0] for column in DSCOVR.MAG_FIELDS + DSCOVR.SWEPAM_FIELDS})
        data[field] = invalid_value

        dscovr_instance._replace_invalid_values(data)

        if np.isnan(expected):
            assert np.isnan(data[field].iloc[0])
        else:
            assert data[field].iloc[0] == pytest.approx(expected)

    def test_with_propagation(self):
        start_time = datetime(2024, 11, 21, tzinfo=timezone.utc)
        end_time = datetime(2024, 11, 24, tzinfo=timezone.utc)

        dscovr_instance = DSCOVR(Path(__file__).parent / "data" / "DSCOVR")
        data = dscovr_instance.read(start_time, end_time, propagation=True)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5109
        assert all(col in data.columns for col in DSCOVR.MAG_FIELDS + DSCOVR.SWEPAM_FIELDS)
        assert any(data["file_name"] == "propagated from previous DSCOVR NOWCAST file")
        assert data.index.is_monotonic_increasing
