# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from swvo.io.solar_wind import SWACE

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_ace"


class TestSWACE:
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        TEST_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        yield

        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR, ignore_errors=True)

    @pytest.fixture
    def swace_instance(self):
        with patch.dict("os.environ", {SWACE.ENV_VAR_NAME: str(DATA_DIR)}):
            instance = SWACE()
            return instance

    @pytest.fixture
    def sample_mag_data(self):
        """Sample magnetometer data"""
        return """:Data_list: 20240315_ace_mag_1m.txt
:Created: 2024 Mar 15 0102 UT

    2024 03 15 0100    60384   3600   0   1.42  -0.89   2.84   3.31  31.42  31.63
    2024 03 15 0101    60384   3660   0   1.39  -0.87   2.80   3.26  31.38  31.59"""

    @pytest.fixture
    def sample_swepam_data(self):
        """Sample SWEPAM data"""
        return """:Data_list: 20240315_ace_swepam_1m.txt
:Created: 2024 Mar 15 0102 UT

    2024 03 15 0100    60384   3600   0      4.2     380      145000
    2024 03 15 0101    60384   3660   0      4.1     375      144000"""

    def test_initialization_with_env_var(self):
        with patch.dict("os.environ", {SWACE.ENV_VAR_NAME: str(DATA_DIR)}):
            swace = SWACE()
            assert swace.data_dir == DATA_DIR

    def test_initialization_with_explicit_path(self):
        explicit_path = DATA_DIR / "explicit"
        swace = SWACE(data_dir=explicit_path)
        assert swace.data_dir == explicit_path

    def test_initialization_without_env_var(self):
        if SWACE.ENV_VAR_NAME in os.environ:
            del os.environ[SWACE.ENV_VAR_NAME]
        with pytest.raises(ValueError):
            SWACE()

    def test_get_processed_file_list(self, swace_instance):
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

        file_paths, time_intervals = swace_instance._get_processed_file_list(start_time, end_time)

        assert len(file_paths) == 366
        assert all(str(path).startswith(str(DATA_DIR)) for path in file_paths)
        assert all(path.name.startswith("ACE_SW_NOWCAST_") for path in file_paths)
        assert len(time_intervals) == 366
        assert all(isinstance(interval, tuple) for interval in time_intervals)

    def test_dated_source_file_names(self, swace_instance):
        request_time = datetime(2024, 3, 15, 1, 0, tzinfo=timezone.utc)

        assert swace_instance._dated_file_name(swace_instance.NAME_MAG, request_time) == "20240315_ace_mag_1m.txt"
        assert swace_instance._dated_file_name(swace_instance.NAME_SWEPAM, request_time) == "20240315_ace_swepam_1m.txt"

    def test_download_and_process(self, swace_instance, sample_mag_data, sample_swepam_data):
        start_time = datetime(2024, 3, 15, 1, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 3, 15, 1, 1, tzinfo=timezone.utc)
        mag_file_name = swace_instance._dated_file_name(swace_instance.NAME_MAG, start_time)
        swepam_file_name = swace_instance._dated_file_name(swace_instance.NAME_SWEPAM, start_time)

        def mock_get(url, **kwargs):
            mock_response = Mock()
            if url.endswith(mag_file_name):
                mock_response.content = sample_mag_data.encode()
            else:
                mock_response.content = sample_swepam_data.encode()
            mock_response.raise_for_status = Mock()
            return mock_response

        with patch("requests.get", side_effect=mock_get) as mock_requests_get:
            swace_instance.download_and_process(start_time, end_time)

        expected_file = DATA_DIR / "2024/03" / "ACE_SW_NOWCAST_20240315.csv"
        assert expected_file.exists()

        data = pd.read_csv(expected_file)
        assert len(data) == 2
        assert all(field in data.columns for field in SWACE.MAG_FIELDS + SWACE.SWEPAM_FIELDS)
        assert data["speed"].iloc[0] == 380.0
        assert data["pdyn"].iloc[0] == pytest.approx(2e-6 * 4.2 * 380.0**2)
        assert mock_requests_get.call_args_list[0].args[0] == SWACE.URL + mag_file_name
        assert mock_requests_get.call_args_list[1].args[0] == SWACE.URL + swepam_file_name

    def test_download_and_process_multiple_days(self, swace_instance):
        start_time = datetime(2024, 3, 15, 23, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 3, 16, 1, 0, tzinfo=timezone.utc)

        def mag_data(date: datetime):
            return f""":Data_list: {date:%Y%m%d}_ace_mag_1m.txt
:Created: {date:%Y} Mar {date:%d} 0102 UT

    {date:%Y %m %d} 0000    60384      0   0   1.42  -0.89   2.84   3.31  31.42  31.63
    {date:%Y %m %d} 0001    60384     60   0   1.39  -0.87   2.80   3.26  31.38  31.59"""

        def swepam_data(date: datetime):
            return f""":Data_list: {date:%Y%m%d}_ace_swepam_1m.txt
:Created: {date:%Y} Mar {date:%d} 0102 UT

    {date:%Y %m %d} 0000    60384      0   0      4.2     380      145000
    {date:%Y %m %d} 0001    60384     60   0      4.1     375      144000"""

        def mock_get(url, **kwargs):
            file_name = Path(url).name
            file_date = datetime.strptime(file_name.split("_", maxsplit=1)[0], "%Y%m%d").replace(tzinfo=timezone.utc)
            mock_response = Mock()
            mock_response.content = (
                mag_data(file_date) if "_ace_mag_" in file_name else swepam_data(file_date)
            ).encode()
            mock_response.raise_for_status = Mock()
            return mock_response

        with patch("requests.get", side_effect=mock_get) as mock_requests_get:
            swace_instance.download_and_process(start_time, end_time)

        assert (DATA_DIR / "2024/03" / "ACE_SW_NOWCAST_20240315.csv").exists()
        assert (DATA_DIR / "2024/03" / "ACE_SW_NOWCAST_20240316.csv").exists()
        assert [call.args[0] for call in mock_requests_get.call_args_list] == [
            SWACE.URL + "20240315_ace_mag_1m.txt",
            SWACE.URL + "20240315_ace_swepam_1m.txt",
            SWACE.URL + "20240316_ace_mag_1m.txt",
            SWACE.URL + "20240316_ace_swepam_1m.txt",
        ]

    def test_process_mag_file(self, swace_instance, sample_mag_data):
        request_time = datetime(2024, 3, 15, 1, 0, tzinfo=timezone.utc)
        test_file = DATA_DIR / swace_instance._dated_file_name(swace_instance.NAME_MAG, request_time)
        test_file.parent.mkdir(exist_ok=True)

        with open(test_file, "w") as f:
            f.write(sample_mag_data)

        data = swace_instance._process_mag_file(DATA_DIR)

        assert isinstance(data, pd.DataFrame)
        assert all(field in data.columns for field in SWACE.MAG_FIELDS)
        assert len(data) == 2
        assert data["bx_gsm"].iloc[0] == 1.42

    def test_process_swepam_file(self, swace_instance, sample_swepam_data):
        request_time = datetime(2024, 3, 15, 1, 0, tzinfo=timezone.utc)
        test_file = DATA_DIR / swace_instance._dated_file_name(swace_instance.NAME_SWEPAM, request_time)
        test_file.parent.mkdir(exist_ok=True)

        with open(test_file, "w") as f:
            f.write(sample_swepam_data)

        data = swace_instance._process_swepam_file(DATA_DIR)

        assert isinstance(data, pd.DataFrame)
        assert all(field in data.columns for field in SWACE.SWEPAM_FIELDS)
        assert len(data) == 2
        assert data["speed"].iloc[0] == 380.0
        assert data["temperature"].iloc[0] == 145000.0

    def test_read_with_no_data(self, swace_instance):
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 1, 2, tzinfo=timezone.utc)

        data = swace_instance.read(start_time, end_time, download=False)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(col in data.columns for col in SWACE.MAG_FIELDS + SWACE.SWEPAM_FIELDS)
        assert data.isna().all().all()

    def test_read_invalid_time_range(self, swace_instance):
        start_time = datetime(2020, 12, 31, tzinfo=timezone.utc)
        end_time = datetime(2020, 1, 1, tzinfo=timezone.utc)

        with pytest.raises(ValueError):
            swace_instance.read(start_time, end_time)

    def test_read_with_existing_data(self, swace_instance):
        start_time = datetime(2024, 3, 15, tzinfo=timezone.utc)
        end_time = datetime(2024, 3, 15, 23, 59, 59, tzinfo=timezone.utc)

        t = pd.date_range(start=start_time, end=end_time, freq="min")

        sample_data = pd.DataFrame(
            {
                "t": t,
                "bavg": np.random.random(len(t)),
                "bx_gsm": np.random.random(len(t)),
                "by_gsm": np.random.random(len(t)),
                "bz_gsm": np.random.random(len(t)),
                "proton_density": np.random.random(len(t)),
                "speed": np.random.random(len(t)),
                "temperature": np.random.random(len(t)),
            }
        )

        file_path = DATA_DIR / f"ACE_SW_NOWCAST_{start_time.strftime('%Y%m%d')}.csv"
        sample_data.to_csv(file_path, index=False)

        data = swace_instance.read(start_time, end_time)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1440
        assert all(col in data.columns for col in SWACE.MAG_FIELDS + SWACE.SWEPAM_FIELDS)

    def test_cleanup_after_download(self, swace_instance, sample_mag_data, sample_swepam_data):
        start_time = datetime(2024, 3, 15, 1, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 3, 15, 1, 1, tzinfo=timezone.utc)
        mag_file_name = swace_instance._dated_file_name(swace_instance.NAME_MAG, start_time)

        def mock_get(url, **kwargs):
            mock_response = Mock()
            if url.endswith(mag_file_name):
                mock_response.content = sample_mag_data.encode()
            else:
                mock_response.content = sample_swepam_data.encode()
            mock_response.raise_for_status = Mock()
            return mock_response

        with patch("requests.get", side_effect=mock_get):
            swace_instance.download_and_process(start_time, end_time)

        temp_dir = Path("./temp_sw_ace_wget")
        assert not temp_dir.exists(), "Temporary directory should be cleaned up"

    @pytest.mark.parametrize(
        "field,invalid_value,expected",
        [
            ("bx_gsm", -999.1, np.nan),
            ("speed", -9999.1, np.nan),
            ("temperature", -99999.1, np.nan),
            ("proton_density", 4.2, 4.2),
        ],
    )
    def test_invalid_value_handling(self, field, invalid_value, expected):
        data = pd.DataFrame({field: [invalid_value]})

        if field in SWACE.MAG_FIELDS:
            mask = data[field] < -999.0
        elif field in ["proton_density", "speed"]:
            mask = data[field] < -9999.0
        else:
            mask = data[field] < -99999.0

        data.loc[mask, field] = np.nan

        if np.isnan(expected):
            assert np.isnan(data[field].iloc[0])
        else:
            assert data[field].iloc[0] == pytest.approx(expected)

    def test_with_propagation(self):
        start_time = datetime(2024, 11, 21, tzinfo=timezone.utc)
        end_time = datetime(2024, 11, 24, tzinfo=timezone.utc)

        swace_instance = SWACE(Path(__file__).parent / "data" / "ACE_RT")
        data = swace_instance.read(start_time, end_time, propagation=True)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5109
        assert all(col in data.columns for col in SWACE.MAG_FIELDS + SWACE.SWEPAM_FIELDS)
        assert any(data["file_name"] == "propagated from previous ACE NOWCAST file")
        assert data.index.is_monotonic_increasing
