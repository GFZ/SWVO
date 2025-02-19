import os
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import numpy as np
import wget

from data_management.io.solar_wind import DSCOVR

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_dscovr"


class TestDSCOVR:
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        TEST_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        yield

        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR, ignore_errors=True)

    @pytest.fixture
    def swace_instance(self):
        with patch.dict("os.environ", {DSCOVR.ENV_VAR_NAME: str(DATA_DIR)}):
            instance = DSCOVR()
            return instance

    def test_initialization_with_env_var(self):
        with patch.dict("os.environ", {DSCOVR.ENV_VAR_NAME: str(DATA_DIR)}):
            swace = DSCOVR()
            assert swace.data_dir == DATA_DIR

    def test_initialization_with_explicit_path(self):
        explicit_path = DATA_DIR / "explicit"
        swace = DSCOVR(data_dir=explicit_path)
        assert swace.data_dir == explicit_path

    def test_initialization_without_env_var(self):
        if DSCOVR.ENV_VAR_NAME in os.environ:
            del os.environ[DSCOVR.ENV_VAR_NAME]
        with pytest.raises(ValueError):
            DSCOVR()

    def test_get_processed_file_list(self, swace_instance):
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

        file_paths, time_intervals = swace_instance._get_processed_file_list(
            start_time, end_time
        )

        assert len(file_paths) == 366
        assert all(str(path).startswith(str(DATA_DIR)) for path in file_paths)
        assert all(path.name.startswith("DSCOVR_SW_NOWCAST_") for path in file_paths)
        assert len(time_intervals) == 366
        assert all(isinstance(interval, tuple) for interval in time_intervals)

    def test_download_and_process(self, swace_instance):
        current_time = datetime.now(timezone.utc)
        swace_instance.download_and_process(current_time)

        expected_file = (
            DATA_DIR / f"DSCOVR_SW_NOWCAST_{current_time.strftime('%Y%m%d')}.csv"
        )
        assert expected_file.exists()

        data = pd.read_csv(expected_file)
        assert len(data) > 0
        assert all(
            field in data.columns for field in DSCOVR.MAG_FIELDS + DSCOVR.SWEPAM_FIELDS
        )

    def test_process_mag_file(self, swace_instance):
        test_file = DATA_DIR / DSCOVR.NAME_MAG
        test_file.parent.mkdir(exist_ok=True)

        wget.download(DSCOVR.URL + DSCOVR.NAME_MAG, str(test_file))

        data = swace_instance._process_mag_file(DATA_DIR)

        assert isinstance(data, pd.DataFrame)
        assert all(field in data.columns for field in DSCOVR.MAG_FIELDS)

    def test_process_swepam_file(self, swace_instance):
        test_file = DATA_DIR / DSCOVR.NAME_SWEPAM
        test_file.parent.mkdir(exist_ok=True)

        wget.download(DSCOVR.URL + DSCOVR.NAME_SWEPAM, str(test_file))

        data = swace_instance._process_swepam_file(DATA_DIR)

        assert isinstance(data, pd.DataFrame)
        assert all(field in data.columns for field in DSCOVR.SWEPAM_FIELDS)

    def test_read_with_no_data(self, swace_instance):
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 1, 2, tzinfo=timezone.utc)

        data = swace_instance.read(start_time, end_time, download=False)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(
            col in data.columns for col in DSCOVR.MAG_FIELDS + DSCOVR.SWEPAM_FIELDS
        )
        assert data.isna().all().all()

    def test_read_invalid_time_range(self, swace_instance):
        start_time = datetime(2020, 12, 31, tzinfo=timezone.utc)
        end_time = datetime(2020, 1, 1, tzinfo=timezone.utc)

        with pytest.raises(AssertionError):
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

        file_path = DATA_DIR / f"DSCOVR_SW_NOWCAST_{start_time.strftime('%Y%m%d')}.csv"
        sample_data.to_csv(file_path, index=False)

        data = swace_instance.read(start_time, end_time)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1440
        assert all(
            col in data.columns for col in DSCOVR.MAG_FIELDS + DSCOVR.SWEPAM_FIELDS
        )

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

        if field in DSCOVR.MAG_FIELDS:
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

        swace_instance = DSCOVR(Path(__file__).parent / "data" / "DSCOVR")
        data = swace_instance.read(start_time, end_time, propagation=True)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5109
        assert all(
            col in data.columns for col in DSCOVR.MAG_FIELDS + DSCOVR.SWEPAM_FIELDS
        )
        assert any(data["file_name"] == "propagated from previous DSCOVR NOWCAST file")
        assert data.index.is_monotonic_increasing
