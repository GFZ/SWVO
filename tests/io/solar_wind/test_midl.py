# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from swvo.io.solar_wind import SWMIDL

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_midl"
EXPECTED_COLUMNS = [*SWMIDL.FIELDS, "pdyn"]


def make_dataset(start: datetime, minutes: int, nan_last: bool = False) -> xr.Dataset:
    """Build a synthetic MIDL dataset with known values on a one minute grid."""
    time = pd.date_range(start=start, periods=minutes, freq="1min", tz="UTC").tz_localize(None)

    def ramp(base: float, step: float) -> np.ndarray:
        values = base + step * np.arange(minutes, dtype=float)
        if nan_last:
            values[-1] = np.nan
        return values

    return xr.Dataset(
        {
            "Bx": ("time", ramp(3.0, 0.1)),
            "By": ("time", ramp(-4.0, 0.1)),
            "Bz": ("time", ramp(12.0, 0.1)),
            "Ux": ("time", ramp(-400.0, -1.0)),
            "Uy": ("time", ramp(30.0, 0.5)),
            "Uz": ("time", ramp(-60.0, 0.5)),
            "rho": ("time", ramp(5.0, 0.01)),
            "T": ("time", ramp(120000.0, 10.0)),
            # Provenance columns MIDL adds for the L1 target; SWVO must drop them.
            "X": ("time", np.full(minutes, 235.0)),
            "B_source": ("time", np.full(minutes, "ACE")),
            "B_interp": ("time", np.zeros(minutes, dtype=int)),
        },
        coords={"time": time},
    )


class TestSWMIDL:
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        TEST_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        yield

        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR, ignore_errors=True)

    @pytest.fixture
    def midl_instance(self):
        with patch.dict("os.environ", {SWMIDL.ENV_VAR_NAME: str(DATA_DIR)}):
            instance = SWMIDL()
            return instance

    @pytest.fixture
    def sample_dataset(self):
        return make_dataset(datetime(2024, 3, 15, 1, 0, tzinfo=timezone.utc), 4)

    def test_initialization_with_env_var(self):
        with patch.dict("os.environ", {SWMIDL.ENV_VAR_NAME: str(DATA_DIR)}):
            midl = SWMIDL()
            assert midl.data_dir == DATA_DIR

    def test_initialization_with_explicit_path(self):
        explicit_path = DATA_DIR / "explicit"
        midl = SWMIDL(data_dir=explicit_path)
        assert midl.data_dir == explicit_path

    def test_initialization_without_env_var(self):
        if SWMIDL.ENV_VAR_NAME in os.environ:
            del os.environ[SWMIDL.ENV_VAR_NAME]
        with pytest.raises(ValueError):
            SWMIDL()

    @pytest.mark.parametrize(
        "target,method,token",
        [
            ("L1", "ballistic", "L1"),
            ("l1", "ballistic", "L1"),
            (32, "ballistic", "32Re"),
            (20.25, "ballistic", "20p25Re"),
            (30, "mhd", "mhd_030Re"),
            (-5, "mhd", "mhd_-05Re"),
        ],
    )
    def test_target_token(self, midl_instance, target, method, token):
        normalized_target, normalized_method = midl_instance._normalize_target_method(target, method)
        assert midl_instance._make_target_token(normalized_target, normalized_method) == token

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"target": "bow_shock"},
            {"target": None},
            {"method": "kriging"},
            {"target": "L1", "method": "mhd"},
            {"target": 20.25, "method": "mhd"},
            {"target": 200, "method": "mhd"},
        ],
    )
    def test_read_with_invalid_target_method(self, midl_instance, kwargs):
        with pytest.raises(ValueError):
            midl_instance.read(
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                datetime(2020, 1, 2, tzinfo=timezone.utc),
                **kwargs,
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"target": "bow_shock"},
            {"target": None},
            {"method": "kriging"},
            {"target": "L1", "method": "mhd"},
            {"target": 20.25, "method": "mhd"},
            {"target": 200, "method": "mhd"},
        ],
    )
    def test_download_and_process_with_invalid_target_method(self, midl_instance, kwargs):
        with pytest.raises(ValueError):
            midl_instance.download_and_process(
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                datetime(2020, 1, 2, tzinfo=timezone.utc),
                **kwargs,
            )

    def test_get_processed_file_list(self, midl_instance):
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

        file_paths, time_intervals = midl_instance._get_processed_file_list(start_time, end_time, "L1")

        assert len(file_paths) == 366
        assert all(str(path).startswith(str(DATA_DIR)) for path in file_paths)
        assert all(path.name.startswith("MIDL_SW_L1_") for path in file_paths)
        assert file_paths[0] == DATA_DIR / "2020/01" / "MIDL_SW_L1_20200101.csv"
        assert len(time_intervals) == 366
        assert all(isinstance(interval, tuple) for interval in time_intervals)

    def test_source_urls(self, midl_instance):
        urls = midl_instance._source_urls(
            datetime(2023, 12, 20, tzinfo=timezone.utc),
            datetime(2024, 2, 3, tzinfo=timezone.utc),
            "L1",
        )

        assert urls == [
            f"{SWMIDL.URL}/2023/12/202312_L1.csv",
            f"{SWMIDL.URL}/2024/01/202401_L1.csv",
            f"{SWMIDL.URL}/2024/02/202402_L1.csv",
        ]

        mhd_urls = midl_instance._source_urls(
            datetime(2024, 1, 5, tzinfo=timezone.utc),
            datetime(2024, 1, 6, tzinfo=timezone.utc),
            "mhd_030Re",
        )
        assert mhd_urls == [f"{SWMIDL.URL}/2024/01/mhd/202401_mhd_030Re.csv"]

    def test_download_and_process(self, midl_instance, sample_dataset):
        assert midl_instance.url == SWMIDL.URL

        start_time = datetime(2024, 3, 15, 1, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 3, 15, 1, 3, tzinfo=timezone.utc)

        with patch("midl.load", return_value=sample_dataset) as mock_load:
            midl_instance.download_and_process(start_time, end_time)

        args, kwargs = mock_load.call_args
        # MIDL slices a naive index, so the UTC bounds must reach it without a timezone.
        assert args == (start_time.replace(tzinfo=None), end_time.replace(tzinfo=None), "L1")
        assert all(arg.tzinfo is None for arg in args[:2])
        assert kwargs == {"method": "ballistic", "coords": "GSM", "orbital_motion": False}

        expected_file = DATA_DIR / "2024/03" / "MIDL_SW_L1_20240315.csv"
        assert expected_file.exists()

        data = pd.read_csv(expected_file)
        assert len(data) == 4
        assert all(field in data.columns for field in EXPECTED_COLUMNS)
        assert not any(column in data.columns for column in ("X", "B_source", "B_interp"))
        assert midl_instance.url == f"{SWMIDL.URL}/2024/03/202403_L1.csv"

    def test_download_and_process_splits_days(self, midl_instance):
        start_time = datetime(2024, 3, 15, 23, 58, tzinfo=timezone.utc)
        end_time = datetime(2024, 3, 16, 0, 1, tzinfo=timezone.utc)
        dataset = make_dataset(start_time, 4)

        with patch("midl.load", return_value=dataset):
            midl_instance.download_and_process(start_time, end_time)

        first_day = DATA_DIR / "2024/03" / "MIDL_SW_L1_20240315.csv"
        second_day = DATA_DIR / "2024/03" / "MIDL_SW_L1_20240316.csv"

        assert len(pd.read_csv(first_day)) == 2
        assert len(pd.read_csv(second_day)) == 2

    def test_download_and_process_merges_existing(self, midl_instance):
        first_start = datetime(2024, 3, 15, 1, 0, tzinfo=timezone.utc)
        second_start = datetime(2024, 3, 15, 2, 0, tzinfo=timezone.utc)

        with patch("midl.load", return_value=make_dataset(first_start, 3)):
            midl_instance.download_and_process(first_start, first_start.replace(hour=1, minute=2))
        with patch("midl.load", return_value=make_dataset(second_start, 3)):
            midl_instance.download_and_process(second_start, second_start.replace(hour=2, minute=2))

        file_path = DATA_DIR / "2024/03" / "MIDL_SW_L1_20240315.csv"
        data = pd.read_csv(file_path)

        assert len(data) == 6
        assert not list(file_path.parent.glob("*.tmp"))

    def test_download_and_process_before_first_year(self, midl_instance):
        with pytest.raises(ValueError, match="only available from"):
            midl_instance.download_and_process(
                datetime(1990, 1, 1, tzinfo=timezone.utc),
                datetime(1990, 1, 2, tzinfo=timezone.utc),
            )

    def test_download_and_process_invalid_time_range(self, midl_instance):
        with pytest.raises(AssertionError, match="Start time must be before end time"):
            midl_instance.download_and_process(
                datetime(2024, 3, 16, tzinfo=timezone.utc),
                datetime(2024, 3, 15, tzinfo=timezone.utc),
            )

    def test_derived_quantities(self, midl_instance, sample_dataset):
        data = midl_instance._process_dataset(sample_dataset)

        assert list(data.columns) == EXPECTED_COLUMNS
        assert data.index.name == "t"
        assert str(data.index.tz) == "UTC"

        assert data["bx_gsm"].iloc[0] == pytest.approx(3.0)
        assert data["by_gsm"].iloc[0] == pytest.approx(-4.0)
        assert data["bz_gsm"].iloc[0] == pytest.approx(12.0)
        assert data["proton_density"].iloc[0] == pytest.approx(5.0)
        assert data["temperature"].iloc[0] == pytest.approx(120000.0)

        # bavg and speed are magnitudes derived from the MIDL vectors.
        assert data["bavg"].iloc[0] == pytest.approx(13.0)
        expected_speed = np.sqrt(400.0**2 + 30.0**2 + 60.0**2)
        assert data["speed"].iloc[0] == pytest.approx(expected_speed)
        assert data["pdyn"].iloc[0] == pytest.approx(2e-6 * 5.0 * expected_speed**2)

    def test_process_dataset_fills_gaps(self, midl_instance):
        dataset = make_dataset(datetime(2024, 3, 15, 1, 0, tzinfo=timezone.utc), 5)
        gapped = dataset.isel(time=[0, 1, 4])

        data = midl_instance._process_dataset(gapped)

        assert len(data) == 5
        assert data["bavg"].iloc[2:4].isna().all()

    def test_read_with_no_data(self, midl_instance):
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 1, 2, tzinfo=timezone.utc)

        data = midl_instance.read(start_time, end_time, download=False)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(col in data.columns for col in EXPECTED_COLUMNS)
        assert data.isna().all().all()

    def test_read_invalid_time_range(self, midl_instance):
        start_time = datetime(2020, 12, 31, tzinfo=timezone.utc)
        end_time = datetime(2020, 1, 1, tzinfo=timezone.utc)

        with pytest.raises(AssertionError, match="Start time must be before end time"):
            midl_instance.read(start_time, end_time)

    def test_read_with_existing_data(self, midl_instance):
        start_time = datetime(2024, 3, 15, tzinfo=timezone.utc)
        end_time = datetime(2024, 3, 15, 23, 59, 59, tzinfo=timezone.utc)

        with patch("midl.load", return_value=make_dataset(start_time, 1440, nan_last=True)):
            midl_instance.download_and_process(start_time, end_time)

        data = midl_instance.read(start_time, end_time)

        file_path = DATA_DIR / "2024/03" / "MIDL_SW_L1_20240315.csv"
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1440
        assert all(col in data.columns for col in EXPECTED_COLUMNS)
        assert data.loc[start_time, "bavg"] == pytest.approx(13.0)
        assert data.loc[start_time, "file_name"] == file_path
        # The all-NaN minute carries no provenance.
        assert data["file_name"].iloc[-1] is None

    def test_read_with_download(self, midl_instance, sample_dataset):
        start_time = datetime(2024, 3, 15, 1, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 3, 15, 1, 3, tzinfo=timezone.utc)

        with patch("midl.load", return_value=sample_dataset) as mock_load:
            data = midl_instance.read(start_time, end_time, download=True)

        assert mock_load.called
        assert data.loc[start_time, "bavg"] == pytest.approx(13.0)

    def test_with_propagation(self, midl_instance):
        start_time = datetime(2024, 3, 15, tzinfo=timezone.utc)
        end_time = datetime(2024, 3, 16, tzinfo=timezone.utc)

        with patch("midl.load", return_value=make_dataset(start_time, 2 * 1440)):
            midl_instance.download_and_process(start_time, end_time)

        data = midl_instance.read(start_time, end_time, propagation=True)

        assert isinstance(data, pd.DataFrame)
        assert all(col in data.columns for col in EXPECTED_COLUMNS)
        assert data.index.is_monotonic_increasing
        assert any(data["file_name"] == "propagated from previous MIDL file")

    def test_propagation_skipped_for_propagated_target(self, midl_instance, caplog):
        start_time = datetime(2024, 3, 15, tzinfo=timezone.utc)
        end_time = datetime(2024, 3, 16, tzinfo=timezone.utc)

        data = midl_instance.read(start_time, end_time, target=32, propagation=True)

        assert "already propagated" in caplog.text
        # No -1 day shift was applied either, so the output starts exactly at start_time
        # instead of the day before, and carries no propagation provenance.
        assert data.index[0] == start_time
        assert "file_name" not in data.columns
