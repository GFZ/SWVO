# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from swvo.io.solar_wind import SWIMAP

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_imap"
EXPECTED_COLUMNS = [*SWIMAP.MAG_FIELDS, *SWIMAP.SWAPI_FIELDS, "pdyn"]


def _mag_record(time_utc: str, vector: list, magnitude: float) -> dict:
    return {
        "time_utc": time_utc,
        "mag_B_GSM": vector,
        "mag_B_magnitude": magnitude,
        "instrument": "mag",
    }


def _swapi_record(time_utc: str, speed: float, density: float, temperature: float) -> dict:
    return {
        "time_utc": time_utc,
        "swapi_pseudo_proton_speed": speed,
        "swapi_pseudo_proton_density": density,
        "swapi_pseudo_proton_temperature": temperature,
        "instrument": "swapi",
    }


def _records_in_window(records: list, window_start: datetime, window_end: datetime) -> list:
    """Filter records the way the real API would for a given request window."""
    return [
        record
        for record in records
        if window_start <= datetime.strptime(record["time_utc"], "%Y-%m-%dT%H:%M:%S") < window_end
    ]


class TestSWIMAP:
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        TEST_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        yield

        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR, ignore_errors=True)

    @pytest.fixture
    def imap_instance(self):
        with patch.dict("os.environ", {SWIMAP.ENV_VAR_NAME: str(DATA_DIR)}):
            instance = SWIMAP()
            return instance

    @pytest.fixture
    def sample_mag_payload(self):
        return {
            "meta": {"count": 3, "type": "science", "instrument": "mag"},
            "data": [
                _mag_record("2026-03-15T01:00:00", [1.0, 2.0, -1.0], 5.0),
                _mag_record("2026-03-15T01:00:04", [3.0, 4.0, -3.0], 7.0),
                _mag_record("2026-03-15T01:02:00", [10.0, 10.0, 10.0], 20.0),
            ],
        }

    @pytest.fixture
    def sample_swapi_payload(self):
        return {
            "meta": {"count": 2, "type": "science", "instrument": "swapi"},
            "data": [
                _swapi_record("2026-03-15T01:00:06", 380.0, 4.0, 100000.0),
                _swapi_record("2026-03-15T01:02:10", 400.0, 5.0, 120000.0),
            ],
        }

    @staticmethod
    def _mock_get(mag_payload, swapi_payload, status_code_by_instrument=None):
        """Build a `requests.get` side effect that filters each fixture's records down to the
        requested `[time_utc_start, time_utc_end)` window, the way the real API does - so the same
        full-day fixture works regardless of how many sub-day requests `mag` is split into."""
        status_code_by_instrument = status_code_by_instrument or {}

        def mock_get(url, params=None, **kwargs):
            instrument = params["instrument"]
            if instrument in status_code_by_instrument:
                response = Mock()
                response.status_code = status_code_by_instrument[instrument]
                raise requests.HTTPError(response=response)

            window_start = datetime.strptime(params["time_utc_start"], "%Y-%m-%dT%H:%M:%S")
            window_end = datetime.strptime(params["time_utc_end"], "%Y-%m-%dT%H:%M:%S")
            source = mag_payload if instrument == "mag" else swapi_payload
            filtered = _records_in_window(source["data"], window_start, window_end)

            response = Mock()
            response.raise_for_status = Mock()
            response.json = Mock(return_value={"meta": {**source["meta"], "count": len(filtered)}, "data": filtered})
            return response

        return mock_get

    def test_initialization_with_env_var(self):
        with patch.dict("os.environ", {SWIMAP.ENV_VAR_NAME: str(DATA_DIR)}):
            imap = SWIMAP()
            assert imap.data_dir == DATA_DIR

    def test_initialization_with_explicit_path(self):
        explicit_path = DATA_DIR / "explicit"
        imap = SWIMAP(data_dir=explicit_path)
        assert imap.data_dir == explicit_path

    def test_initialization_without_env_var(self):
        if SWIMAP.ENV_VAR_NAME in os.environ:
            del os.environ[SWIMAP.ENV_VAR_NAME]
        with pytest.raises(ValueError):
            SWIMAP()

    def test_get_processed_file_list(self, imap_instance):
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

        file_paths, time_intervals = imap_instance._get_processed_file_list(start_time, end_time)

        assert len(file_paths) == 366
        assert all(str(path).startswith(str(DATA_DIR)) for path in file_paths)
        assert all(path.name.startswith("IMAP_SW_NOWCAST_") for path in file_paths)
        assert len(time_intervals) == 366
        assert all(isinstance(interval, tuple) for interval in time_intervals)

    def test_fetch_chunks_makes_hourly_requests_for_both_instruments(
        self, imap_instance, sample_mag_payload, sample_swapi_payload
    ):
        """Both `mag` and `swapi` are always requested as 24 fixed 1-hour windows spanning the
        requested day - never one big per-day request - since `mag`'s dense cadence exceeds the
        API's per-query size cap for a full day, and the two instruments are chunked uniformly."""
        start_time = datetime(2026, 3, 15, tzinfo=timezone.utc)
        end_time = datetime(2026, 3, 15, 23, 59, 59, tzinfo=timezone.utc)

        with patch("requests.get", side_effect=self._mock_get(sample_mag_payload, sample_swapi_payload)) as mock_get:
            imap_instance.download_and_process(start_time, end_time)

        day_start = start_time.replace(tzinfo=None)
        for instrument in ("mag", "swapi"):
            calls = [c for c in mock_get.call_args_list if c.kwargs["params"]["instrument"] == instrument]
            assert len(calls) == 24

            windows = sorted(
                (
                    datetime.strptime(c.kwargs["params"]["time_utc_start"], "%Y-%m-%dT%H:%M:%S"),
                    datetime.strptime(c.kwargs["params"]["time_utc_end"], "%Y-%m-%dT%H:%M:%S"),
                )
                for c in calls
            )
            assert windows[0][0] == day_start
            assert windows[-1][1] == day_start + timedelta(days=1)
            assert all(window_end - window_start == timedelta(hours=1) for window_start, window_end in windows)
            # The windows tile the day with no gaps or overlaps.
            assert all(windows[i][1] == windows[i + 1][0] for i in range(len(windows) - 1))

    def test_download_and_process(self, imap_instance, sample_mag_payload, sample_swapi_payload):
        start_time = datetime(2026, 3, 15, tzinfo=timezone.utc)
        end_time = datetime(2026, 3, 15, 23, 59, 59, tzinfo=timezone.utc)

        with patch("requests.get", side_effect=self._mock_get(sample_mag_payload, sample_swapi_payload)) as mock_get:
            imap_instance.download_and_process(start_time, end_time)

        expected_file = DATA_DIR / "2026/03" / "IMAP_SW_NOWCAST_20260315.csv"
        assert expected_file.exists()

        data = pd.read_csv(expected_file)
        assert len(data) == 1440
        assert all(field in data.columns for field in EXPECTED_COLUMNS)

        data["t"] = pd.to_datetime(data["t"], utc=True)
        data = data.set_index("t")

        # Two mag samples land in the 01:00 minute bin (:00 and :04 seconds) -> mean.
        row = data.loc["2026-03-15 01:00:00+00:00"]
        assert row["bx_gsm"] == pytest.approx(2.0)
        assert row["by_gsm"] == pytest.approx(3.0)
        assert row["bz_gsm"] == pytest.approx(-2.0)
        assert row["bavg"] == pytest.approx(6.0)
        assert row["speed"] == pytest.approx(380.0)
        assert row["proton_density"] == pytest.approx(4.0)
        assert row["temperature"] == pytest.approx(100000.0)
        assert row["pdyn"] == pytest.approx(2e-6 * 4.0 * 380.0**2)

        # A single mag sample lands alone in the 01:02 minute bin.
        row = data.loc["2026-03-15 01:02:00+00:00"]
        assert row["bx_gsm"] == pytest.approx(10.0)
        assert row["speed"] == pytest.approx(400.0)

        # A minute with no samples from either instrument is all-NaN.
        assert data.loc["2026-03-15 01:01:00+00:00"].isna().all()

        # 24 hourly `mag` requests + 24 hourly `swapi` requests.
        assert mock_get.call_count == 48
        assert isinstance(imap_instance.url, list)
        assert len(imap_instance.url) == 48
        assert all("instrument=" in url for url in imap_instance.url)

    def test_download_and_process_partial_instrument_outage(self, imap_instance, sample_mag_payload):
        start_time = datetime(2026, 3, 15, tzinfo=timezone.utc)
        end_time = datetime(2026, 3, 15, 23, 59, 59, tzinfo=timezone.utc)
        empty_swapi_payload = {"meta": {"count": 0, "type": "science", "instrument": "swapi"}, "data": []}

        with patch("requests.get", side_effect=self._mock_get(sample_mag_payload, empty_swapi_payload)):
            imap_instance.download_and_process(start_time, end_time)

        expected_file = DATA_DIR / "2026/03" / "IMAP_SW_NOWCAST_20260315.csv"
        assert expected_file.exists()

        data = pd.read_csv(expected_file)
        data["t"] = pd.to_datetime(data["t"], utc=True)
        data = data.set_index("t")

        row = data.loc["2026-03-15 01:00:00+00:00"]
        assert row["bx_gsm"] == pytest.approx(2.0)
        assert np.isnan(row["speed"])
        assert np.isnan(row["proton_density"])

    def test_download_and_process_400_error_skips_day(
        self, imap_instance, sample_mag_payload, sample_swapi_payload, caplog
    ):
        start_time = datetime(2026, 3, 15, tzinfo=timezone.utc)
        end_time = datetime(2026, 3, 16, 23, 59, 59, tzinfo=timezone.utc)

        def mock_get(url, params=None, **kwargs):
            request_date = params["time_utc_start"][:10]
            if request_date == "2026-03-15":
                response = Mock()
                response.status_code = 400
                raise requests.HTTPError(response=response)

            window_start = datetime.strptime(params["time_utc_start"], "%Y-%m-%dT%H:%M:%S")
            window_end = datetime.strptime(params["time_utc_end"], "%Y-%m-%dT%H:%M:%S")
            source = sample_mag_payload if params["instrument"] == "mag" else sample_swapi_payload
            filtered = _records_in_window(source["data"], window_start, window_end)

            response = Mock()
            response.raise_for_status = Mock()
            response.json = Mock(return_value={"meta": {**source["meta"], "count": len(filtered)}, "data": filtered})
            return response

        with patch("requests.get", side_effect=mock_get):
            with pytest.warns(RuntimeWarning, match="2026-03-15"):
                imap_instance.download_and_process(start_time, end_time)

        assert not (DATA_DIR / "2026/03" / "IMAP_SW_NOWCAST_20260315.csv").exists()
        assert (DATA_DIR / "2026/03" / "IMAP_SW_NOWCAST_20260316.csv").exists()

    def test_download_and_process_before_public_data_start_raises(self, imap_instance):
        """Data before `_PUBLIC_DATA_START` requires an API key this reader does not use, so the
        request should fail fast with a clear `ValueError` instead of attempting any chunk."""
        start_time = datetime(2026, 1, 31, tzinfo=timezone.utc)
        end_time = datetime(2026, 2, 2, tzinfo=timezone.utc)

        with patch("requests.get") as mock_get:
            with pytest.raises(ValueError, match="API key"):
                imap_instance.download_and_process(start_time, end_time)

        mock_get.assert_not_called()

    def test_api_key_required_response_is_not_retried(self, imap_instance):
        """If the API ever reports "API key required" for a chunk despite the upfront guard, that
        chunk should fail immediately (not burn through 4 retry attempts) with a clear reason."""
        start_time = datetime(2026, 3, 15, tzinfo=timezone.utc)
        end_time = datetime(2026, 3, 15, 1, 0, 0, tzinfo=timezone.utc)

        def mock_get(url, params=None, **kwargs):
            response = Mock()
            response.status_code = 400
            response.json = Mock(return_value={"message": "API key required for data prior to 2026-02-01T00:00:00"})
            return response

        with patch("requests.get", side_effect=mock_get) as mock_requests_get:
            with pytest.raises(ValueError, match="API key"):
                imap_instance._request_instrument("mag", start_time, end_time)

        # A single attempt: the "API key required" response is not treated as retryable.
        assert mock_requests_get.call_count == 1

    def test_read_with_no_data(self, imap_instance):
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 1, 2, tzinfo=timezone.utc)

        data = imap_instance.read(start_time, end_time, download=False)

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert all(col in data.columns for col in EXPECTED_COLUMNS)
        assert data.isna().all().all()

    def test_read_invalid_time_range(self, imap_instance):
        start_time = datetime(2020, 12, 31, tzinfo=timezone.utc)
        end_time = datetime(2020, 1, 1, tzinfo=timezone.utc)

        with pytest.raises(AssertionError, match="Start time must be before end time"):
            imap_instance.read(start_time, end_time)

    def test_read_with_existing_data(self, imap_instance):
        start_time = datetime(2026, 3, 15, tzinfo=timezone.utc)
        end_time = datetime(2026, 3, 15, 23, 59, 59, tzinfo=timezone.utc)

        t = pd.date_range(start=start_time, end=end_time, freq="min")

        sample_data = pd.DataFrame(
            {
                "t": t,
                "bx_gsm": np.full(len(t), 1.0),
                "by_gsm": np.full(len(t), 2.0),
                "bz_gsm": np.full(len(t), -1.0),
                "bavg": np.full(len(t), 5.0),
                "speed": np.full(len(t), 380.0),
                "proton_density": np.full(len(t), 4.2),
                "temperature": np.full(len(t), 145000.0),
                "pdyn": np.full(len(t), 2e-6 * 4.2 * 380.0**2),
            }
        )

        file_path = DATA_DIR / start_time.strftime("%Y/%m") / f"IMAP_SW_NOWCAST_{start_time.strftime('%Y%m%d')}.csv"
        file_path.parent.mkdir(parents=True)
        sample_data.to_csv(file_path, index=False)

        data = imap_instance.read(start_time, end_time)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1440
        assert all(col in data.columns for col in EXPECTED_COLUMNS)
        assert data.loc[start_time, "bavg"] == pytest.approx(sample_data["bavg"].iloc[0])
        assert data.loc[start_time, "file_name"] == file_path

    def test_expand_vector_handles_malformed_entry(self, imap_instance):
        series = pd.Series([[1.0, 2.0, 3.0], None, [1.0, 2.0], "not-a-list"])

        expanded = imap_instance._expand_vector(series, length=len(series))

        assert expanded.iloc[0].tolist() == [1.0, 2.0, 3.0]
        assert expanded.iloc[1].isna().all()
        assert expanded.iloc[2].isna().all()
        assert expanded.iloc[3].isna().all()
