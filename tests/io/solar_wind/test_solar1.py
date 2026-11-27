# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import gzip
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from swvo.io.solar_wind import SWSOLAR1

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_solar1"
DAY = datetime(2026, 11, 20, tzinfo=timezone.utc)
SWIPS_LINK = (
    "https://archive.example/oe_swips-l2_solar1_s20261120T000000Z_e20261120T235959Z_p20261121T021703Z_pub.nc.gz"
)


def swips_netcdf_gz(start_times, density, speed, temperature) -> bytes:
    """Minimal SWiPS L2 file: velocity along -X GSE so its magnitude is the speed."""
    velocity = np.zeros((len(speed), 3), dtype="float32")
    velocity[:, 0] = -np.asarray(speed, dtype="float32")
    velocity[np.asarray(speed) <= -9998] = -9999.99
    ds = xr.Dataset(
        {
            "proton_n_uncorr": ("sweep", np.asarray(density, dtype="float32")),
            "proton_t_uncorr": ("sweep", np.asarray(temperature, dtype="float32")),
            "proton_v_uncorr_gse": (("sweep", "coordinate"), velocity),
            "proton_n_corr": ("sweep", np.full(len(speed), -9999.0, dtype="float32")),
            "sweep_start_time": ("sweep", pd.DatetimeIndex(start_times).tz_localize(None).to_numpy()),
        }
    )
    return gzip.compress(ds.to_netcdf())


def values_payload(times, bx, by, bz, bt) -> dict:
    return {
        "status": {"code": 200, "message": "OK"},
        "data": {
            "time": [t.strftime("%Y-%m-%dT%H:%M:%S.000Z") for t in times],
            "mag-l3_solar1.b_gsm_min_x": bx,
            "mag-l3_solar1.b_gsm_min_y": by,
            "mag-l3_solar1.b_gsm_min_z": bz,
            "mag-l3_solar1.b_gsm_sphr_min_x": bt,
        },
    }


def files_payload(*records) -> dict:
    return {"status": {"code": 200}, "data": list(records)}


def response(json_data=None, content=None) -> Mock:
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json = Mock(return_value=json_data)
    mock_response.content = content
    return mock_response


class TestSWSOLAR1:
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        TEST_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        yield

        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR, ignore_errors=True)

    @pytest.fixture
    def solar1(self):
        with patch.dict("os.environ", {SWSOLAR1.ENV_VAR_NAME: str(DATA_DIR)}):
            return SWSOLAR1()

    @pytest.fixture
    def archive_responses(self):
        """Mock NCEI responses for one day: 3 MAG minutes and 4 SWiPS sweeps."""
        times = [DAY + timedelta(minutes=i) for i in range(3)]
        mag = values_payload(times, [1.0, -9999.0, 3.0], [2.0, 2.0, 2.0], [-1.0, -2.0, -3.0], [5.0, 5.0, 5.0])
        files = files_payload(
            {
                "id": "oe_swips-l2_solar1_s20261120T000000Z_e20261120T235959Z_p20261121T000000Z_pub.nc.gz",
                "time_coverage_start": "2026-11-20T00:00:00.000Z",
                "file_link": "https://archive.example/older_version.nc.gz",
            },
            {
                "id": Path(SWIPS_LINK).name,
                "time_coverage_start": "2026-11-20T00:00:00.000Z",
                "file_link": SWIPS_LINK,
            },
        )
        swips = swips_netcdf_gz(
            [
                DAY + timedelta(seconds=55),
                DAY + timedelta(minutes=1, seconds=10),
                DAY + timedelta(minutes=1, seconds=50),
                DAY + timedelta(minutes=2, seconds=5),
            ],
            density=[2.0, 0.0, 3.0, 4.0],
            speed=[400.0, -9999.99, 410.0, 420.0],
            temperature=[5e4, 0.0, 6e4, 7e4],
        )

        def mock_get(url, params=None, **kwargs):
            if url.endswith("/values"):
                return response(mag)
            if url.endswith("/files"):
                return response(files)
            if url == SWIPS_LINK:
                return response(content=swips)
            raise AssertionError(f"unexpected URL {url}")

        return mock_get

    def test_initialization_with_env_var(self):
        with patch.dict("os.environ", {SWSOLAR1.ENV_VAR_NAME: str(DATA_DIR)}):
            assert SWSOLAR1().data_dir == DATA_DIR

    def test_initialization_without_env_var(self):
        if SWSOLAR1.ENV_VAR_NAME in os.environ:
            del os.environ[SWSOLAR1.ENV_VAR_NAME]
        with pytest.raises(ValueError):
            SWSOLAR1()

    def test_get_processed_file_list(self, solar1):
        file_paths, time_intervals = solar1._get_processed_file_list(DAY, DAY + timedelta(days=1))

        assert file_paths == [
            DATA_DIR / "2026/11" / "SOLAR1_SW_NOWCAST_20261120.csv",
            DATA_DIR / "2026/11" / "SOLAR1_SW_NOWCAST_20261121.csv",
        ]
        assert len(time_intervals) == 2

    def test_download_before_data_start_downloads_nothing(self, solar1):
        with patch("requests.get") as mock_get:
            solar1.download_and_process(
                datetime(2026, 6, 1, tzinfo=timezone.utc), datetime(2026, 6, 2, tzinfo=timezone.utc)
            )

        mock_get.assert_not_called()
        assert not list(DATA_DIR.rglob("*.csv"))

    def test_download_across_data_start_is_clipped(self, solar1):
        with patch.object(SWSOLAR1, "_download_archive", return_value=None) as mock_archive:
            solar1.download_and_process(
                datetime(2026, 6, 1, tzinfo=timezone.utc), datetime(2026, 6, 10, tzinfo=timezone.utc)
            )

        assert mock_archive.call_args.args[0] == SWSOLAR1.DATA_START

    def test_download_and_process_archive(self, solar1, archive_responses):
        with patch("requests.get", side_effect=archive_responses) as mock_get:
            solar1.download_and_process(DAY, DAY + timedelta(hours=1))

        requested = [call.args[0] for call in mock_get.call_args_list]
        assert SWIPS_LINK in requested
        assert "https://archive.example/older_version.nc.gz" not in requested
        assert not any("rtsw" in url for url in requested)

        data = pd.read_csv(DATA_DIR / "2026/11" / "SOLAR1_SW_NOWCAST_20261120.csv", index_col="t", parse_dates=True)
        assert len(data) == 3
        assert (data["data_source"] == "archive").all()
        # MAG fill value is masked.
        assert np.isnan(data["bx_gsm"].iloc[1])
        assert data["bavg"].iloc[0] == pytest.approx(5.0)
        # Sweep start 00:00:55 is tagged 00:00. Minute 00:01 has a fill sweep then a valid one; the valid one is kept.
        assert data["speed"].tolist() == pytest.approx([400.0, 410.0, 420.0])
        assert data["proton_density"].tolist() == pytest.approx([2.0, 3.0, 4.0])
        assert data["pdyn"].iloc[0] == pytest.approx(2e-6 * 2.0 * 400.0**2)

    def test_parse_swips_drops_whole_bad_sweep(self, solar1):
        ds = xr.open_dataset(
            gzip.decompress(
                swips_netcdf_gz(
                    [DAY, DAY + timedelta(minutes=1), DAY + timedelta(minutes=2)],
                    density=[2.0, 3.0, 4.0],
                    speed=[400.0, 100.0, 450.0],
                    temperature=[5e4, 6e4, 0.0],
                )
            )
        )
        data = solar1._parse_swips(ds)

        assert data["speed"].iloc[0] == pytest.approx(400.0)
        # Speed below the instrument range and zero temperature: the whole sweep is dropped.
        assert data.iloc[1].isna().all()
        assert data.iloc[2].isna().all()

    def test_list_swips_files_picks_latest_version(self, solar1, archive_responses):
        with patch("requests.get", side_effect=archive_responses):
            links = solar1._list_swips_files(DAY.date(), DAY.date())

        assert links == {DAY.date(): SWIPS_LINK}

    def test_download_realtime_filters_source(self, solar1):
        mag = [
            {
                "time_tag": "2026-11-20T00:00:00",
                "source": "SOLAR1",
                "bt": 5.0,
                "bx_gsm": 1.0,
                "by_gsm": 2.0,
                "bz_gsm": -1.0,
            },
            {
                "time_tag": "2026-11-20T00:00:00",
                "source": "ACE",
                "bt": 9.0,
                "bx_gsm": 9.0,
                "by_gsm": 9.0,
                "bz_gsm": 9.0,
            },
            {
                "time_tag": "2026-11-20T00:01:02",
                "source": "SOLAR1",
                "bt": 6.0,
                "bx_gsm": 1.5,
                "by_gsm": 2.5,
                "bz_gsm": -1.5,
            },
        ]
        wind = [
            {
                "time_tag": "2026-11-20T00:00:00",
                "source": "SOLAR1",
                "proton_speed": 400.0,
                "proton_density": 2.0,
                "proton_temperature": 5e4,
            },
            {
                "time_tag": "2026-11-20T00:00:00",
                "source": "IMAP",
                "proton_speed": 900.0,
                "proton_density": 9.0,
                "proton_temperature": 9e4,
            },
            {
                "time_tag": "2026-11-20T00:01:00",
                "source": "SOLAR1",
                "proton_speed": None,
                "proton_density": 2.0,
                "proton_temperature": 5e4,
            },
        ]

        def mock_get(url, **kwargs):
            return response(mag if url.endswith(SWSOLAR1.RTSW_NAME_MAG) else wind)

        with patch("requests.get", side_effect=mock_get):
            data = solar1._download_realtime()

        assert solar1.url == [SWSOLAR1.RTSW_URL + SWSOLAR1.RTSW_NAME_MAG, SWSOLAR1.RTSW_URL + SWSOLAR1.RTSW_NAME_WIND]
        assert list(data.index) == [DAY, DAY + timedelta(minutes=1)]
        assert data["bavg"].tolist() == pytest.approx([5.0, 6.0])
        assert data["speed"].iloc[0] == pytest.approx(400.0)
        assert data["pdyn"].iloc[0] == pytest.approx(2e-6 * 2.0 * 400.0**2)
        # Missing speed drops the whole plasma row.
        assert data[["speed", "proton_density", "temperature"]].iloc[1].isna().all()

    def _frame(self, values: dict, minutes: int = 3) -> pd.DataFrame:
        index = pd.date_range(DAY, periods=minutes, freq="1min", tz="UTC")
        data = pd.DataFrame({field: np.nan for field in [*SWSOLAR1.MAG_FIELDS, *SWSOLAR1.PLASMA_FIELDS]}, index=index)
        for column, column_values in values.items():
            data[column] = column_values
        data["pdyn"] = 2e-6 * data["proton_density"] * data["speed"] ** 2
        return data

    def test_archive_takes_precedence_over_realtime(self, solar1):
        archive = self._frame(
            {"speed": [400.0, np.nan, 420.0], "proton_density": [2.0, np.nan, 4.0], "temperature": [5e4, np.nan, 7e4]}
        )
        realtime = self._frame(
            {"speed": [999.0, 410.0, 999.0], "proton_density": [9.0, 3.0, 9.0], "temperature": [9e4, 6e4, 9e4]}
        )
        file_path = DATA_DIR / "2026/11" / "SOLAR1_SW_NOWCAST_20261120.csv"

        # Real-time first, then archive: the archive overwrites.
        solar1._save_processed_data(realtime, source="realtime")
        solar1._save_processed_data(archive, source="archive")
        data = solar1._read_single_file(file_path)
        assert data["speed"].tolist() == pytest.approx([400.0, 410.0, 420.0])
        assert data["data_source"].tolist() == ["archive", "realtime_no_archive", "archive"]

        # Archive first, then real-time: real-time only fills gaps.
        file_path.unlink()
        solar1._save_processed_data(archive, source="archive")
        solar1._save_processed_data(realtime, source="realtime")
        data = solar1._read_single_file(file_path)
        assert data["speed"].tolist() == pytest.approx([400.0, 410.0, 420.0])
        assert data["data_source"].tolist() == ["archive", "realtime", "archive"]

    def test_needs_archive_refresh(self, solar1):
        realtime = self._frame(
            {"speed": [400.0, 410.0, 420.0], "proton_density": [2.0, 3.0, 4.0], "temperature": [5e4, 6e4, 7e4]}
        )
        solar1._save_processed_data(realtime, source="realtime")
        file_path = DATA_DIR / "2026/11" / "SOLAR1_SW_NOWCAST_20261120.csv"

        assert not solar1._needs_archive_refresh(file_path, DAY + timedelta(hours=1))
        assert solar1._needs_archive_refresh(file_path, DAY + SWSOLAR1.ARCHIVE_LATENCY + timedelta(hours=1))

    def test_read_refreshes_old_realtime_days(self, solar1, archive_responses):
        realtime = self._frame(
            {"speed": [999.0, 999.0, 999.0], "proton_density": [9.0, 9.0, 9.0], "temperature": [9e4, 9e4, 9e4]}
        )
        solar1._save_processed_data(realtime, source="realtime")

        with patch("requests.get", side_effect=archive_responses):
            data = solar1.read(DAY, DAY + timedelta(minutes=2), download=True)

        assert data["speed"].tolist() == pytest.approx([400.0, 410.0, 420.0])
        assert "data_source" not in data.columns

    def test_read_with_no_data(self, solar1):
        data = solar1.read(DAY, DAY + timedelta(days=1))

        assert len(data) == 1441
        assert all(col in data.columns for col in [*SWSOLAR1.MAG_FIELDS, *SWSOLAR1.PLASMA_FIELDS])

    def test_read_invalid_time_range(self, solar1):
        with pytest.raises(ValueError):
            solar1.read(DAY + timedelta(days=1), DAY)

    def test_read_with_propagation(self, solar1):
        values = {
            "speed": [400.0] * 3,
            "proton_density": [2.0] * 3,
            "temperature": [5e4] * 3,
            "bavg": [5.0] * 3,
            "bx_gsm": [1.0] * 3,
            "by_gsm": [1.0] * 3,
            "bz_gsm": [1.0] * 3,
        }
        # Just before midnight, so propagation (about 1 hour at 400 km/s) carries it into the next day.
        late_previous_day = self._frame(values)
        late_previous_day.index = late_previous_day.index + timedelta(hours=23, minutes=57)
        next_day = self._frame(values)
        next_day.index = next_day.index + timedelta(days=1)
        solar1._save_processed_data(late_previous_day, source="archive")
        solar1._save_processed_data(next_day, source="archive")

        data = solar1.read(DAY + timedelta(days=1), DAY + timedelta(days=1, hours=1), propagation=True)

        assert data.index.is_monotonic_increasing
        assert any(data["file_name"] == "propagated from previous SOLAR1 NOWCAST file")
