# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests
import xarray as xr

from swvo.io.solar_wind.enlil import (
    SWENLIL,
    SWENLIL_BKG,
    SWENLIL_CME,
    _spherical_vector_to_cartesian,
)

TEST_DIR = Path("test_data")
DATA_DIR = TEST_DIR / "mock_enlil"

EXPECTED_COLUMNS = ["bx_gsm", "by_gsm", "bz_gsm", "bavg", "speed", "proton_density", "temperature", "pdyn"]


def _write_sample_nc(path: Path, refdate: datetime, num_samples: int = 5, seconds_step: float = 3600.0) -> None:
    """Write a minimal ENLIL-shaped NetCDF file with an `earth_t`-length Earth_* timeline."""
    au_m = 1.496e11
    variables = {
        # Start well before REFDATE so tests can exercise the start_time cutoff.
        "Earth_TIME": np.arange(num_samples) * seconds_step - 2 * seconds_step,
        "Earth_X1": np.full(num_samples, 1.01 * au_m),
        "Earth_X2": np.full(num_samples, np.pi / 2),
        "Earth_X3": np.linspace(0, 0.01, num_samples),
        "Earth_B1": np.full(num_samples, 3e-9),
        "Earth_B2": np.full(num_samples, 1e-9),
        "Earth_B3": np.full(num_samples, 2e-9),
        "Earth_V1": np.full(num_samples, 4.0e5),
        "Earth_V2": np.full(num_samples, 3.0e4),
        "Earth_V3": np.full(num_samples, 4.0e4),
        "Earth_Density": np.full(num_samples, 5e-21),
        "Earth_Temperature": np.full(num_samples, 1e5),
    }

    dataset = xr.Dataset(
        {name: ("earth_t", values.astype("f4")) for name, values in variables.items()},
        attrs={"REFDATE_CAL": refdate.strftime("%Y-%m-%dT%H:%M:%S")},
    )
    # Real runs carry no fill value on these; without this xarray would add one of its own and
    # the fixture would stop matching the files it stands in for.
    dataset.to_netcdf(path, encoding={name: {"_FillValue": None} for name in variables})


def _write_sample_archive(archive_path: Path, nc_name: str, refdate: datetime, **nc_kwargs) -> None:
    """Write a `.tar.gz` containing one ENLIL-shaped `.nc` file, matching the real archive layout."""
    tmp_dir = archive_path.parent / "tar_build"
    tmp_dir.mkdir(exist_ok=True)
    nc_path = tmp_dir / nc_name
    _write_sample_nc(nc_path, refdate, **nc_kwargs)

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(nc_path, arcname=f"./{nc_name}")

    shutil.rmtree(tmp_dir)


class TestSWENLIL:
    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        TEST_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)

        yield

        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR, ignore_errors=True)
        temp_dirs = list(Path(".").glob("temp_sw_enlil_wget_*"))
        for d in temp_dirs:
            shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def bkg_instance(self):
        with patch.dict("os.environ", {SWENLIL.ENV_VAR_NAME: str(DATA_DIR)}):
            return SWENLIL_BKG()

    @pytest.fixture
    def cme_instance(self):
        with patch.dict("os.environ", {SWENLIL.ENV_VAR_NAME: str(DATA_DIR)}):
            return SWENLIL_CME()

    def test_parent_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            SWENLIL(data_dir=DATA_DIR)

    @pytest.mark.parametrize(
        ("cls", "mode", "label", "product"),
        [
            (SWENLIL_BKG, "bkg", "enlil_bkg", "swpc_wsaenlil_bkg"),
            (SWENLIL_CME, "cme", "enlil_cme", "swpc_wsaenlil_cme"),
        ],
    )
    def test_subclass_mode_attributes(self, cls, mode, label, product):
        assert (cls.MODE, cls.LABEL) == (mode, label)
        assert cls(data_dir=DATA_DIR)._product == product

    @pytest.mark.parametrize("cls", [SWENLIL_BKG, SWENLIL_CME])
    def test_initialization_with_env_var(self, cls):
        with patch.dict("os.environ", {SWENLIL.ENV_VAR_NAME: str(DATA_DIR)}):
            instance = cls()
            assert instance.data_dir == DATA_DIR

    @pytest.mark.parametrize("cls", [SWENLIL_BKG, SWENLIL_CME])
    def test_initialization_with_explicit_path(self, cls):
        explicit_path = DATA_DIR / "explicit"
        instance = cls(data_dir=explicit_path)
        assert instance.data_dir == explicit_path

    @pytest.mark.parametrize("cls", [SWENLIL_BKG, SWENLIL_CME])
    def test_initialization_without_env_var(self, cls, monkeypatch):
        monkeypatch.delenv(SWENLIL.ENV_VAR_NAME, raising=False)
        with pytest.raises(ValueError):
            cls()

    def test_url_falls_back_to_api_url_before_any_request(self, bkg_instance):
        assert bkg_instance.url == SWENLIL.API_URL

    def test_run_time_from_id_bkg(self):
        assert SWENLIL._run_time_from_id("swpc_wsaenlil_bkg_20250818_0000.tar.gz") == "0000"

    def test_run_time_from_id_cme(self):
        assert SWENLIL._run_time_from_id("swpc_wsaenlil_cme_20250818_0237.tar.gz") == "0237"

    def test_file_path_for(self, cme_instance):
        target_date = datetime(2025, 8, 18, tzinfo=timezone.utc)
        path = cme_instance._file_path_for(target_date, "0237")
        assert path == DATA_DIR / "2025/08" / "ENLIL_FORECAST_cme_20250818_0237.csv"

    def test_csvs_for_sorts_by_run_time(self, cme_instance):
        target_date = datetime(2024, 5, 3, tzinfo=timezone.utc)
        directory = DATA_DIR / "2024/05"
        directory.mkdir(parents=True)
        for run_time in ["1610", "0613", "1414"]:
            (directory / f"ENLIL_FORECAST_cme_20240503_{run_time}.csv").touch()

        paths = cme_instance._csvs_for(target_date)

        assert [p.name.split("_")[-1].removesuffix(".csv") for p in paths] == ["0613", "1414", "1610"]

    def test_csvs_for_only_lists_its_own_mode(self, bkg_instance, cme_instance):
        target_date = datetime(2024, 5, 3, tzinfo=timezone.utc)
        directory = DATA_DIR / "2024/05"
        directory.mkdir(parents=True)
        (directory / "ENLIL_FORECAST_bkg_20240503_0000.csv").touch()
        (directory / "ENLIL_FORECAST_cme_20240503_0613.csv").touch()

        assert [p.name for p in bkg_instance._csvs_for(target_date)] == ["ENLIL_FORECAST_bkg_20240503_0000.csv"]
        assert [p.name for p in cme_instance._csvs_for(target_date)] == ["ENLIL_FORECAST_cme_20240503_0613.csv"]

    def test_csvs_for_missing_directory(self, bkg_instance):
        assert bkg_instance._csvs_for(datetime(2099, 1, 1, tzinfo=timezone.utc)) == []

    def test_warn_if_pre_archive_cutoff_logs_before_cutoff(self, bkg_instance, caplog):
        with caplog.at_level("CRITICAL"):
            bkg_instance._warn_if_pre_archive_cutoff(datetime(2022, 1, 1, tzinfo=timezone.utc))
        assert SWENLIL.PRE_ARCHIVE_CUTOFF_MESSAGE in caplog.text

    def test_warn_if_pre_archive_cutoff_silent_after_cutoff(self, bkg_instance, caplog):
        with caplog.at_level("CRITICAL"):
            bkg_instance._warn_if_pre_archive_cutoff(datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert caplog.text == ""

    def test_spherical_vector_to_cartesian_radial_unit_vector(self):
        theta, phi = np.array([np.pi / 2]), np.array([0.0])
        x, y, z = _spherical_vector_to_cartesian(theta, phi, np.array([1.0]), np.array([0.0]), np.array([0.0]))
        assert x[0] == pytest.approx(1.0)
        assert y[0] == pytest.approx(0.0, abs=1e-10)
        assert z[0] == pytest.approx(0.0, abs=1e-10)

    def test_transform_b_to_gsm_preserves_magnitude(self, bkg_instance):
        r = np.array([1.5e11, 1.5e11])
        theta = np.array([np.pi / 2, np.pi / 2.1])
        phi = np.array([0.0, 0.1])
        b_r = np.array([3e-9, 2e-9])
        b_theta = np.array([1e-9, 0.5e-9])
        b_phi = np.array([2e-9, 1.5e-9])
        timestamps = [
            datetime(2024, 5, 3, tzinfo=timezone.utc),
            datetime(2024, 5, 3, 1, tzinfo=timezone.utc),
        ]

        bx, by, bz = bkg_instance._transform_b_to_gsm(r, theta, phi, b_r, b_theta, b_phi, timestamps)

        gsm_magnitude = np.sqrt(bx**2 + by**2 + bz**2)
        original_magnitude = np.sqrt(b_r**2 + b_theta**2 + b_phi**2)
        # The GSM transform goes through a position-difference trick (see
        # _transform_b_to_gsm's docstring) that subtracts two very close large
        # numbers, so float precision loss is expected; this only checks the
        # magnitude is preserved to a physically negligible tolerance.
        np.testing.assert_allclose(gsm_magnitude, original_magnitude, rtol=1e-4)

    def test_extract_nc_file(self, bkg_instance):
        refdate = datetime(2024, 5, 3, tzinfo=timezone.utc)
        archive_path = DATA_DIR / "sample.tar.gz"
        _write_sample_archive(archive_path, "wsa_enlil.mrid00000000.suball.nc", refdate)

        nc_path = bkg_instance._extract_nc_file(archive_path, DATA_DIR)

        assert nc_path.name == "wsa_enlil.mrid00000000.suball.nc"
        assert nc_path.exists()
        assert not archive_path.exists()

    def test_extract_nc_file_raises_when_no_nc_member(self, bkg_instance):
        archive_path = DATA_DIR / "empty.tar.gz"
        with tarfile.open(archive_path, "w:gz"):
            pass

        with pytest.raises(FileNotFoundError, match="Expected exactly one .nc file"):
            bkg_instance._extract_nc_file(archive_path, DATA_DIR)

    def test_extract_nc_file_raises_when_multiple_nc_members(self, bkg_instance, tmp_path):
        refdate = datetime(2024, 5, 3, tzinfo=timezone.utc)
        _write_sample_nc(tmp_path / "one.nc", refdate)
        _write_sample_nc(tmp_path / "two.nc", refdate)

        archive_path = DATA_DIR / "double.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(tmp_path / "one.nc", arcname="./one.nc")
            tar.add(tmp_path / "two.nc", arcname="./two.nc")

        with pytest.raises(FileNotFoundError, match="found 2"):
            bkg_instance._extract_nc_file(archive_path, DATA_DIR)

    def test_read_single_file_columns_and_units(self, bkg_instance):
        refdate = datetime(2024, 5, 3, tzinfo=timezone.utc)
        nc_path = DATA_DIR / "sample.nc"
        _write_sample_nc(nc_path, refdate)

        df = bkg_instance._read_single_file(nc_path, refdate)

        assert set(df.columns) == set(EXPECTED_COLUMNS)
        assert df.index.tz is not None
        assert df.index.is_monotonic_increasing

    def test_read_single_file_discards_data_before_start_time(self, bkg_instance):
        refdate = datetime(2024, 5, 3, tzinfo=timezone.utc)
        nc_path = DATA_DIR / "sample.nc"
        # Earth_TIME starts 2 hours before REFDATE (see _write_sample_nc).
        _write_sample_nc(nc_path, refdate, num_samples=5, seconds_step=3600.0)

        df = bkg_instance._read_single_file(nc_path, refdate)

        assert (df.index >= refdate).all()
        assert len(df) == 3  # only samples at/after REFDATE are kept out of 5

    def test_read_single_file_speed_uses_full_velocity_magnitude(self, bkg_instance):
        refdate = datetime(2024, 5, 3, tzinfo=timezone.utc)
        nc_path = DATA_DIR / "sample.nc"
        _write_sample_nc(nc_path, refdate)

        df = bkg_instance._read_single_file(nc_path, refdate)

        v_r, v_theta, v_phi = 4.0e5, 3.0e4, 4.0e4
        expected_speed_km_s = np.sqrt(v_r**2 + v_theta**2 + v_phi**2) / 1000.0
        assert df["speed"].iloc[0] == pytest.approx(expected_speed_km_s, rel=1e-5)
        # Sanity: this must differ from radial-only speed to prove V2/V3 are included.
        assert df["speed"].iloc[0] != pytest.approx(v_r / 1000.0, rel=1e-5)

    def test_read_single_file_bavg_matches_gsm_vector_norm(self, bkg_instance):
        refdate = datetime(2024, 5, 3, tzinfo=timezone.utc)
        nc_path = DATA_DIR / "sample.nc"
        _write_sample_nc(nc_path, refdate)

        df = bkg_instance._read_single_file(nc_path, refdate)

        expected_bavg = np.sqrt(df["bx_gsm"] ** 2 + df["by_gsm"] ** 2 + df["bz_gsm"] ** 2)
        pd.testing.assert_series_equal(df["bavg"], expected_bavg, check_names=False)

    def test_read_single_file_pdyn_formula(self, bkg_instance):
        refdate = datetime(2024, 5, 3, tzinfo=timezone.utc)
        nc_path = DATA_DIR / "sample.nc"
        _write_sample_nc(nc_path, refdate)

        df = bkg_instance._read_single_file(nc_path, refdate)

        expected_pdyn = 2e-6 * df["proton_density"] * df["speed"] ** 2
        pd.testing.assert_series_equal(df["pdyn"], expected_pdyn, check_names=False)

    def test_read_single_file_empty_when_all_before_start_time(self, bkg_instance):
        refdate = datetime(2024, 5, 3, tzinfo=timezone.utc)
        nc_path = DATA_DIR / "sample.nc"
        _write_sample_nc(nc_path, refdate)

        df = bkg_instance._read_single_file(nc_path, refdate + pd.Timedelta(days=10))

        assert len(df) == 0
        assert list(df.columns) == [
            "bx_gsm",
            "by_gsm",
            "bz_gsm",
            "speed",
            "proton_density",
            "temperature",
            "bavg",
            "pdyn",
        ]

    def _mock_files_response(self, entries: list[dict]) -> Mock:
        response = Mock()
        response.raise_for_status = Mock()
        response.json = Mock(return_value={"status": {"code": 200}, "data": entries})
        return response

    def _bkg_entry(self, date_str: str, file_link: str = "https://example.com/bkg.tar.gz") -> dict:
        return {
            "id": f"swpc_wsaenlil_bkg_{date_str}_0000.tar.gz",
            "time_coverage_start": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T00:00:00.000Z",
            "product": "swpc_wsaenlil_bkg",
            "file_link": file_link,
        }

    def _cme_entry(self, date_str: str, run_time: str, file_link: str = "https://example.com/cme.tar.gz") -> dict:
        return {
            "id": f"swpc_wsaenlil_cme_{date_str}_{run_time}.tar.gz",
            "time_coverage_start": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}T00:00:00.000Z",
            "product": "swpc_wsaenlil_cme",
            "file_link": file_link,
        }

    def test_find_file_entries_filters_by_product(self, bkg_instance, cme_instance):
        entries = [self._bkg_entry("20240503"), self._cme_entry("20240503", "0613")]
        target_date = datetime(2024, 5, 3, tzinfo=timezone.utc)

        with patch("requests.get", return_value=self._mock_files_response(entries)):
            bkg_entries = bkg_instance._find_file_entries(target_date)
            cme_entries = cme_instance._find_file_entries(target_date)

        assert [e["product"] for e in bkg_entries] == ["swpc_wsaenlil_bkg"]
        assert [e["product"] for e in cme_entries] == ["swpc_wsaenlil_cme"]

    def test_download_and_process_raises_when_bkg_missing(self, bkg_instance):
        with patch("requests.get", return_value=self._mock_files_response([])):
            with pytest.raises(FileNotFoundError, match="No ENLIL bkg run found"):
                bkg_instance.download_and_process(datetime(2099, 1, 1, tzinfo=timezone.utc))

    def test_download_and_process_raises_when_lookup_fails(self, bkg_instance):
        with patch("requests.get", side_effect=requests.ConnectionError("boom")):
            with pytest.raises(requests.ConnectionError, match="boom"):
                bkg_instance.download_and_process(datetime(2024, 5, 3, tzinfo=timezone.utc))

    def test_download_and_process_bkg_only_creates_csv(self, bkg_instance, tmp_path):
        target_date = datetime(2024, 5, 3, tzinfo=timezone.utc)
        archive_path = tmp_path / "bkg.tar.gz"
        _write_sample_archive(archive_path, "wsa_enlil.mrid1.suball.nc", target_date)

        entries = [self._bkg_entry("20240503")]

        def fake_get(url, *args, **kwargs):
            if url == SWENLIL.API_URL:
                return self._mock_files_response(entries)
            response = Mock()
            response.raise_for_status = Mock()
            response.iter_content = Mock(return_value=[archive_path.read_bytes()])
            return response

        with patch("requests.get", side_effect=fake_get):
            bkg_instance.download_and_process(target_date)

        expected_file = DATA_DIR / "2024/05" / "ENLIL_FORECAST_bkg_20240503_0000.csv"
        assert expected_file.exists()
        assert not (DATA_DIR / "2024/05" / "ENLIL_FORECAST_cme_20240503_0000.csv").exists()

        temp_dirs = list(Path(".").glob("temp_sw_enlil_wget_*"))
        assert temp_dirs == []

    def test_download_and_process_missing_cme_is_not_an_error(self, cme_instance):
        target_date = datetime(2024, 5, 3, tzinfo=timezone.utc)
        entries = [self._bkg_entry("20240503")]  # no cme entry

        with patch("requests.get", return_value=self._mock_files_response(entries)) as mock_get:
            cme_instance.download_and_process(target_date)  # must not raise

        assert list(DATA_DIR.rglob("*.csv")) == []
        # Only the archive lookup happened; the bkg entry is not this class's to download.
        assert all(call.args[0] == SWENLIL.API_URL for call in mock_get.call_args_list)

    def test_download_and_process_cme_keeps_real_run_time(self, cme_instance, tmp_path):
        target_date = datetime(2024, 5, 3, tzinfo=timezone.utc)
        cme_archive = tmp_path / "cme.tar.gz"
        _write_sample_archive(
            cme_archive, "wsa_enlil.mridcme.suball.nc", target_date + pd.Timedelta(hours=6, minutes=13)
        )

        entries = [
            self._bkg_entry("20240503", file_link="https://example.com/bkg.tar.gz"),
            self._cme_entry("20240503", "0613", file_link="https://example.com/cme.tar.gz"),
        ]

        def fake_get(url, *args, **kwargs):
            if url == SWENLIL.API_URL:
                return self._mock_files_response(entries)
            response = Mock()
            response.raise_for_status = Mock()
            response.iter_content = Mock(return_value=[cme_archive.read_bytes()])
            return response

        with patch("requests.get", side_effect=fake_get):
            cme_instance.download_and_process(target_date)

        assert (DATA_DIR / "2024/05" / "ENLIL_FORECAST_cme_20240503_0613.csv").exists()
        # The bkg entry of the same date belongs to SWENLIL_BKG and is left alone.
        assert list(DATA_DIR.rglob("ENLIL_FORECAST_bkg_*.csv")) == []
        assert not (DATA_DIR / "2024/05" / "ENLIL_FORECAST_cme_20240503_0000.csv").exists()

    def test_download_and_process_skips_existing_file(self, bkg_instance):
        target_date = datetime(2024, 5, 3, tzinfo=timezone.utc)
        file_path = DATA_DIR / "2024/05" / "ENLIL_FORECAST_bkg_20240503_0000.csv"
        file_path.parent.mkdir(parents=True)
        file_path.write_text("t,bx_gsm\n2024-05-03T00:00:00+00:00,1.0\n")

        with patch("requests.get") as mock_get:
            mock_get.return_value = self._mock_files_response([self._bkg_entry("20240503")])
            bkg_instance.download_and_process(target_date)

        # The files-API lookup still happens, but no archive download is attempted.
        for call in mock_get.call_args_list:
            assert call.args[0] != "https://example.com/bkg.tar.gz"

    def test_download_and_process_reprocess_files_forces_redownload(self, bkg_instance, tmp_path):
        target_date = datetime(2024, 5, 3, tzinfo=timezone.utc)
        file_path = DATA_DIR / "2024/05" / "ENLIL_FORECAST_bkg_20240503_0000.csv"
        file_path.parent.mkdir(parents=True)
        file_path.write_text("stale content")

        archive_path = tmp_path / "bkg.tar.gz"
        _write_sample_archive(archive_path, "wsa_enlil.mrid1.suball.nc", target_date)
        entries = [self._bkg_entry("20240503")]

        def fake_get(url, *args, **kwargs):
            if url == SWENLIL.API_URL:
                return self._mock_files_response(entries)
            response = Mock()
            response.raise_for_status = Mock()
            response.iter_content = Mock(return_value=[archive_path.read_bytes()])
            return response

        with patch("requests.get", side_effect=fake_get):
            bkg_instance.download_and_process(target_date, reprocess_files=True)

        assert "stale content" not in file_path.read_text()

    def test_download_and_process_records_url(self, bkg_instance):
        target_date = datetime(2099, 1, 1, tzinfo=timezone.utc)
        with patch("requests.get", return_value=self._mock_files_response([])):
            with pytest.raises(FileNotFoundError):
                bkg_instance.download_and_process(target_date)

        assert bkg_instance.url.startswith(SWENLIL.API_URL)
        assert "start_time=2099-01-01T00%3A00%3A00" in bkg_instance.url

    def test_download_and_process_invalid_time_range(self, bkg_instance):
        with pytest.raises(ValueError, match="start_time must be before end_time"):
            bkg_instance.download_and_process(
                datetime(2024, 5, 5, tzinfo=timezone.utc), datetime(2024, 5, 1, tzinfo=timezone.utc)
            )

    def test_download_and_process_warns_before_cutoff(self, bkg_instance, caplog):
        with patch("requests.get", return_value=self._mock_files_response([])):
            with caplog.at_level("CRITICAL"), pytest.raises(FileNotFoundError):
                bkg_instance.download_and_process(datetime(2022, 1, 1, tzinfo=timezone.utc))

        assert SWENLIL.PRE_ARCHIVE_CUTOFF_MESSAGE in caplog.text

    @pytest.mark.network
    def test_download_and_process_multiple_cme_runs_real_network(self, bkg_instance, cme_instance):
        target_date = datetime(2024, 5, 8, tzinfo=timezone.utc)

        bkg_instance.download_and_process(target_date)
        cme_instance.download_and_process(target_date)

        bkg_files = sorted((DATA_DIR / "2024/05").glob("ENLIL_FORECAST_bkg_*.csv"))
        cme_files = sorted((DATA_DIR / "2024/05").glob("ENLIL_FORECAST_cme_*.csv"))

        assert len(bkg_files) == 1
        assert len(cme_files) == 3

        sample_df = pd.read_csv(bkg_files[0])
        assert all(col in sample_df.columns for col in EXPECTED_COLUMNS)
        assert len(sample_df) > 0

    def test_read_invalid_time_range(self, bkg_instance):
        with pytest.raises(ValueError, match="start_time must be before end_time"):
            bkg_instance.read(datetime(2024, 5, 5, tzinfo=timezone.utc), datetime(2024, 5, 1, tzinfo=timezone.utc))

    def test_read_bkg_no_data_warns_and_returns_empty_dataframe(self, bkg_instance):
        with pytest.warns(UserWarning, match="No bkg file found"):
            df = bkg_instance.read(datetime(2099, 1, 1, tzinfo=timezone.utc), download=False)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert all(col in df.columns for col in EXPECTED_COLUMNS)

    def test_read_cme_no_data_returns_empty_list(self, cme_instance):
        result = cme_instance.read(datetime(2099, 1, 1, tzinfo=timezone.utc), download=False)

        assert result == []

    def test_read_bkg_returns_single_dataframe(self, bkg_instance):
        target_date = datetime(2024, 5, 3, tzinfo=timezone.utc)
        t = pd.date_range(target_date, periods=5, freq="h", tz="UTC")
        sample = pd.DataFrame(
            {
                "bx_gsm": np.full(len(t), 1.0),
                "by_gsm": np.full(len(t), 2.0),
                "bz_gsm": np.full(len(t), 3.0),
                "bavg": np.full(len(t), 3.74),
                "speed": np.full(len(t), 400.0),
                "proton_density": np.full(len(t), 5.0),
                "temperature": np.full(len(t), 1e5),
                "pdyn": np.full(len(t), 1.6),
                "file_name": "some/path.csv",
            },
            index=t,
        )
        file_path = DATA_DIR / "2024/05" / "ENLIL_FORECAST_bkg_20240503_0000.csv"
        file_path.parent.mkdir(parents=True)
        sample.to_csv(file_path)

        df = bkg_instance.read(target_date)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert df["bx_gsm"].iloc[0] == pytest.approx(1.0)

    def test_read_cme_returns_list_of_dataframes(self, cme_instance):
        target_date = datetime(2024, 5, 3, tzinfo=timezone.utc)
        directory = DATA_DIR / "2024/05"
        directory.mkdir(parents=True)

        t = pd.date_range(target_date, periods=3, freq="h", tz="UTC")
        for run_time in ["0613", "1414"]:
            sample = pd.DataFrame(
                {col: np.full(len(t), 1.0) for col in EXPECTED_COLUMNS} | {"file_name": "x"},
                index=t,
            )
            sample.to_csv(directory / f"ENLIL_FORECAST_cme_20240503_{run_time}.csv")

        result = cme_instance.read(target_date)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(df, pd.DataFrame) for df in result)

    def test_read_truncates_to_time_range(self, bkg_instance):
        target_date = datetime(2024, 5, 3, tzinfo=timezone.utc)
        t = pd.date_range(target_date, periods=10, freq="h", tz="UTC")
        sample = pd.DataFrame(
            {col: np.arange(len(t), dtype=float) for col in EXPECTED_COLUMNS} | {"file_name": "x"},
            index=t,
        )
        file_path = DATA_DIR / "2024/05" / "ENLIL_FORECAST_bkg_20240503_0000.csv"
        file_path.parent.mkdir(parents=True)
        sample.to_csv(file_path)

        end_time = target_date + pd.Timedelta(hours=3)
        df = bkg_instance.read(target_date, end_time)

        assert df.index.max() <= end_time
        assert len(df) == 4

    def test_read_only_touches_start_time_date(self, bkg_instance):
        # Two files exist (03 and 04); reading with end_time on the 04th must
        # still only read the file keyed at start_time's date (the 03rd).
        directory = DATA_DIR / "2024/05"
        directory.mkdir(parents=True)
        for day, value in [(3, 1.0), (4, 2.0)]:
            t = pd.date_range(datetime(2024, 5, day, tzinfo=timezone.utc), periods=3, freq="h", tz="UTC")
            sample = pd.DataFrame(
                {col: np.full(len(t), value) for col in EXPECTED_COLUMNS} | {"file_name": "x"},
                index=t,
            )
            sample.to_csv(directory / f"ENLIL_FORECAST_bkg_202405{day:02d}_0000.csv")

        df = bkg_instance.read(datetime(2024, 5, 3, tzinfo=timezone.utc), datetime(2024, 5, 4, 23, tzinfo=timezone.utc))

        assert (df["bx_gsm"] == 1.0).all()

    def test_read_download_triggers_download_and_process(self, bkg_instance, tmp_path):
        target_date = datetime(2024, 5, 3, tzinfo=timezone.utc)
        archive_path = tmp_path / "bkg.tar.gz"
        _write_sample_archive(archive_path, "wsa_enlil.mrid1.suball.nc", target_date)
        entries = [self._bkg_entry("20240503")]

        def fake_get(url, *args, **kwargs):
            if url == SWENLIL.API_URL:
                return self._mock_files_response(entries)
            response = Mock()
            response.raise_for_status = Mock()
            response.iter_content = Mock(return_value=[archive_path.read_bytes()])
            return response

        with patch("requests.get", side_effect=fake_get):
            df = bkg_instance.read(target_date, download=True)

        assert (DATA_DIR / "2024/05" / "ENLIL_FORECAST_bkg_20240503_0000.csv").exists()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_read_download_not_triggered_when_file_exists(self, bkg_instance):
        target_date = datetime(2024, 5, 3, tzinfo=timezone.utc)
        t = pd.date_range(target_date, periods=2, freq="h", tz="UTC")
        sample = pd.DataFrame({col: np.full(len(t), 1.0) for col in EXPECTED_COLUMNS} | {"file_name": "x"}, index=t)
        file_path = DATA_DIR / "2024/05" / "ENLIL_FORECAST_bkg_20240503_0000.csv"
        file_path.parent.mkdir(parents=True)
        sample.to_csv(file_path)

        with patch("requests.get") as mock_get:
            bkg_instance.read(target_date, download=True)

        mock_get.assert_not_called()

    def test_read_download_swallows_download_failure(self, bkg_instance):
        target_date = datetime(2099, 1, 1, tzinfo=timezone.utc)

        with patch("requests.get", return_value=self._mock_files_response([])):
            with pytest.warns(UserWarning, match="No bkg file found"):
                df = bkg_instance.read(target_date, download=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_read_warns_before_cutoff(self, bkg_instance, caplog):
        with caplog.at_level("CRITICAL"):
            bkg_instance.read(datetime(2022, 1, 1, tzinfo=timezone.utc), download=False)

        assert SWENLIL.PRE_ARCHIVE_CUTOFF_MESSAGE in caplog.text
