# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Dmitrii Gurev
# SPDX-FileContributor: Simon Mischel
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from swvo.io.sme import SMESuperMAG

TEST_USERNAME = "test:user"
UTC = timezone.utc


def _records(*, include_fill_values: bool = False) -> list[dict[str, float]]:
    records = [
        {"tval": 1577836800.0, "SME": 30.0, "SML": -18.0, "SMU": 12.0},
        {"tval": 1577836860.0, "SME": 35.0, "SML": -20.0, "SMU": 15.0},
        {"tval": 1577836920.0, "SME": 40.0, "SML": -25.0, "SMU": 15.0},
    ]
    if include_fill_values:
        records[1].update({"SME": 999999.0, "SML": 999998.0, "SMU": 1_000_000.0})
    return records


def _response(text: str | None = None) -> Mock:
    response = Mock()
    response.text = text if text is not None else json.dumps(_records())
    response.raise_for_status.return_value = None
    return response


def _write_cache(path: Path, columns: tuple[str, ...] = ("sme", "sml", "smu")) -> None:
    data = pd.DataFrame(_records()).rename(columns={"SME": "sme", "SML": "sml", "SMU": "smu"})
    data["timestamp"] = pd.to_datetime(data.pop("tval"), unit="s", utc=True)
    data = data.set_index("timestamp").loc[:, list(columns)]
    data.to_csv(path)


@pytest.fixture
def reader(tmp_path: Path) -> SMESuperMAG:
    return SMESuperMAG(TEST_USERNAME, data_dir=tmp_path)


class TestInitializationAndFileNames:
    def test_initialization_with_data_dir(self, tmp_path: Path):
        reader = SMESuperMAG(TEST_USERNAME, data_dir=tmp_path)
        assert reader.data_dir == tmp_path
        assert reader.username == TEST_USERNAME

    def test_initialization_with_env_var(self, tmp_path: Path):
        with patch.dict(os.environ, {SMESuperMAG.ENV_VAR_NAME: str(tmp_path)}):
            reader = SMESuperMAG(TEST_USERNAME)
        assert reader.data_dir == tmp_path

    def test_data_dir_takes_precedence_by_default(self, tmp_path: Path):
        explicit = tmp_path / "explicit"
        from_environment = tmp_path / "environment"
        with patch.dict(os.environ, {SMESuperMAG.ENV_VAR_NAME: str(from_environment)}):
            reader = SMESuperMAG(TEST_USERNAME, data_dir=explicit)
        assert reader.data_dir == explicit

    def test_prefer_env_var(self, tmp_path: Path):
        explicit = tmp_path / "explicit"
        from_environment = tmp_path / "environment"
        with patch.dict(os.environ, {SMESuperMAG.ENV_VAR_NAME: str(from_environment)}):
            reader = SMESuperMAG(TEST_USERNAME, data_dir=explicit, prefer_env_var=True)
        assert reader.data_dir == from_environment

    def test_initialization_without_data_location(self):
        with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError, match="SUPERMAG_STREAM_DIR"):
            SMESuperMAG(TEST_USERNAME)

    def test_get_processed_file_list_preserves_daily_names(self, reader: SMESuperMAG):
        paths, intervals = reader._get_processed_file_list(datetime(2020, 1, 1), datetime(2020, 1, 2))
        assert [path.name for path in paths] == ["SuperMAG_SME_20200101.csv", "SuperMAG_SME_20200102.csv"]
        assert intervals == [datetime(2020, 1, 1), datetime(2020, 1, 2)]

    def test_url_reflects_encoded_request_after_download(self, reader: SMESuperMAG):
        assert reader.url == SMESuperMAG.URL

        with patch("requests.get", return_value=_response()):
            reader.download_and_process(datetime(2020, 1, 1), datetime(2020, 1, 2))

        assert isinstance(reader.url, list)
        assert len(reader.url) == 2
        assert all(url.startswith(f"{SMESuperMAG.URL}?") for url in reader.url)
        assert all("logon=test%3Auser" in url for url in reader.url)
        assert all("indices=sme%2Csml%2Csmu" in url for url in reader.url)


class TestVariableSelection:
    @pytest.mark.parametrize(
        ("variables", "expected"),
        [
            (None, ["sme"]),
            ("all", ["sme", "sml", "smu"]),
            ("sml", ["sml"]),
            (["smu", "sme"], ["smu", "sme"]),
            (("sml", "sml", "sme"), ["sml", "sme"]),
            ((item for item in ["smu", "sml"]), ["smu", "sml"]),
        ],
    )
    def test_valid_selections(self, reader: SMESuperMAG, variables, expected: list[str]):
        assert reader._resolve_variables(variables) == expected

    @pytest.mark.parametrize("variables", [[], (), iter(())])
    def test_empty_selection(self, reader: SMESuperMAG, variables):
        with pytest.raises(ValueError, match="at least one"):
            reader._resolve_variables(variables)

    @pytest.mark.parametrize("variables", [1, 1.5, object()])
    def test_invalid_container_type(self, reader: SMESuperMAG, variables):
        with pytest.raises(TypeError, match="string, an iterable"):
            reader._resolve_variables(variables)

    @pytest.mark.parametrize("variables", [["sme", 1], [None], {"sme": 1}.values()])
    def test_non_string_entry(self, reader: SMESuperMAG, variables):
        with pytest.raises(TypeError, match="every variables entry"):
            reader._resolve_variables(variables)

    def test_unknown_names_report_available_names(self, reader: SMESuperMAG):
        with pytest.raises(ValueError, match=r"Unknown SuperMAG variables: smx.*sme, sml, smu"):
            reader._resolve_variables(["smx", "smx"])


class TestResponseProcessing:
    def test_complete_response_schema_and_utc_index(self, reader: SMESuperMAG):
        result = reader._process_response_text(json.dumps(_records()))
        assert list(result.columns) == ["sme", "sml", "smu"]
        assert isinstance(result.index, pd.DatetimeIndex)
        assert str(result.index.tz) == "UTC"
        assert result.index.name == "timestamp"
        expected_index = pd.date_range("2020-01-01T00:00:00Z", periods=3, freq="min", name="timestamp")
        assert result.index.tolist() == expected_index.tolist()
        np.testing.assert_allclose(result["sme"], result["smu"] - result["sml"])

    def test_fill_values_are_masked_for_every_variable(self, reader: SMESuperMAG):
        result = reader._process_response_text(json.dumps(_records(include_fill_values=True)))
        assert result.iloc[1].isna().all()
        assert result.iloc[0].notna().all()

    def test_non_numeric_measurement_is_masked(self, reader: SMESuperMAG):
        records = _records()
        records[0]["SML"] = "missing"  # type: ignore[assignment]
        result = reader._process_response_text(json.dumps(records))
        assert pd.isna(result.iloc[0]["sml"])

    def test_single_json_object_is_supported(self, reader: SMESuperMAG):
        result = reader._process_response_text(json.dumps(_records()[0]))
        assert len(result) == 1

    def test_wrapped_json_is_supported(self, reader: SMESuperMAG):
        result = reader._process_response_text(f"response follows:\n{json.dumps(_records())}\nend")
        assert len(result) == 3

    @pytest.mark.parametrize(
        ("response_text", "message"),
        [
            ("", "empty response"),
            ("ERROR Invalid user", "SuperMAG ERROR Invalid user"),
            ("not json", "No JSON object or array"),
            ("prefix [{broken}] suffix", "Malformed JSON"),
            ("[]", "at least one record"),
            ('"unexpected"', "at least one record"),
        ],
    )
    def test_invalid_responses(self, reader: SMESuperMAG, response_text: str, message: str):
        with pytest.raises(ValueError, match=message):
            reader._process_response_text(response_text)

    @pytest.mark.parametrize("missing_field", ["tval", "SME", "SML", "SMU"])
    def test_missing_response_fields(self, reader: SMESuperMAG, missing_field: str):
        records = _records()
        for record in records:
            del record[missing_field]
        with pytest.raises(ValueError, match=missing_field):
            reader._process_response_text(json.dumps(records))

    def test_invalid_timestamp(self, reader: SMESuperMAG):
        records = _records()
        records[0]["tval"] = "not-a-time"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="invalid timestamp"):
            reader._process_response_text(json.dumps(records))

    def test_file_processor_uses_same_parser(self, reader: SMESuperMAG, tmp_path: Path):
        response_path = tmp_path / "response.txt"
        response_path.write_text(json.dumps(_records()))
        result = reader._process_single_file(response_path)
        assert list(result.columns) == ["sme", "sml", "smu"]


class TestDownloadAndAtomicCache:
    def test_exact_encoded_request_parameters_and_complete_cache(
        self, reader: SMESuperMAG, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.DEBUG)
        with patch("swvo.io.sme.supermag.requests.get", return_value=_response()) as get:
            reader.download_and_process(datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 2))

        get.assert_called_once_with(
            "https://supermag.jhuapl.edu/services/indices.php",
            params={
                "python": "",
                "nohead": "",
                "logon": TEST_USERNAME,
                "start": "2020-01-01T00:00",
                "extent": 86400,
                "indices": "sme,sml,smu",
            },
            timeout=10,
        )
        cache = pd.read_csv(tmp_path / "SuperMAG_SME_20200101.csv")
        assert list(cache.columns) == ["timestamp", "sme", "sml", "smu"]
        assert TEST_USERNAME not in caplog.text
        assert not list(tmp_path.glob("*.tmp"))

    def test_complete_cache_is_not_downloaded(self, reader: SMESuperMAG, tmp_path: Path):
        _write_cache(tmp_path / "SuperMAG_SME_20200101.csv")
        with patch("swvo.io.sme.supermag.requests.get") as get:
            reader.download_and_process(datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 2))
        get.assert_not_called()

    def test_reprocess_replaces_complete_cache(self, reader: SMESuperMAG, tmp_path: Path):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        _write_cache(cache_path)
        with patch("swvo.io.sme.supermag.requests.get", return_value=_response()) as get:
            reader.download_and_process(datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 2), reprocess_files=True)
        get.assert_called_once()
        assert list(pd.read_csv(cache_path).columns) == ["timestamp", "sme", "sml", "smu"]

    def test_legacy_cache_is_upgraded(self, reader: SMESuperMAG, tmp_path: Path):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        _write_cache(cache_path, ("sme",))
        with patch("swvo.io.sme.supermag.requests.get", return_value=_response()) as get:
            reader.download_and_process(datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 2))
        get.assert_called_once()
        assert list(pd.read_csv(cache_path).columns) == ["timestamp", "sme", "sml", "smu"]

    @pytest.mark.parametrize(
        "side_effect",
        [requests.Timeout("timed out"), requests.HTTPError("server failed")],
    )
    def test_request_failure_preserves_existing_cache_and_cleans_temporary_file(
        self, reader: SMESuperMAG, tmp_path: Path, side_effect: requests.RequestException
    ):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        original = "timestamp,sme\n2020-01-01T00:00:00+00:00,7\n"
        cache_path.write_text(original)
        with (
            patch("swvo.io.sme.supermag.requests.get", side_effect=side_effect),
            pytest.raises(type(side_effect), match=str(side_effect)),
        ):
            reader.download_and_process(datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 2), reprocess_files=True)
        assert cache_path.read_text() == original
        assert not cache_path.with_suffix(".csv.tmp").exists()

    def test_http_status_failure_preserves_existing_cache(self, reader: SMESuperMAG, tmp_path: Path):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        cache_path.write_text("timestamp,sme\n2020-01-01T00:00:00+00:00,7\n")
        response = _response()
        response.raise_for_status.side_effect = requests.HTTPError("503")
        with (
            patch("swvo.io.sme.supermag.requests.get", return_value=response),
            pytest.raises(requests.HTTPError, match="503"),
        ):
            reader.download_and_process(datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 2), reprocess_files=True)
        assert "7" in cache_path.read_text()
        assert not cache_path.with_suffix(".csv.tmp").exists()

    def test_parse_failure_preserves_existing_cache(self, reader: SMESuperMAG, tmp_path: Path):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        cache_path.write_text("timestamp,sme\n2020-01-01T00:00:00+00:00,7\n")
        with (
            patch("swvo.io.sme.supermag.requests.get", return_value=_response("malformed")),
            pytest.raises(ValueError, match="No JSON"),
        ):
            reader.download_and_process(datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 2), reprocess_files=True)
        assert "7" in cache_path.read_text()
        assert not cache_path.with_suffix(".csv.tmp").exists()

    def test_write_failure_preserves_existing_cache(self, reader: SMESuperMAG, tmp_path: Path):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        cache_path.write_text("timestamp,sme\n2020-01-01T00:00:00+00:00,7\n")
        with (
            patch("swvo.io.sme.supermag.requests.get", return_value=_response()),
            patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            reader.download_and_process(datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 2), reprocess_files=True)
        assert "7" in cache_path.read_text()
        assert not cache_path.with_suffix(".csv.tmp").exists()

    def test_download_rejects_equal_or_reversed_ranges(self, reader: SMESuperMAG):
        with pytest.raises(ValueError, match="before"):
            reader.download_and_process(datetime(2020, 1, 1), datetime(2020, 1, 1))
        with pytest.raises(ValueError, match="before"):
            reader.download_and_process(datetime(2020, 1, 2), datetime(2020, 1, 1))


class TestReadAndCacheCompatibility:
    def test_legacy_default_schema_and_values(self, reader: SMESuperMAG, tmp_path: Path):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        _write_cache(cache_path, ("sme",))
        result = reader.read(datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 2))
        assert list(result.columns) == ["sme", "file_name"]
        assert len(result) == 3
        assert result["sme"].tolist() == [30.0, 35.0, 40.0]
        assert result["file_name"].eq(cache_path).all()

    @pytest.mark.parametrize(
        ("variables", "columns"),
        [
            ("all", ["sme", "sml", "smu", "file_name"]),
            ("sml", ["sml", "file_name"]),
            (["smu", "sme"], ["smu", "sme", "file_name"]),
            (["sml", "sml", "sme"], ["sml", "sme", "file_name"]),
        ],
    )
    def test_requested_output_order(self, reader: SMESuperMAG, tmp_path: Path, variables, columns: list[str]):
        _write_cache(tmp_path / "SuperMAG_SME_20200101.csv")
        result = reader.read(datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 2), variables=variables)
        assert list(result.columns) == columns

    def test_utc_conversion_and_interval_boundaries(self, reader: SMESuperMAG, tmp_path: Path):
        _write_cache(tmp_path / "SuperMAG_SME_20200101.csv")
        plus_one = timezone(timedelta(hours=1))
        result = reader.read(
            datetime(2020, 1, 1, 1, 1, tzinfo=plus_one),
            datetime(2020, 1, 1, 1, 2, tzinfo=plus_one),
        )
        assert result.index.tolist() == [
            pd.Timestamp("2020-01-01T00:01:00Z"),
            pd.Timestamp("2020-01-01T00:02:00Z"),
        ]
        assert str(result.index.tz) == "UTC"

    def test_missing_files_warn_and_return_default_nan_schema(self, reader: SMESuperMAG):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = reader.read(datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 2))
        assert list(result.columns) == ["sme", "file_name"]
        assert len(result) == 3
        assert result.isna().all().all()
        assert "SuperMAG_SME_20200101.csv not found" in str(caught[-1].message)

    def test_missing_cache_is_downloaded_on_demand(self, reader: SMESuperMAG):
        with patch("swvo.io.sme.supermag.requests.get", return_value=_response()) as get:
            result = reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1, 0, 2),
                download=True,
                variables="all",
            )
        get.assert_called_once()
        assert result[["sme", "sml", "smu"]].notna().all().all()

    def test_legacy_cache_reports_missing_fields_without_download(self, reader: SMESuperMAG, tmp_path: Path):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        _write_cache(cache_path, ("sme",))
        with pytest.raises(ValueError, match=r"sml, smu.*download=True.*upgrade"):
            reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1, 0, 2),
                variables=["sml", "smu"],
            )
        assert list(pd.read_csv(cache_path).columns) == ["timestamp", "sme"]

    def test_legacy_cache_is_upgraded_for_expanded_read(self, reader: SMESuperMAG, tmp_path: Path):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        _write_cache(cache_path, ("sme",))
        with patch("swvo.io.sme.supermag.requests.get", return_value=_response()) as get:
            result = reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1, 0, 2),
                download=True,
                variables=["smu", "sml"],
            )
        get.assert_called_once()
        assert list(result.columns) == ["smu", "sml", "file_name"]
        assert list(pd.read_csv(cache_path).columns) == ["timestamp", "sme", "sml", "smu"]

    def test_partially_expanded_cache_still_requires_complete_schema(self, reader: SMESuperMAG, tmp_path: Path):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        _write_cache(cache_path, ("sme", "sml"))
        with pytest.raises(ValueError, match=r"smu.*download=True.*upgrade"):
            reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1, 0, 2),
                variables="sml",
            )

    def test_partially_expanded_cache_is_upgraded_when_download_enabled(self, reader: SMESuperMAG, tmp_path: Path):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        _write_cache(cache_path, ("sme", "sml"))
        with patch("swvo.io.sme.supermag.requests.get", return_value=_response()) as get:
            result = reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1, 0, 2),
                download=True,
                variables="sml",
            )
        get.assert_called_once()
        assert list(result.columns) == ["sml", "file_name"]
        assert list(pd.read_csv(cache_path).columns) == ["timestamp", "sme", "sml", "smu"]

    def test_complete_cache_never_requires_credentials_during_read(self, reader: SMESuperMAG, tmp_path: Path):
        _write_cache(tmp_path / "SuperMAG_SME_20200101.csv")
        with patch("swvo.io.sme.supermag.requests.get") as get:
            result = reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1, 0, 2),
                download=True,
                variables="all",
            )
        get.assert_not_called()
        assert result[["sme", "sml", "smu"]].notna().all().all()

    def test_corrupt_cache_reports_remediation(self, reader: SMESuperMAG, tmp_path: Path):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        cache_path.write_text("not,a,valid,cache\n")
        with pytest.raises(ValueError, match=r"Cannot read.*download=True.*corrupt"):
            reader.read(datetime(2020, 1, 1), datetime(2020, 1, 1, 0, 2))

    def test_corrupt_cache_is_replaced_when_download_enabled(self, reader: SMESuperMAG, tmp_path: Path):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        cache_path.write_text("not,a,valid,cache\n")
        with patch("swvo.io.sme.supermag.requests.get", return_value=_response()):
            result = reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1, 0, 2),
                download=True,
                variables="all",
            )
        assert result[["sme", "sml", "smu"]].notna().all().all()
        assert list(pd.read_csv(cache_path).columns) == ["timestamp", "sme", "sml", "smu"]

    @pytest.mark.parametrize(
        "contents",
        [
            "timestamp,sme,sml,smu\n",
            "timestamp,sme,sml,smu\n2020-01-01T00:00:00Z,invalid,-5,10\n",
            "timestamp,sme,sml,smu\nnot-a-time,15,-5,10\n",
        ],
    )
    def test_complete_headers_do_not_hide_corrupt_cache(self, reader: SMESuperMAG, tmp_path: Path, contents: str):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        cache_path.write_text(contents)
        with pytest.raises(ValueError, match=r"Cannot read.*download=True.*corrupt"):
            reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1, 0, 2),
                variables="all",
            )

    def test_fill_values_in_existing_cache_are_masked(self, reader: SMESuperMAG, tmp_path: Path):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        cache_path.write_text("timestamp,sme,sml,smu\n2020-01-01T00:00:00Z,999999,999998,1000000\n")
        result = reader.read(datetime(2020, 1, 1), datetime(2020, 1, 1), variables="all")
        assert result[["sme", "sml", "smu"]].isna().all().all()
        assert pd.isna(result.iloc[0]["file_name"])

    def test_provenance_follows_selected_variables(self, reader: SMESuperMAG, tmp_path: Path):
        cache_path = tmp_path / "SuperMAG_SME_20200101.csv"
        data = pd.DataFrame(
            {
                "sme": [np.nan],
                "sml": [-5.0],
                "smu": [10.0],
            },
            index=pd.DatetimeIndex(["2020-01-01T00:00:00Z"], name="timestamp"),
        )
        data.to_csv(cache_path)
        sme = reader.read(datetime(2020, 1, 1), datetime(2020, 1, 1), variables="sme")
        all_indices = reader.read(datetime(2020, 1, 1), datetime(2020, 1, 1), variables="all")
        assert pd.isna(sme.iloc[0]["file_name"])
        assert all_indices.iloc[0]["file_name"] == cache_path

    def test_read_rejects_reversed_range(self, reader: SMESuperMAG):
        with pytest.raises(ValueError, match="start_time must be before end_time"):
            reader.read(datetime(2020, 1, 2), datetime(2020, 1, 1))


@pytest.mark.skipif(
    os.environ.get("SWVO_RUN_LIVE_TESTS") != "1" or not os.environ.get("SUPERMAG_USERNAME"),
    reason="set SWVO_RUN_LIVE_TESTS=1 and SUPERMAG_USERNAME to run the authenticated smoke test",
)
def test_live_authenticated_historical_smoke(tmp_path: Path):
    """Exercise one historical day against SuperMAG only when explicitly enabled."""
    reader = SMESuperMAG(os.environ["SUPERMAG_USERNAME"], data_dir=tmp_path)
    result = reader.read(
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 1, 0, 2, tzinfo=UTC),
        download=True,
        variables="all",
    )
    assert list(result.columns) == ["sme", "sml", "smu", "file_name"]
    assert result[["sme", "sml", "smu"]].notna().all().all()
    np.testing.assert_allclose(result["sme"], result["smu"] - result["sml"], rtol=0, atol=1e-3)
