# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Simon Mischel
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest
import requests

from swvo.io.substorms import SubstormsSuperMAG, available_catalogs

TEST_USERNAME = "test:user"
UTC = timezone.utc


def _ascii_response(
    records: tuple[str, ...] = (
        "2020 01 01 00 30 23.50 67.25 265.95 61.11",
        "2020 01 01 01 45 00.25 68.00 212.83 64.70",
        "2020 01 02 00 00 01.50 69.50 080.56 65.00",
    ),
    *,
    catalog_label: str = "Newell and Gjerloev [2011]",
) -> str:
    rows = "\n".join(records)
    return (
        "All existing substorm onset identification techniques have limitations.\n"
        f"File contains a list identified by {catalog_label}.\n"
        "Substorm database generated on Tue, 31 Oct 2023 15:29:59 +0000.\n"
        "Downloaded from https://supermag.jhuapl.edu/products/.\n"
        "Data Revision:0006\n"
        "============================================================\n"
        "<year>\t<month>\t<day>\t<hour>\t<min>\t<mlt>\t<mlat>\t<glon>\t<glat>\n"
        f"{rows}\n"
    )


def _response(text: str | None = None) -> Mock:
    response = Mock()
    response.text = _ascii_response() if text is None else text
    response.raise_for_status.return_value = None
    return response


def _http_error_response(status_code: int) -> Mock:
    response = _response()
    response.status_code = status_code
    response.raise_for_status.side_effect = requests.HTTPError(
        f"HTTP {status_code}",
        response=response,
    )
    return response


@pytest.fixture
def reader(tmp_path: Path) -> SubstormsSuperMAG:
    return SubstormsSuperMAG(TEST_USERNAME, data_dir=tmp_path)


class TestInitializationAndDiscovery:
    def test_initialization_with_explicit_data_dir(self, tmp_path: Path):
        reader = SubstormsSuperMAG(TEST_USERNAME, data_dir=tmp_path)
        assert reader.data_dir == tmp_path
        assert reader.username == TEST_USERNAME
        assert reader.url == SubstormsSuperMAG.URL

    def test_initialization_with_environment_directory(self, tmp_path: Path):
        with patch.dict(os.environ, {SubstormsSuperMAG.ENV_VAR_NAME: str(tmp_path)}):
            reader = SubstormsSuperMAG(TEST_USERNAME)
        assert reader.data_dir == tmp_path

    def test_explicit_directory_takes_precedence_by_default(self, tmp_path: Path):
        explicit = tmp_path / "explicit"
        environment = tmp_path / "environment"
        with patch.dict(os.environ, {SubstormsSuperMAG.ENV_VAR_NAME: str(environment)}):
            reader = SubstormsSuperMAG(TEST_USERNAME, data_dir=explicit)
        assert reader.data_dir == explicit

    def test_environment_directory_can_be_preferred(self, tmp_path: Path):
        explicit = tmp_path / "explicit"
        environment = tmp_path / "environment"
        with patch.dict(os.environ, {SubstormsSuperMAG.ENV_VAR_NAME: str(environment)}):
            reader = SubstormsSuperMAG(
                TEST_USERNAME,
                data_dir=explicit,
                prefer_env_var=True,
            )
        assert reader.data_dir == environment

    @pytest.mark.parametrize("username", [None, 1, object()])
    def test_username_must_be_a_string(self, tmp_path: Path, username):
        with pytest.raises(TypeError, match="username must be a string"):
            SubstormsSuperMAG(username, data_dir=tmp_path)

    @pytest.mark.parametrize("username", ["", " ", "\t"])
    def test_username_must_not_be_empty(self, tmp_path: Path, username: str):
        with pytest.raises(ValueError, match="must not be empty"):
            SubstormsSuperMAG(username, data_dir=tmp_path)

    def test_missing_data_directory_configuration_is_reported(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(
                ValueError,
                match="SUPERMAG_STREAM_DIR",
            ),
        ):
            SubstormsSuperMAG(TEST_USERNAME)

    def test_available_catalogs_describes_all_official_lists(self, reader: SubstormsSuperMAG):
        metadata = reader.available_catalogs()
        assert metadata["name"].tolist() == [
            "newell",
            "forsyth",
            "ohtani",
            "frey",
            "liou",
        ]
        assert list(metadata.columns) == [
            "name",
            "label",
            "aliases",
            "method",
            "coverage",
            "continuously_updated",
            "location_description",
            "reference",
        ]
        assert metadata.loc[metadata["name"] == "forsyth", "aliases"].item() == ("sophie",)
        assert metadata.loc[metadata["name"] == "newell", "continuously_updated"].item()
        assert not metadata.loc[metadata["name"] == "frey", "continuously_updated"].item()
        assert metadata["reference"].str.startswith("https://doi.org/").all()

    def test_module_discovery_returns_an_independent_table(self):
        first = available_catalogs()
        first.loc[0, "name"] = "changed"
        second = available_catalogs()
        assert second.loc[0, "name"] == "newell"


class TestCatalogSelectionAndFileNames:
    @pytest.mark.parametrize(
        ("requested", "canonical"),
        [
            ("newell", "newell"),
            ("NEWELL", "newell"),
            ("newell-gjerloev", "newell"),
            ("newell gjerloev", "newell"),
            ("forsyth", "forsyth"),
            ("sophie", "forsyth"),
            (" SOPHIE ", "forsyth"),
            ("ohtani_gjerloev", "ohtani"),
            ("frey_mende", "frey"),
            ("liou", "liou"),
        ],
    )
    def test_catalog_names_and_aliases(self, requested: str, canonical: str):
        assert SubstormsSuperMAG._resolve_catalog(requested).name == canonical

    @pytest.mark.parametrize("catalog", [None, 1, ["newell"]])
    def test_catalog_must_be_a_string(self, catalog):
        with pytest.raises(TypeError, match="catalog must be a string"):
            SubstormsSuperMAG._resolve_catalog(catalog)

    @pytest.mark.parametrize("catalog", ["", " ", "\n"])
    def test_catalog_must_not_be_empty(self, catalog: str):
        with pytest.raises(ValueError, match="must not be empty"):
            SubstormsSuperMAG._resolve_catalog(catalog)

    def test_unknown_catalog_lists_canonical_choices(self):
        with pytest.raises(
            ValueError,
            match=r"Unknown.*'other'.*newell, forsyth, ohtani, frey, liou",
        ):
            SubstormsSuperMAG._resolve_catalog("other")

    def test_annual_cache_paths_are_catalog_specific(self, reader: SubstormsSuperMAG):
        paths, years = reader._get_cache_file_list(
            datetime(2019, 12, 31),
            datetime(2021, 1, 1),
            reader._resolve_catalog("sophie"),
        )
        assert years == [2019, 2020, 2021]
        assert [path.name for path in paths] == [
            "SuperMAG_SUBSTORMS_FORSYTH_2019.txt",
            "SuperMAG_SUBSTORMS_FORSYTH_2020.txt",
            "SuperMAG_SUBSTORMS_FORSYTH_2021.txt",
        ]
        assert all(path.parent == reader.data_dir / "substorms" / "forsyth" for path in paths)


class TestResponseParsing:
    def test_complete_response_schema_values_and_utc_index(self, reader: SubstormsSuperMAG):
        result = reader._process_response_text(
            _ascii_response(),
            reader._resolve_catalog("newell"),
        )
        assert list(result.columns) == ["mlt", "mlat", "glon", "glat"]
        assert result.index.name == "onset"
        assert str(result.index.tz) == "UTC"
        assert result.index.tolist() == [
            pd.Timestamp("2020-01-01T00:30:00Z"),
            pd.Timestamp("2020-01-01T01:45:00Z"),
            pd.Timestamp("2020-01-02T00:00:00Z"),
        ]
        assert result.iloc[0].to_dict() == {
            "mlt": 23.5,
            "mlat": 67.25,
            "glon": 265.95,
            "glat": 61.11,
        }
        assert all(pd.api.types.is_numeric_dtype(result[column]) for column in result.columns)

    def test_whitespace_separated_header_and_records_are_supported(self, reader: SubstormsSuperMAG):
        text = _ascii_response().replace("\t", "    ")
        result = reader._process_response_text(text, reader._resolve_catalog("newell"))
        assert len(result) == 3

    def test_valid_no_event_response_returns_typed_empty_table(self, reader: SubstormsSuperMAG):
        result = reader._process_response_text(
            _ascii_response(records=()),
            reader._resolve_catalog("liou"),
        )
        assert result.empty
        assert list(result.columns) == ["mlt", "mlat", "glon", "glat"]
        assert result.index.name == "onset"
        assert str(result.index.tz) == "UTC"
        assert all(pd.api.types.is_float_dtype(result[column]) for column in result.columns)

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("", "empty response"),
            (" \n\t", "empty response"),
            ("ERROR: Invalid username", "SuperMAG ERROR: Invalid username"),
            ("an HTML proxy page", "expected table header"),
            (
                _ascii_response(records=("2020 01 01 00 30 23.5 67.2",)),
                "expected 9 fields, found 7",
            ),
            (
                _ascii_response(records=("2020 01 01 00 30 bad 67.2 265.9 61.1",)),
                "non-numeric mlt",
            ),
            (
                _ascii_response(records=("2020 13 01 00 30 23.5 67.2 265.9 61.1",)),
                "invalid onset time",
            ),
        ],
    )
    def test_invalid_responses_are_rejected(
        self,
        reader: SubstormsSuperMAG,
        text: str,
        message: str,
    ):
        with pytest.raises(ValueError, match=message):
            reader._process_response_text(text, reader._resolve_catalog("newell"))

    def test_duplicate_onset_times_are_preserved(self, reader: SubstormsSuperMAG):
        record = "2020 01 01 00 30 23.50 67.25 265.95 61.11"
        result = reader._process_response_text(
            _ascii_response(records=(record, record)),
            reader._resolve_catalog("newell"),
        )
        assert len(result) == 2
        assert result.index.duplicated().sum() == 1

    def test_liou_mlt_degrees_are_normalized_to_hours(
        self,
        reader: SubstormsSuperMAG,
    ):
        result = reader._process_response_text(
            _ascii_response(
                records=("2000 01 06 19 36 337.90 63.80 39.90 68.00",),
                catalog_label="Liou [2010]",
            ),
            reader._resolve_catalog("liou"),
        )
        assert result.iloc[0]["mlt"] == pytest.approx(337.9 / 15)


class TestDownloadingAndAtomicCaches:
    def test_exact_request_parameters_and_sanitized_provenance(
        self,
        reader: SubstormsSuperMAG,
        caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level(logging.DEBUG)
        with patch(
            "swvo.io.substorms.supermag.requests.get",
            return_value=_response(),
        ) as get:
            reader.download_and_process(
                datetime(2020, 6, 1),
                datetime(2020, 6, 2),
                catalog="sophie",
            )

        get.assert_called_once_with(
            SubstormsSuperMAG.URL,
            params={
                "service": "substorms",
                "downloadtype": "substorm_list",
                "user": TEST_USERNAME,
                "fmt": "ascii",
                "start": "2020-01-01T00:00:00+00:00",
                "end": "2020-12-31T23:59:59+00:00",
                "list": "forsyth",
            },
            timeout=30,
        )
        assert isinstance(reader.url, str)
        assert "user=%3Credacted%3E" in reader.url
        assert "list=forsyth" in reader.url
        assert TEST_USERNAME not in reader.url
        assert TEST_USERNAME not in caplog.text

    @pytest.mark.parametrize("catalog", ["newell", "forsyth", "ohtani", "frey", "liou"])
    def test_every_catalog_can_be_downloaded(
        self,
        reader: SubstormsSuperMAG,
        catalog: str,
    ):
        with patch(
            "swvo.io.substorms.supermag.requests.get",
            return_value=_response(),
        ) as get:
            reader.download_and_process(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                catalog=catalog,
            )
        assert get.call_args.kwargs["params"]["list"] == catalog
        path = reader.data_dir / "substorms" / catalog / f"SuperMAG_SUBSTORMS_{catalog.upper()}_2020.txt"
        assert path.read_text(encoding="utf-8") == _ascii_response()

    def test_valid_no_event_year_is_cached_without_retry(self, reader: SubstormsSuperMAG):
        text = _ascii_response(records=())
        with (
            patch(
                "swvo.io.substorms.supermag.requests.get",
                return_value=_response(text),
            ) as get,
            patch("swvo.io.substorms.supermag.sleep") as retry_sleep,
        ):
            reader.download_and_process(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                catalog="liou",
            )
        get.assert_called_once()
        retry_sleep.assert_not_called()
        path = reader.data_dir / "substorms" / "liou" / "SuperMAG_SUBSTORMS_LIOU_2020.txt"
        assert path.read_text(encoding="utf-8") == text

    def test_out_of_year_record_is_rejected_before_cache_install(
        self,
        reader: SubstormsSuperMAG,
    ):
        text = _ascii_response(
            records=("2021 01 01 00 00 00.25 68.00 212.83 64.70",),
        )
        with (
            patch(
                "swvo.io.substorms.supermag.requests.get",
                return_value=_response(text),
            ),
            pytest.raises(ValueError, match="outside that year"),
        ):
            reader.download_and_process(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
            )

        path = reader.data_dir / "substorms" / "newell" / "SuperMAG_SUBSTORMS_NEWELL_2020.txt"
        assert not path.exists()

    def test_existing_cache_is_not_replaced_without_reprocess(self, reader: SubstormsSuperMAG):
        catalog = reader._resolve_catalog("newell")
        path = reader._get_cache_file_list(
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            catalog,
        )[0][0]
        path.parent.mkdir(parents=True)
        path.write_text(_ascii_response(records=()), encoding="utf-8")

        with patch("swvo.io.substorms.supermag.requests.get") as get:
            reader.download_and_process(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
            )
        get.assert_not_called()

    def test_reprocess_replaces_existing_cache(self, reader: SubstormsSuperMAG):
        catalog = reader._resolve_catalog("newell")
        path = reader._get_cache_file_list(
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            catalog,
        )[0][0]
        path.parent.mkdir(parents=True)
        path.write_text(_ascii_response(records=()), encoding="utf-8")

        with patch(
            "swvo.io.substorms.supermag.requests.get",
            return_value=_response(),
        ) as get:
            reader.download_and_process(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                reprocess_files=True,
            )
        get.assert_called_once()
        assert path.read_text(encoding="utf-8") == _ascii_response()

    def test_batch_download_replaces_corrupt_cache_without_reprocess(
        self,
        reader: SubstormsSuperMAG,
    ):
        catalog = reader._resolve_catalog("newell")
        path = reader._get_cache_file_list(
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            catalog,
        )[0][0]
        path.parent.mkdir(parents=True)
        path.write_text("corrupt", encoding="utf-8")

        with patch(
            "swvo.io.substorms.supermag.requests.get",
            return_value=_response(),
        ) as get:
            reader.download_and_process(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
            )

        get.assert_called_once()
        assert path.read_text(encoding="utf-8") == _ascii_response()

    def test_zero_byte_response_is_retried_then_succeeds(self, reader: SubstormsSuperMAG):
        with (
            patch(
                "swvo.io.substorms.supermag.requests.get",
                side_effect=[_response(""), _response()],
            ) as get,
            patch("swvo.io.substorms.supermag.sleep") as retry_sleep,
        ):
            reader.download_and_process(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
            )
        assert get.call_count == 2
        retry_sleep.assert_called_once_with(1.0)

    @pytest.mark.parametrize(
        "error",
        [
            requests.Timeout("timed out"),
            requests.ConnectionError("connection failed"),
        ],
    )
    def test_request_failures_are_retried(
        self,
        reader: SubstormsSuperMAG,
        error: requests.RequestException,
    ):
        with (
            patch(
                "swvo.io.substorms.supermag.requests.get",
                side_effect=[error, _response()],
            ) as get,
            patch("swvo.io.substorms.supermag.sleep") as retry_sleep,
        ):
            reader.download_and_process(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
            )
        assert get.call_count == 2
        retry_sleep.assert_called_once_with(1.0)

    @pytest.mark.parametrize("status_code", [429, 500, 503])
    def test_retryable_http_statuses(
        self,
        reader: SubstormsSuperMAG,
        status_code: int,
    ):
        with (
            patch(
                "swvo.io.substorms.supermag.requests.get",
                side_effect=[_http_error_response(status_code), _response()],
            ) as get,
            patch("swvo.io.substorms.supermag.sleep") as retry_sleep,
        ):
            reader.download_and_process(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
            )
        assert get.call_count == 2
        retry_sleep.assert_called_once_with(1.0)

    def test_exhausted_transient_year_warns_and_batch_continues(
        self,
        reader: SubstormsSuperMAG,
    ):
        responses = [_response("") for _ in range(reader._MAX_DOWNLOAD_ATTEMPTS)]
        responses.append(
            _response(
                _ascii_response(
                    records=("2021 01 01 00 30 23.50 67.25 265.95 61.11",),
                )
            )
        )
        with (
            patch(
                "swvo.io.substorms.supermag.requests.get",
                side_effect=responses,
            ) as get,
            patch("swvo.io.substorms.supermag.sleep") as retry_sleep,
            pytest.warns(
                RuntimeWarning,
                match=r"newell.*4 attempts.*1 year.*2020.*Re-run",
            ),
        ):
            reader.download_and_process(
                datetime(2020, 1, 1),
                datetime(2021, 1, 2),
            )

        assert get.call_count == reader._MAX_DOWNLOAD_ATTEMPTS + 1
        assert retry_sleep.call_args_list == [call(1.0), call(2.0), call(4.0)]
        assert not (reader.data_dir / "substorms" / "newell" / "SuperMAG_SUBSTORMS_NEWELL_2020.txt").exists()
        assert (reader.data_dir / "substorms" / "newell" / "SuperMAG_SUBSTORMS_NEWELL_2021.txt").exists()

    def test_permanent_supermag_error_is_not_retried(self, reader: SubstormsSuperMAG):
        with (
            patch(
                "swvo.io.substorms.supermag.requests.get",
                return_value=_response("ERROR: Invalid username"),
            ) as get,
            patch("swvo.io.substorms.supermag.sleep") as retry_sleep,
            pytest.raises(ValueError, match="Invalid username"),
        ):
            reader.download_and_process(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
            )
        get.assert_called_once()
        retry_sleep.assert_not_called()

    def test_non_retryable_http_error_is_not_retried(self, reader: SubstormsSuperMAG):
        with (
            patch(
                "swvo.io.substorms.supermag.requests.get",
                return_value=_http_error_response(404),
            ) as get,
            patch("swvo.io.substorms.supermag.sleep") as retry_sleep,
            pytest.raises(requests.HTTPError, match="404"),
        ):
            reader.download_and_process(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
            )
        get.assert_called_once()
        retry_sleep.assert_not_called()

    @pytest.mark.parametrize(
        "side_effect",
        [
            requests.Timeout("timed out"),
            requests.ConnectionError("connection failed"),
        ],
    )
    def test_failed_reprocess_preserves_cache_and_removes_temporary_file(
        self,
        reader: SubstormsSuperMAG,
        side_effect: requests.RequestException,
    ):
        catalog = reader._resolve_catalog("newell")
        path = reader._get_cache_file_list(
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            catalog,
        )[0][0]
        path.parent.mkdir(parents=True)
        original = _ascii_response(records=())
        path.write_text(original, encoding="utf-8")

        with (
            patch(
                "swvo.io.substorms.supermag.requests.get",
                side_effect=side_effect,
            ),
            patch("swvo.io.substorms.supermag.sleep"),
            pytest.raises(type(side_effect), match=str(side_effect)),
        ):
            reader._download_and_process_single_file(path, 2020, catalog)

        assert path.read_text(encoding="utf-8") == original
        assert not path.with_suffix(".txt.tmp").exists()

    def test_parse_failure_preserves_existing_cache(self, reader: SubstormsSuperMAG):
        catalog = reader._resolve_catalog("newell")
        path = reader._get_cache_file_list(
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            catalog,
        )[0][0]
        path.parent.mkdir(parents=True)
        original = _ascii_response(records=())
        path.write_text(original, encoding="utf-8")

        with (
            patch(
                "swvo.io.substorms.supermag.requests.get",
                return_value=_response("not a catalogue"),
            ),
            pytest.raises(ValueError, match="expected table header"),
        ):
            reader._download_and_process_single_file(path, 2020, catalog)

        assert path.read_text(encoding="utf-8") == original
        assert not path.with_suffix(".txt.tmp").exists()

    def test_write_failure_preserves_existing_cache(self, reader: SubstormsSuperMAG):
        catalog = reader._resolve_catalog("newell")
        path = reader._get_cache_file_list(
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            catalog,
        )[0][0]
        path.parent.mkdir(parents=True)
        original = _ascii_response(records=())
        path.write_text(original, encoding="utf-8")

        with (
            patch(
                "swvo.io.substorms.supermag.requests.get",
                return_value=_response(),
            ),
            patch("pathlib.Path.write_text", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            reader._download_and_process_single_file(path, 2020, catalog)

        assert path.read_text(encoding="utf-8") == original
        assert not path.with_suffix(".txt.tmp").exists()

    def test_download_rejects_equal_and_reversed_ranges(self, reader: SubstormsSuperMAG):
        with pytest.raises(ValueError, match="before"):
            reader.download_and_process(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1),
            )
        with pytest.raises(ValueError, match="before"):
            reader.download_and_process(
                datetime(2020, 1, 2),
                datetime(2020, 1, 1),
            )


class TestReading:
    @staticmethod
    def _write_cache(
        reader: SubstormsSuperMAG,
        year: int,
        text: str,
        catalog: str = "newell",
    ) -> Path:
        selected = reader._resolve_catalog(catalog)
        path = reader._get_cache_file_list(
            datetime(year, 1, 1),
            datetime(year, 1, 2),
            selected,
        )[0][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_default_read_schema_values_and_provenance(self, reader: SubstormsSuperMAG):
        path = self._write_cache(reader, 2020, _ascii_response())
        result = reader.read(
            datetime(2020, 1, 1),
            datetime(2020, 1, 1, 2),
        )
        assert list(result.columns) == [
            "mlt",
            "mlat",
            "glon",
            "glat",
            "catalog",
            "file_name",
        ]
        assert len(result) == 2
        assert result["catalog"].eq("newell").all()
        assert result["file_name"].eq(path).all()

    def test_alias_read_returns_canonical_catalog_name(self, reader: SubstormsSuperMAG):
        path = self._write_cache(reader, 2020, _ascii_response(), catalog="forsyth")
        result = reader.read(
            datetime(2020, 1, 1),
            datetime(2020, 1, 1, 2),
            catalog="sophie",
        )
        assert result["catalog"].eq("forsyth").all()
        assert result["file_name"].eq(path).all()

    def test_timezone_conversion_and_inclusive_boundaries(self, reader: SubstormsSuperMAG):
        self._write_cache(reader, 2020, _ascii_response())
        plus_one = timezone(timedelta(hours=1))
        result = reader.read(
            datetime(2020, 1, 1, 1, 30, tzinfo=plus_one),
            datetime(2020, 1, 1, 2, 45, tzinfo=plus_one),
        )
        assert result.index.tolist() == [
            pd.Timestamp("2020-01-01T00:30:00Z"),
            pd.Timestamp("2020-01-01T01:45:00Z"),
        ]

    def test_equal_read_boundaries_select_exact_onset(self, reader: SubstormsSuperMAG):
        self._write_cache(reader, 2020, _ascii_response())
        onset = datetime(2020, 1, 1, 0, 30, tzinfo=UTC)
        result = reader.read(onset, onset)
        assert result.index.tolist() == [pd.Timestamp(onset)]

    def test_reverse_read_range_is_rejected(self, reader: SubstormsSuperMAG):
        with pytest.raises(ValueError, match="before or equal"):
            reader.read(
                datetime(2020, 1, 2),
                datetime(2020, 1, 1),
            )

    def test_multiple_years_are_combined_and_sorted(self, reader: SubstormsSuperMAG):
        self._write_cache(
            reader,
            2019,
            _ascii_response(
                records=("2019 12 31 23 59 23.50 67.25 265.95 61.11",),
            ),
        )
        self._write_cache(
            reader,
            2020,
            _ascii_response(
                records=("2020 01 01 00 00 00.25 68.00 212.83 64.70",),
            ),
        )
        result = reader.read(
            datetime(2019, 12, 31, 23, 59),
            datetime(2020, 1, 1, 0, 0),
        )
        assert result.index.tolist() == [
            pd.Timestamp("2019-12-31T23:59:00Z"),
            pd.Timestamp("2020-01-01T00:00:00Z"),
        ]

    def test_missing_cache_warns_and_returns_sparse_empty_schema(self, reader: SubstormsSuperMAG):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
            )
        assert result.empty
        assert list(result.columns) == list(SubstormsSuperMAG._OUTPUT_COLUMNS)
        assert str(result.index.tz) == "UTC"
        assert "download=True" in str(caught[-1].message)

    def test_valid_empty_cache_does_not_warn(self, reader: SubstormsSuperMAG):
        self._write_cache(reader, 2020, _ascii_response(records=()), catalog="liou")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                catalog="liou",
            )
        assert result.empty
        assert not caught

    def test_missing_cache_is_downloaded_on_demand(self, reader: SubstormsSuperMAG):
        with patch(
            "swvo.io.substorms.supermag.requests.get",
            return_value=_response(),
        ) as get:
            result = reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1, 2),
                download=True,
            )
        get.assert_called_once()
        assert len(result) == 2

    def test_complete_cache_does_not_download_again(self, reader: SubstormsSuperMAG):
        self._write_cache(reader, 2020, _ascii_response())
        with patch("swvo.io.substorms.supermag.requests.get") as get:
            result = reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1, 2),
                download=True,
            )
        get.assert_not_called()
        assert len(result) == 2

    def test_corrupt_cache_reports_remediation_without_download(self, reader: SubstormsSuperMAG):
        self._write_cache(reader, 2020, "corrupt")
        with pytest.raises(
            ValueError,
            match=r"Cannot read.*download=True.*corrupt",
        ):
            reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
            )

    def test_corrupt_cache_is_replaced_when_download_is_enabled(self, reader: SubstormsSuperMAG):
        path = self._write_cache(reader, 2020, "corrupt")
        with patch(
            "swvo.io.substorms.supermag.requests.get",
            return_value=_response(),
        ) as get:
            result = reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 1, 2),
                download=True,
            )
        get.assert_called_once()
        assert len(result) == 2
        assert path.read_text(encoding="utf-8") == _ascii_response()

    def test_strict_read_surfaces_exhausted_transient_failure(self, reader: SubstormsSuperMAG):
        with (
            patch(
                "swvo.io.substorms.supermag.requests.get",
                return_value=_response(""),
            ) as get,
            patch("swvo.io.substorms.supermag.sleep") as retry_sleep,
            pytest.raises(ValueError, match="empty response"),
        ):
            reader.read(
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                download=True,
            )
        assert get.call_count == reader._MAX_DOWNLOAD_ATTEMPTS
        assert retry_sleep.call_args_list == [call(1.0), call(2.0), call(4.0)]


@pytest.mark.skipif(
    os.environ.get("SWVO_RUN_LIVE_TESTS") != "1",
    reason="set SWVO_RUN_LIVE_TESTS=1 to run authenticated SuperMAG smoke tests",
)
@pytest.mark.parametrize(
    ("catalog", "year"),
    [
        ("newell", 2001),
        ("forsyth", 2001),
        ("ohtani", 2001),
        ("frey", 2001),
        ("liou", 2000),
    ],
)
def test_live_catalogue_download(
    tmp_path: Path,
    catalog: str,
    year: int,
):
    username = os.environ.get("SUPERMAG_USERNAME")
    if not username:
        pytest.skip("SUPERMAG_USERNAME is not configured")

    reader = SubstormsSuperMAG(username, data_dir=tmp_path)
    events = reader.read(
        datetime(year, 1, 1, tzinfo=UTC),
        datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC),
        download=True,
        catalog=catalog,
    )

    assert not events.empty
    assert list(events.columns) == list(SubstormsSuperMAG._OUTPUT_COLUMNS)
    assert str(events.index.tz) == "UTC"
    assert events["catalog"].eq(catalog).all()
    assert events[["mlt", "mlat", "glon", "glat"]].notna().all().all()
    assert events["mlt"].between(0, 24).all()
    assert events["mlat"].between(-90, 90).all()
    assert events["glat"].between(-90, 90).all()
