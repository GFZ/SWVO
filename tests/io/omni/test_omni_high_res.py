# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Simon Mischel
#
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import requests

from swvo.io.omni.omni_high_res import OMNIHighRes
from swvo.io.omni.variables import HIGH_RES_DEFAULT_VARIABLES, HIGH_RES_VARIABLES

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/OMNI"))


class TestOMNIHighRes:
    @pytest.fixture
    def omni_high_res(self):
        os.environ["OMNI_HIGH_RES_STREAM_DIR"] = str(DATA_DIR)
        yield OMNIHighRes()

    def test_initialization_with_env_var(self, omni_high_res):
        assert omni_high_res.data_dir.exists()

    def test_initialization_with_data_dir(self):
        omni_high_res = OMNIHighRes(data_dir=DATA_DIR)
        assert omni_high_res.data_dir == DATA_DIR

    def test_initialization_without_env_var(self):
        if "OMNI_HIGH_RES_STREAM_DIR" in os.environ:
            del os.environ["OMNI_HIGH_RES_STREAM_DIR"]
        with pytest.raises(ValueError):
            OMNIHighRes()

    def test_download_and_process(self, tmp_path, mocker):
        omni_high_res = OMNIHighRes(data_dir=tmp_path)
        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 1, 31, tzinfo=timezone.utc)
        all_names = [variable.name for variable in HIGH_RES_VARIABLES if 1 in variable.cadences]
        processed = pd.DataFrame(
            {name: [1.0] for name in all_names},
            index=pd.DatetimeIndex(["2020-01-01"], tz="UTC", name="timestamp"),
        )
        mocker.patch.object(omni_high_res, "_get_data_from_omni", return_value=[])
        mocker.patch.object(omni_high_res, "_process_single_month", return_value=processed)

        omni_high_res.download_and_process(start_time, end_time)

        output = tmp_path / "2020/OMNI_HIGH_RES_1min_202001.csv"
        assert output.exists()
        assert all(name in pd.read_csv(output, nrows=0).columns for name in all_names)

    def test_read_without_download(self, omni_high_res, mocker):
        start_time = datetime(2021, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2021, 12, 31, tzinfo=timezone.utc)
        with pytest.raises(
            ValueError
        ):  # value error is raised when no files are found hence no concatenation is possible
            omni_high_res.read(start_time, end_time, download=False)

    def test_read_with_download(self, tmp_path, mocker):
        omni_high_res = OMNIHighRes(data_dir=tmp_path)
        start_time = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2022, 2, 28, tzinfo=timezone.utc)
        index = pd.DatetimeIndex(["2022-01-01"], tz="UTC", name="timestamp")

        def create_month(file_path, time_interval, cadence_min):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = pd.DatetimeIndex([time_interval[0]], name="timestamp")
            pd.DataFrame({name: [1.0] for name in HIGH_RES_DEFAULT_VARIABLES}, index=timestamp).to_csv(file_path)

        download_one = mocker.patch.object(omni_high_res, "_download_and_process_single_file", side_effect=create_month)
        result = omni_high_res.read(start_time, end_time, download=True)

        assert download_one.call_count == 2
        assert list(result.columns) == [*HIGH_RES_DEFAULT_VARIABLES, "file_name"]
        assert index[0] in result.index

    def test_download_and_process_calls_get_data_per_month(self, omni_high_res, mocker):
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2023, 12, 31, tzinfo=timezone.utc)

        dummy_df = pd.DataFrame(
            {
                col: [1.0]
                for col in [
                    "bavg",
                    "bx_gsm",
                    "by_gsm",
                    "bz_gsm",
                    "speed",
                    "proton_density",
                    "temperature",
                    "pdyn",
                    "sym-h",
                ]
            },
            index=pd.DatetimeIndex(["2023-01-01"], tz="UTC", name="timestamp"),
        )

        mocker.patch.object(omni_high_res, "_get_data_from_omni", return_value=[])
        mocker.patch.object(omni_high_res, "_process_single_month", return_value=dummy_df)

        omni_high_res.download_and_process(start_time, end_time)
        assert omni_high_res._get_data_from_omni.call_count == 12

    def test_download_and_process_uses_parallel_for_more_than_10_files(self, tmp_path, mocker):
        omni_high_res = OMNIHighRes(data_dir=tmp_path)
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2023, 12, 31, tzinfo=timezone.utc)
        executor_max_workers = []

        class RecordingExecutor:
            def __init__(self, max_workers=None):
                executor_max_workers.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def submit(self, fn, *args, **kwargs):
                future = Future()
                try:
                    future.set_result(fn(*args, **kwargs))
                except Exception as exc:
                    future.set_exception(exc)
                return future

        mocker.patch("swvo.io.omni.omni_high_res.ThreadPoolExecutor", RecordingExecutor)
        process_single_file = mocker.patch.object(omni_high_res, "_download_and_process_single_file")

        omni_high_res.download_and_process(start_time, end_time)

        assert executor_max_workers == [10]
        assert process_single_file.call_count == 12

    def test_download_and_process_stays_sequential_for_10_files(self, tmp_path, mocker):
        omni_high_res = OMNIHighRes(data_dir=tmp_path)
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2023, 10, 31, tzinfo=timezone.utc)

        executor = mocker.patch("swvo.io.omni.omni_high_res.ThreadPoolExecutor")
        process_single_file = mocker.patch.object(omni_high_res, "_download_and_process_single_file")

        omni_high_res.download_and_process(start_time, end_time)

        executor.assert_not_called()
        assert process_single_file.call_count == 10

    def test_invalid_cadence(self, omni_high_res):
        start_time = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2022, 12, 31, tzinfo=timezone.utc)

        with pytest.raises(AssertionError):
            omni_high_res.read(start_time, end_time, cadence_min=2)

        with pytest.raises(AssertionError):
            omni_high_res.download_and_process(start_time, end_time, cadence_min=10)

    def test_start_year_behind(self, omni_high_res, mocker):
        start_time = datetime(1920, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

        mocker.patch.object(omni_high_res, "_get_processed_file_list", return_value=([], []))
        mocker.patch.object(omni_high_res, "_read_single_file", return_value=pd.DataFrame())

        mocker.patch("pandas.concat", return_value=pd.DataFrame())

        mocker.patch.object(pd.DataFrame, "truncate", return_value=pd.DataFrame())

        with patch("logging.Logger.warning") as mock_warning:
            with pytest.raises(ValueError, match="No OMNI"):
                omni_high_res.read(start_time, end_time)
            mock_warning.assert_any_call(
                "Start date chosen falls behind the existing data. Moving start date to first available mission files..."
            )

    def test_year_transition(self, tmp_path, mocker):
        omni_high_res = OMNIHighRes(data_dir=tmp_path)
        start_time = datetime(2012, 12, 31, 23, 59, 0, tzinfo=timezone.utc)
        end_time = datetime(2012, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        def create_month(file_path, time_interval, cadence_min):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = (
                datetime(2012, 12, 31, 23, 59, tzinfo=timezone.utc)
                if time_interval[0].month == 12
                else datetime(2013, 1, 1, tzinfo=timezone.utc)
            )
            index = pd.DatetimeIndex([timestamp], name="timestamp")
            pd.DataFrame({name: 1.0 for name in HIGH_RES_DEFAULT_VARIABLES}, index=index).to_csv(file_path)

        mocker.patch.object(omni_high_res, "_download_and_process_single_file", side_effect=create_month)

        result_df = omni_high_res.read(start_time, end_time, download=True)

        assert result_df.index.min() == pd.Timestamp("2012-12-31 23:59:00+00:00")
        assert result_df.index.max() == pd.Timestamp("2013-01-01 00:00:00+00:00")

    def test_process_single_month_parses_data_correctly(self, omni_high_res):
        data = [
            "19 Vx Velocity,km/s",
            "20 Vy Velocity, km/s",
            "YYYY DOY HR MN bavg bx_gsm by_gsm bz_gsm speed proton_density temperature pdyn sym-h",
            "2020 1 0 0 5.1 1.2 2.3 3.4 400 5.5 1000000 99 -15",
            "2020 1 0 1 9999.99 9999.99 9999.99 9999.99 99999.9 999.99 9999999.0 99.99 99999.0",
        ]

        df = omni_high_res._process_single_month(data)
        assert isinstance(df.index[0], pd.Timestamp)
        assert len(df) >= 2
        expected_cols = [
            "bavg",
            "bx_gsm",
            "by_gsm",
            "bz_gsm",
            "speed",
            "proton_density",
            "temperature",
            "pdyn",
            "sym-h",
        ]
        assert list(df.columns) == expected_cols
        assert np.isnan(df.iloc[1]["bavg"])
        assert np.isnan(df.iloc[1]["bx_gsm"])
        assert np.isnan(df.iloc[1]["by_gsm"])
        assert np.isnan(df.iloc[1]["bz_gsm"])
        assert np.isnan(df.iloc[1]["speed"])
        assert np.isnan(df.iloc[1]["proton_density"])
        assert np.isnan(df.iloc[1]["temperature"])
        assert np.isnan(df.iloc[1]["sym-h"])
        assert df.iloc[0]["bavg"] == 5.1
        assert df.iloc[0]["bx_gsm"] == 1.2
        assert df.iloc[0]["by_gsm"] == 2.3
        assert df.iloc[0]["bz_gsm"] == 3.4
        assert df.iloc[0]["speed"] == 400
        assert df.iloc[0]["proton_density"] == 5.5
        assert df.iloc[0]["temperature"] == 1000000
        assert df.iloc[0]["sym-h"] == -15

    def test_process_single_month_handles_missing_data_lines(self, omni_high_res):
        data = ["YYYY DOY HR MN bavg bx_gsm by_gsm bz_gsm speed proton_density temperature sym-h"]
        with pytest.raises(ValueError):
            _ = omni_high_res._process_single_month(data)

    def test_process_single_month_raises_on_missing_header(self, omni_high_res):
        data = ["2020 1 0 0 5.1 1.2 2.3 3.4 400 5.5 1000000 -15"]
        with pytest.raises(ValueError, match="expected YYYY data header"):
            omni_high_res._process_single_month(data)

    def test_available_variables_cover_both_cadences(self, omni_high_res):
        one_minute = omni_high_res.available_variables(1)
        five_minute = omni_high_res.available_variables(5)

        assert len(one_minute) == 42
        assert len(five_minute) == 45
        assert one_minute["nasa_id"].tolist() == list(range(4, 46))
        assert five_minute["nasa_id"].tolist() == list(range(4, 49))
        assert {"name", "nasa_id", "description", "unit", "fill_value", "cadences", "aliases"} == set(
            one_minute.columns
        )
        assert dict(zip(five_minute["name"], five_minute["unit"])) == {
            variable.name: variable.unit for variable in HIGH_RES_VARIABLES
        }
        assert one_minute.set_index("name").loc["bavg", "fill_value"] == 9999.99

    @pytest.mark.parametrize("cadence,expected_count", [(1, 42), (5, 45)])
    def test_processes_complete_cadence_schema(self, omni_high_res, cadence, expected_count):
        names = [variable.name for variable in HIGH_RES_VARIABLES if cadence in variable.cadences]
        data = [
            "YYYY DOY HR MN " + " ".join(str(index) for index in range(1, expected_count + 1)),
            "2020 1 0 0 " + " ".join("1" for _ in names),
        ]

        result = omni_high_res._process_single_month(data, cadence_min=cadence, variable_names=names)

        assert list(result.columns) == names
        assert len(result.columns) == expected_count
        assert result.index.tz is not None

    def test_complete_request_uses_nasa_ids_in_registry_order(self, omni_high_res, mocker):
        response = mocker.Mock()
        response.text = "YYYY DOY HR MN\n"
        response.raise_for_status = mocker.Mock()
        post = mocker.patch("requests.post", return_value=response)
        names = [variable.name for variable in HIGH_RES_VARIABLES if 5 in variable.cadences]

        omni_high_res._get_data_from_omni(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 2, tzinfo=timezone.utc),
            cadence=5,
            variable_names=names,
        )

        payload = post.call_args.kwargs["data"]
        assert payload["vars"] == [str(identifier) for identifier in range(4, 49)]
        assert payload["res"] == "5min"
        assert payload["spacecraft"] == "omni_5min"
        assert post.call_args.kwargs["timeout"] == 30

    @pytest.mark.parametrize("cadence", [1, 5])
    def test_fill_values_are_masked_for_every_cadence_field(self, omni_high_res, cadence):
        variables = [variable for variable in HIGH_RES_VARIABLES if cadence in variable.cadences]
        row = " ".join(str(variable.fill_value) for variable in variables)
        data = ["YYYY DOY HR MN", f"2020 1 0 0 {row}"]

        result = omni_high_res._process_single_month(
            data,
            cadence_min=cadence,
            variable_names=[variable.name for variable in variables],
        )

        assert result.isna().all().all()

    def test_selection_preserves_order_deduplicates_and_accepts_aliases(self, tmp_path):
        omni_high_res = OMNIHighRes(data_dir=tmp_path)
        output = tmp_path / "2020/OMNI_HIGH_RES_1min_202001.csv"
        output.parent.mkdir(parents=True)
        pd.DataFrame(
            {"speed": [400.0], "sym-h": [-10.0]},
            index=pd.DatetimeIndex(["2020-01-01"], tz="UTC", name="timestamp"),
        ).to_csv(output)

        result = omni_high_res.read(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 1, 1, 0, 1, tzinfo=timezone.utc),
            variables=["sym_h", "speed", "sym-h"],
        )

        assert list(result.columns) == ["sym-h", "speed", "file_name"]
        assert result["file_name"].notna().all()

    def test_rejects_unknown_and_cadence_incompatible_variables(self, omni_high_res):
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2020, 1, 2, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="Unknown OMNI variables"):
            omni_high_res.read(start, end, variables="not_a_variable")
        with pytest.raises(ValueError, match="unavailable at 1-minute cadence"):
            omni_high_res.read(start, end, cadence_min=1, variables="proton_flux_10_mev")
        with pytest.raises(ValueError, match="at least one"):
            omni_high_res.read(start, end, variables=[])

    def test_partial_cache_requires_download_or_is_upgraded(self, tmp_path, mocker):
        omni_high_res = OMNIHighRes(data_dir=tmp_path)
        output = tmp_path / "2020/OMNI_HIGH_RES_1min_202001.csv"
        output.parent.mkdir(parents=True)
        index = pd.DatetimeIndex(["2020-01-01"], tz="UTC", name="timestamp")
        pd.DataFrame({"speed": [400.0]}, index=index).to_csv(output)
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2020, 1, 1, 0, 1, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="does not contain: ae"):
            omni_high_res.read(start, end, variables="ae")

        def upgrade(file_path, _time_interval, _cadence_min):
            pd.DataFrame({"speed": [400.0], "ae": [25.0]}, index=index).to_csv(file_path)

        upgrade_file = mocker.patch.object(
            omni_high_res,
            "_download_and_process_single_file",
            side_effect=upgrade,
        )
        result = omni_high_res.read(start, end, download=True, variables="ae")

        upgrade_file.assert_called_once()
        assert result["ae"].iloc[0] == 25.0

    def test_failed_processing_preserves_existing_cache_and_removes_temporary_file(self, tmp_path, mocker):
        omni_high_res = OMNIHighRes(data_dir=tmp_path)
        output = tmp_path / "2020/OMNI_HIGH_RES_1min_202001.csv"
        output.parent.mkdir(parents=True)
        output.write_text("original-cache\n")
        mocker.patch.object(omni_high_res, "_get_data_from_omni", return_value=["invalid"])
        mocker.patch.object(omni_high_res, "_process_single_month", side_effect=ValueError("bad response"))

        omni_high_res._download_and_process_single_file(
            output,
            (datetime(2020, 1, 1), datetime(2020, 1, 31)),
            1,
        )

        assert output.read_text() == "original-cache\n"
        assert not output.with_suffix(".csv.tmp").exists()

    @pytest.mark.parametrize("method", ["read", "download_and_process"])
    @pytest.mark.parametrize(
        "start,end",
        [
            (datetime(2020, 1, 1), datetime(2020, 1, 1)),
            (datetime(2020, 1, 2), datetime(2020, 1, 1)),
        ],
    )
    def test_equal_and_reversed_ranges_raise_value_error(self, omni_high_res, method, start, end):
        with pytest.raises(ValueError, match="start_time must be before end_time"):
            getattr(omni_high_res, method)(start, end)

    def test_mixed_naive_and_aware_times_are_normalized_before_comparison(self, tmp_path):
        reader = OMNIHighRes(data_dir=tmp_path)
        output = tmp_path / "2020/OMNI_HIGH_RES_1min_202001.csv"
        output.parent.mkdir(parents=True)
        pd.DataFrame(
            {name: [1.0] for name in HIGH_RES_DEFAULT_VARIABLES},
            index=pd.DatetimeIndex(["2020-01-01T00:00:00Z"], name="timestamp"),
        ).to_csv(output)

        result = reader.read(
            datetime(2020, 1, 1),
            datetime(2020, 1, 1, 0, 1, tzinfo=timezone.utc),
        )

        assert str(result.index.tz) == "UTC"

    @pytest.mark.parametrize(
        "doy,hour,minute,error",
        [
            (0, 0, 0, "invalid day-of-year, hour, or minute"),
            (367, 0, 0, "invalid day-of-year, hour, or minute"),
            (1, 24, 0, "invalid day-of-year, hour, or minute"),
            (1, 0, 60, "invalid day-of-year, hour, or minute"),
            (366, 0, 0, "invalid day-of-year for its record year"),
        ],
    )
    def test_rejects_invalid_time_fields(self, omni_high_res, doy, hour, minute, error):
        row = " ".join("1" for _ in HIGH_RES_DEFAULT_VARIABLES)
        data = ["YYYY DOY HR MN", f"2021 {doy} {hour} {minute} {row}"]

        with pytest.raises(ValueError, match=error):
            omni_high_res._process_single_month(data)

    def test_rejects_malformed_width_and_non_numeric_values(self, omni_high_res):
        with pytest.raises(ValueError, match="record widths"):
            omni_high_res._process_single_month(["YYYY DOY HR MN", "2020 1 0 0 1 2"])

        values = ["1" for _ in HIGH_RES_DEFAULT_VARIABLES]
        values[2] = "not-a-number"
        with pytest.raises(ValueError, match="Unable to parse|string"):
            omni_high_res._process_single_month(["YYYY DOY HR MN", f"2020 1 0 0 {' '.join(values)}"])

    def test_missing_header_has_public_value_error(self, omni_high_res):
        with pytest.raises(ValueError, match="expected YYYY data header"):
            omni_high_res._process_single_month(["2020 1 0 0 " + " ".join("1" for _ in range(9))])

    def test_empty_response_builds_utc_cadence_grid(self, omni_high_res):
        result = omni_high_res._process_single_month(
            [],
            original_end=datetime(2020, 1, 1, 0, 10),
            cadence_min=5,
        )

        assert len(result) == 3
        assert result.index.tolist() == list(pd.date_range("2020-01-01", periods=3, freq="5min", tz="UTC"))
        assert result.isna().all().all()

    def test_empty_response_without_end_preserves_requested_schema(self, omni_high_res):
        result = omni_high_res._process_single_month([], variable_names=["speed", "sym_h"])

        assert result.empty
        assert list(result.columns) == ["speed", "sym-h"]

    def test_values_below_fill_sentinel_are_preserved(self, omni_high_res):
        variables = [variable for variable in HIGH_RES_VARIABLES if 1 in variable.cadences]
        values = [float(variable.fill_value) - 0.01 for variable in variables]
        data = ["YYYY DOY HR MN", "2020 1 0 0 " + " ".join(map(str, values))]

        result = omni_high_res._process_single_month(
            data,
            cadence_min=1,
            variable_names=[variable.name for variable in variables],
        )

        assert result.notna().all().all()

    def test_provenance_depends_only_on_selected_fields(self, tmp_path):
        reader = OMNIHighRes(data_dir=tmp_path)
        output = tmp_path / "2020/OMNI_HIGH_RES_1min_202001.csv"
        output.parent.mkdir(parents=True)
        index = pd.DatetimeIndex(["2020-01-01T00:00Z", "2020-01-01T00:01Z"], name="timestamp")
        pd.DataFrame({"speed": [np.nan, 400.0], "ae": [20.0, np.nan]}, index=index).to_csv(output)

        result = reader.read(
            index[0].to_pydatetime(), (index[1] + timedelta(minutes=1)).to_pydatetime(), variables="speed"
        )

        assert pd.isna(result["file_name"].iloc[0])
        assert result["file_name"].iloc[1] == output

    def test_complete_cache_is_skipped_but_partial_and_reprocess_are_downloaded(self, tmp_path, mocker):
        reader = OMNIHighRes(data_dir=tmp_path)
        output = tmp_path / "2020/OMNI_HIGH_RES_1min_202001.csv"
        output.parent.mkdir(parents=True)
        all_names = [variable.name for variable in HIGH_RES_VARIABLES if 1 in variable.cadences]
        pd.DataFrame(columns=all_names).to_csv(output)
        download = mocker.patch.object(reader, "_download_and_process_single_file")
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 31)

        reader.download_and_process(start, end)
        download.assert_not_called()

        pd.DataFrame(columns=HIGH_RES_DEFAULT_VARIABLES).to_csv(output)
        reader.download_and_process(start, end)
        assert download.call_count == 1

        download.reset_mock()
        pd.DataFrame(columns=all_names).to_csv(output)
        reader.download_and_process(start, end, reprocess_files=True)
        assert download.call_count == 1

    def test_corrupt_cache_is_reported_or_replaced(self, tmp_path, mocker):
        reader = OMNIHighRes(data_dir=tmp_path)
        output = tmp_path / "2020/OMNI_HIGH_RES_1min_202001.csv"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"\xff\xfe\x00broken")
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2020, 1, 1, 0, 1, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="Cannot read processed OMNI file"):
            reader.read(start, end, variables="speed")

        def replace_cache(file_path, _interval, _cadence):
            pd.DataFrame(
                {"speed": [401.0]},
                index=pd.DatetimeIndex([start], name="timestamp"),
            ).to_csv(file_path)

        replacement = mocker.patch.object(reader, "_download_and_process_single_file", side_effect=replace_cache)
        result = reader.read(start, end, download=True, variables="speed")

        replacement.assert_called_once()
        assert result["speed"].iloc[0] == 401.0

    def test_unreadable_cache_after_failed_upgrade_has_clear_error(self, tmp_path, mocker):
        reader = OMNIHighRes(data_dir=tmp_path)
        output = tmp_path / "2020/OMNI_HIGH_RES_1min_202001.csv"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"\xff\xfe")
        mocker.patch.object(reader, "_download_and_process_single_file")

        with pytest.raises(ValueError, match="remains unreadable after attempted cache upgrade"):
            reader.read(
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                datetime(2020, 1, 1, 0, 1, tzinfo=timezone.utc),
                download=True,
                variables="speed",
            )

    def test_html_wrapped_correct_range_retries_with_suggested_end(self, omni_high_res, mocker):
        first = mocker.Mock()
        first.text = "<HTML>\n<H1>Error</H1>\nCorrect range: 20200101 - 20200102"
        first.raise_for_status = mocker.Mock()
        second = mocker.Mock()
        second.text = "YYYY DOY HR MN"
        second.raise_for_status = mocker.Mock()
        post = mocker.patch("requests.post", side_effect=[first, second])

        result = omni_high_res._get_data_from_omni(
            datetime(2020, 1, 1),
            datetime(2020, 1, 31),
        )

        assert result == ["YYYY DOY HR MN"]
        assert post.call_count == 2
        assert post.call_args_list[1].kwargs["data"]["end_date"] == "20200102"

    def test_correct_range_before_start_returns_empty(self, omni_high_res, mocker):
        response = mocker.Mock()
        response.text = "<HTML>\nError\ncorrect range: 20190101 - 20191231"
        response.raise_for_status = mocker.Mock()
        mocker.patch("requests.post", return_value=response)

        result = omni_high_res._get_data_from_omni(datetime(2020, 1, 1), datetime(2020, 1, 2))

        assert result == []

    def test_unspecified_html_error_and_http_error_propagate(self, omni_high_res, mocker):
        response = mocker.Mock()
        response.text = "<HTML>\n<H1>Error while processing request</H1>"
        response.raise_for_status = mocker.Mock()
        mocker.patch("requests.post", return_value=response)
        with pytest.raises(ValueError, match="unspecified error"):
            omni_high_res._get_data_from_omni(datetime(2020, 1, 1), datetime(2020, 1, 2))

        response.raise_for_status.side_effect = requests.HTTPError("503")
        with pytest.raises(requests.HTTPError):
            omni_high_res._get_data_from_omni(datetime(2020, 1, 1), datetime(2020, 1, 2))

    def test_parallel_worker_exception_is_not_silenced(self, tmp_path, mocker):
        reader = OMNIHighRes(data_dir=tmp_path)
        mocker.patch.object(reader, "_download_and_process_single_file", side_effect=RuntimeError("worker failed"))

        with pytest.raises(RuntimeError, match="worker failed"):
            reader.download_and_process(datetime(2020, 1, 1), datetime(2020, 12, 31))

    def test_range_entirely_before_mission_start_has_clear_error(self, omni_high_res):
        with pytest.raises(ValueError, match="ends before OMNI data begin in 1981"):
            omni_high_res.read(datetime(1970, 1, 1), datetime(1971, 1, 1))

    @pytest.mark.parametrize(
        "cadence,end,expected_months",
        [
            (1, datetime(2020, 1, 31, 23, 58, 59, tzinfo=timezone.utc), ["202001"]),
            (1, datetime(2020, 1, 31, 23, 59, tzinfo=timezone.utc), ["202001", "202002"]),
            (5, datetime(2020, 1, 31, 23, 55, tzinfo=timezone.utc), ["202001", "202002"]),
        ],
    )
    def test_month_boundary_neighbor_selection(self, tmp_path, cadence, end, expected_months):
        reader = OMNIHighRes(data_dir=tmp_path)

        paths, intervals = reader._get_processed_file_list(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            end,
            cadence,
        )

        assert [path.stem[-6:] for path in paths] == expected_months
        assert all(interval[0].tzinfo is not None and interval[1].tzinfo is not None for interval in intervals)

    def test_corrected_range_that_does_not_shorten_request_is_rejected(self, omni_high_res, mocker):
        response = mocker.Mock()
        response.text = "<HTML>\nError\ncorrect range: 20200101 - 20200131"
        response.raise_for_status = mocker.Mock()
        post = mocker.patch("requests.post", return_value=response)

        with pytest.raises(ValueError, match="would not shorten"):
            omni_high_res._get_data_from_omni(datetime(2020, 1, 1), datetime(2020, 1, 31))

        post.assert_called_once()

    def test_successful_reprocessing_replaces_cache_atomically(self, tmp_path, mocker):
        reader = OMNIHighRes(data_dir=tmp_path)
        output = tmp_path / "2020/OMNI_HIGH_RES_1min_202001.csv"
        output.parent.mkdir(parents=True)
        output.write_text("old-cache\n")
        all_names = [variable.name for variable in HIGH_RES_VARIABLES if 1 in variable.cadences]
        processed = pd.DataFrame(
            {name: [1.0] for name in all_names},
            index=pd.DatetimeIndex(["2020-01-01T00:00Z"], name="timestamp"),
        )
        mocker.patch.object(reader, "_get_data_from_omni", return_value=[])
        mocker.patch.object(reader, "_process_single_month", return_value=processed)

        reader._download_and_process_single_file(
            output,
            (datetime(2020, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 31, tzinfo=timezone.utc)),
            1,
        )

        assert pd.read_csv(output)["speed"].iloc[0] == 1.0
        assert not output.with_suffix(".csv.tmp").exists()

    @pytest.mark.skipif(
        os.environ.get("SWVO_RUN_LIVE_TESTS") != "1",
        reason="set SWVO_RUN_LIVE_TESTS=1 for the one-day NASA OMNIWeb smoke check",
    )
    @pytest.mark.parametrize("cadence", [1, 5])
    def test_live_one_day_complete_omniweb_response(self, tmp_path, cadence):
        omni_high_res = OMNIHighRes(data_dir=tmp_path)
        names = [variable.name for variable in HIGH_RES_VARIABLES if cadence in variable.cadences]
        raw = omni_high_res._get_data_from_omni(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            cadence=cadence,
            variable_names=names,
        )

        result = omni_high_res._process_single_month(raw, cadence_min=cadence, variable_names=names)

        assert list(result.columns) == names
        assert not result.empty

    def test_remove_processed_file(self):
        shutil.rmtree(Path(TEST_DIR) / "data/OMNI/2022", ignore_errors=True)
        shutil.rmtree(Path(TEST_DIR) / "data/OMNI/2023", ignore_errors=True)
