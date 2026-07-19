# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Simon Mischel
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from swvo.io.omni.omni_low_res import OMNILowRes
from swvo.io.omni.variables import LOW_RES_DEFAULT_VARIABLES, LOW_RES_VARIABLES

TEST_DIR = os.path.dirname(__file__)
DATA_DIR = Path(os.path.join(TEST_DIR, "data/"))


class TestOMNILowRes:
    @pytest.fixture
    def omni_low_res(self):
        os.environ["OMNI_LOW_RES_STREAM_DIR"] = str(DATA_DIR)
        yield OMNILowRes()

    def test_initialization_with_env_var(self, omni_low_res):
        assert omni_low_res.data_dir.exists()

    def test_initialization_with_data_dir(self):
        omni_low_res = OMNILowRes(data_dir=DATA_DIR)
        assert omni_low_res.data_dir == DATA_DIR

    def test_initialization_without_env_var(self):
        if "OMNI_LOW_RES_STREAM_DIR" in os.environ:
            del os.environ["OMNI_LOW_RES_STREAM_DIR"]
        with pytest.raises(ValueError):
            OMNILowRes()

    def test_download_and_process(self, omni_low_res, mocker):
        mock_response = mocker.Mock()
        mock_response.content = b"test content"
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch("requests.get", return_value=mock_response)
        mocker.patch.object(
            omni_low_res,
            "_process_single_file",
            return_value=omni_low_res._process_single_file(Path(TEST_DIR) / "data/omni2_2020.dat"),
        )

        start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2020, 12, 31, tzinfo=timezone.utc)

        omni_low_res.download_and_process(start_time, end_time)

        assert (TEST_DIR / Path("data/omni2_2020.dat")).exists()

    def test_read_without_download(self, omni_low_res, mocker):
        start_time = datetime(2021, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2021, 12, 31, tzinfo=timezone.utc)
        with warnings.catch_warnings(record=True) as w:
            df = omni_low_res.read(start_time, end_time, download=False)
            assert "OMNI_LOW_RES_2021.csv not found" in str(w[-1].message)
            assert not df.empty
            assert "f107" in df.columns
            assert "kp" in df.columns
            assert "dst" in df.columns
            assert all(df["f107"].isna())
            assert all(df["kp"].isna())
            assert all(df["dst"].isna())
            assert all(df["file_name"].isnull())

    def test_read_with_download(self, omni_low_res, mocker):
        mocker.patch.object(omni_low_res, "download_and_process")
        mocker.patch.object(
            omni_low_res,
            "_read_single_file",
            return_value=pd.DataFrame(
                index=pd.date_range(start=datetime(2022, 1, 1), end=datetime(2022, 12, 31), tz=timezone.utc)
            ),
        )
        start_time = datetime(2022, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2022, 12, 31, tzinfo=timezone.utc)
        omni_low_res.read(start_time, end_time, download=True)
        omni_low_res.download_and_process.assert_called_once()

    def test_process_single_file(self, omni_low_res):
        file = Path(TEST_DIR) / "data/omni2_2020.dat"
        df = omni_low_res._process_single_file(file)

        assert isinstance(df, pd.DataFrame)
        assert all(column in df.columns for column in ["kp", "dst", "f107"])
        assert len(df.columns) == 54
        assert df["lyman_alpha"].isna().all()
        assert df["proton_quasi_invariant"].isna().all()
        assert len(df) > 0

    def test_read_single_file(self, omni_low_res):
        csv_file = Path(TEST_DIR) / "data/OMNI_LOW_RES_2020.csv"
        df = omni_low_res._read_single_file(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert all(column in df.columns for column in ["kp", "dst", "f107"])
        assert len(df) > 0

    def test_start_year_behind(self, omni_low_res, mocker):
        start_time = datetime(1920, 1, 1)
        end_time = datetime(2020, 12, 31)

        mocked_df = pd.DataFrame(index=pd.date_range(start_time, end_time))

        mocker.patch.object(omni_low_res, "_get_processed_file_list", return_value=([], []))
        mocker.patch.object(omni_low_res, "_read_single_file", return_value=mocked_df)

        mocker.patch("pandas.concat", return_value=pd.DataFrame())
        mocker.patch.object(pd.DataFrame, "truncate", return_value=pd.DataFrame())

        with patch("logging.Logger.warning") as mock_warning:
            df = omni_low_res.read(start_time, end_time)
            mock_warning.assert_any_call(
                "Start date chosen falls behind the existing data. Moving start date to first available mission files..."
            )

            assert "f107" in df.columns
            assert "kp" in df.columns
            assert "dst" in df.columns
            assert all(df["f107"].isna())
            assert all(df["kp"].isna())
            assert all(df["dst"].isna())
            assert all(df["file_name"].isnull())

    def test_available_variables_describe_all_non_time_fields(self, omni_low_res):
        variables = omni_low_res.available_variables()

        assert len(variables) == 54
        assert variables["name"].tolist() == [variable.name for variable in LOW_RES_VARIABLES]
        assert {"name", "description", "unit", "fill_value", "aliases"} == set(variables.columns)
        assert dict(zip(variables["name"], variables["unit"])) == {
            variable.name: variable.unit for variable in LOW_RES_VARIABLES
        }

    def test_available_variables_delegates_to_shared_utility(self, omni_low_res, mocker):
        expected = pd.DataFrame({"name": ["dst"]})
        utility = mocker.patch("swvo.io.omni.omni_low_res.get_available_variables", return_value=expected)

        result = omni_low_res.available_variables()

        assert result is expected
        utility.assert_called_once_with()

    def test_private_cache_helper_is_an_instance_method(self):
        assert not isinstance(OMNILowRes.__dict__["available_variables"], classmethod)
        assert not isinstance(OMNILowRes.__dict__["_cache_contains"], staticmethod)

    def test_fill_values_are_masked_for_every_hourly_field(self, omni_low_res, tmp_path):
        values = ["0" if variable.fill_value is None else str(variable.fill_value) for variable in LOW_RES_VARIABLES]
        raw_file = tmp_path / "omni2_fill_values.dat"
        raw_file.write_text(" ".join(["2020", "1", "0", *values]) + "\n")

        result = omni_low_res._process_single_file(raw_file)

        assert result.drop(columns="magnetospheric_flux_flag").isna().all().all()
        assert result["magnetospheric_flux_flag"].iloc[0] == 0

    def test_processes_current_57_word_record(self, omni_low_res, tmp_path):
        historic_line = (Path(TEST_DIR) / "data/omni2_2020.dat").read_text().splitlines()[0]
        current_file = tmp_path / "omni2_current.dat"
        current_file.write_text(f"{historic_line} 0.005123 0.4567\n")

        result = omni_low_res._process_single_file(current_file)

        assert len(result.columns) == 54
        assert result["lyman_alpha"].iloc[0] == pytest.approx(0.005123)
        assert result["proton_quasi_invariant"].iloc[0] == pytest.approx(0.4567)

    def test_all_and_subset_selection_preserve_public_schema(self, tmp_path):
        omni_low_res = OMNILowRes(data_dir=tmp_path)
        processed = omni_low_res._process_single_file(Path(TEST_DIR) / "data/omni2_2020.dat")
        output = tmp_path / "OMNI_LOW_RES_2020.csv"
        processed.to_csv(output, index=True)
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2020, 1, 1, 1, tzinfo=timezone.utc)

        default = omni_low_res.read(start, end)
        all_variables = omni_low_res.read(start, end, variables="all")
        subset = omni_low_res.read(start, end, variables=["pc", "speed", "pcn"])

        assert list(default.columns) == [*LOW_RES_DEFAULT_VARIABLES, "file_name"]
        assert list(all_variables.columns) == [*[variable.name for variable in LOW_RES_VARIABLES], "file_name"]
        assert list(subset.columns) == ["pcn", "speed", "file_name"]
        assert all_variables.index.tz is not None
        assert subset["file_name"].notna().any()

    def test_rejects_unknown_and_empty_variable_selection(self, omni_low_res):
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2020, 1, 2, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="Unknown OMNI variables"):
            omni_low_res.read(start, end, variables="not_a_variable")
        with pytest.raises(ValueError, match="at least one"):
            omni_low_res.read(start, end, variables=[])

    def test_partial_cache_requires_download_or_is_upgraded(self, tmp_path, mocker):
        omni_low_res = OMNILowRes(data_dir=tmp_path)
        output = tmp_path / "OMNI_LOW_RES_2020.csv"
        index = pd.DatetimeIndex(["2020-01-01"], tz="UTC", name="timestamp")
        pd.DataFrame({"dst": [-5.0], "kp": [1.0], "f107": [70.0]}, index=index).to_csv(output)
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2020, 1, 1, 1, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="does not contain: speed"):
            omni_low_res.read(start, end, variables="speed")

        def upgrade(_start, _end, reprocess_files=False):
            assert reprocess_files
            pd.DataFrame({"speed": [400.0]}, index=index).to_csv(output)

        upgrade_cache = mocker.patch.object(omni_low_res, "download_and_process", side_effect=upgrade)
        result = omni_low_res.read(start, end, download=True, variables="speed")

        upgrade_cache.assert_called_once()
        assert result["speed"].iloc[0] == 400.0

    def test_failed_processing_preserves_existing_cache_and_removes_temporary_file(self, tmp_path, mocker):
        omni_low_res = OMNILowRes(data_dir=tmp_path)
        output = tmp_path / "OMNI_LOW_RES_2020.csv"
        output.write_text("original-cache\n")
        mocker.patch.object(omni_low_res, "_download")
        mocker.patch.object(omni_low_res, "_process_single_file", side_effect=ValueError("bad response"))

        omni_low_res.download_and_process(
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2020, 12, 30, tzinfo=timezone.utc),
            reprocess_files=True,
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
    def test_equal_and_reversed_ranges_raise_value_error(self, omni_low_res, method, start, end):
        with pytest.raises(ValueError, match="start_time must be before end_time"):
            getattr(omni_low_res, method)(start, end)

    def test_mixed_naive_and_aware_times_are_normalized_before_comparison(self, tmp_path):
        reader = OMNILowRes(data_dir=tmp_path)

        result = reader.read(
            datetime(2020, 1, 1),
            datetime(2020, 1, 1, 1, tzinfo=timezone.utc),
            variables="speed",
        )

        assert str(result.index.tz) == "UTC"
        assert result["speed"].isna().all()

    @pytest.mark.parametrize(
        "word_count",
        [54, 56, 58],
    )
    def test_rejects_unsupported_record_widths(self, omni_low_res, tmp_path, word_count):
        raw_file = tmp_path / f"omni2_{word_count}.dat"
        raw_file.write_text(" ".join(["2020", "1", "0", *(["1"] * (word_count - 3))]) + "\n")

        with pytest.raises(ValueError, match="expected 55 historic or 57 current"):
            omni_low_res._process_single_file(raw_file)

    def test_empty_and_mixed_width_sources_have_clear_errors(self, omni_low_res, tmp_path):
        empty = tmp_path / "empty.dat"
        empty.write_text("")
        with pytest.raises(ValueError, match="Cannot parse OMNI2 source file"):
            omni_low_res._process_single_file(empty)

        historic = (Path(TEST_DIR) / "data/omni2_2020.dat").read_text().splitlines()[0]
        mixed = tmp_path / "mixed.dat"
        mixed.write_text(f"{historic}\n{historic} 0.1 0.2\n")
        with pytest.raises(ValueError, match="Cannot parse OMNI2 source file"):
            omni_low_res._process_single_file(mixed)

    @pytest.mark.parametrize(
        "day,hour,error",
        [
            (0, 0, "invalid day-of-year or hour"),
            (367, 0, "invalid day-of-year or hour"),
            (1, 24, "invalid day-of-year or hour"),
            (366, 0, "invalid day-of-year for its record year"),
        ],
    )
    def test_rejects_invalid_hourly_time_fields(self, omni_low_res, tmp_path, day, hour, error):
        values = ["1" for _ in range(len(LOW_RES_VARIABLES) - 2)]
        raw_file = tmp_path / "invalid_time.dat"
        raw_file.write_text(" ".join(["2021", str(day), str(hour), *values]) + "\n")

        with pytest.raises(ValueError, match=error):
            omni_low_res._process_single_file(raw_file)

    def test_kp_integer_tenths_are_converted_to_conventional_thirds(self, omni_low_res, tmp_path):
        kp_index = 3 + next(index for index, variable in enumerate(LOW_RES_VARIABLES) if variable.name == "kp")
        rows = []
        for hour, kp in enumerate([0, 3, 7, 10, 13]):
            words = ["2020", "1", str(hour), *(["1"] * (len(LOW_RES_VARIABLES) - 2))]
            words[kp_index] = str(kp)
            rows.append(" ".join(words))
        raw_file = tmp_path / "kp.dat"
        raw_file.write_text("\n".join(rows) + "\n")

        result = omni_low_res._process_single_file(raw_file)

        assert result["kp"].tolist() == pytest.approx([0, 1 / 3, 2 / 3, 1, 4 / 3])

    def test_provenance_depends_only_on_selected_fields(self, tmp_path):
        reader = OMNILowRes(data_dir=tmp_path)
        output = tmp_path / "OMNI_LOW_RES_2020.csv"
        index = pd.DatetimeIndex(["2020-01-01T00:00Z", "2020-01-01T01:00Z"], name="timestamp")
        pd.DataFrame({"speed": [float("nan"), 400.0], "dst": [-10.0, float("nan")]}, index=index).to_csv(output)

        result = reader.read(
            index[0].to_pydatetime(),
            (index[1] + timedelta(hours=1)).to_pydatetime(),
            variables="speed",
        )

        assert pd.isna(result.loc[index[0], "file_name"])
        assert result.loc[index[1], "file_name"] == output

    def test_complete_cache_is_skipped_but_partial_and_reprocess_are_downloaded(self, tmp_path, mocker):
        reader = OMNILowRes(data_dir=tmp_path)
        output = tmp_path / "OMNI_LOW_RES_2020.csv"
        all_names = [variable.name for variable in LOW_RES_VARIABLES]
        pd.DataFrame(columns=all_names).to_csv(output)
        download = mocker.patch.object(reader, "_download")
        process = mocker.patch.object(reader, "_process_single_file", return_value=pd.DataFrame(columns=all_names))
        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 30)

        reader.download_and_process(start, end)
        download.assert_not_called()
        process.assert_not_called()

        pd.DataFrame(columns=LOW_RES_DEFAULT_VARIABLES).to_csv(output)
        reader.download_and_process(start, end)
        assert download.call_count == 1
        assert process.call_count == 1

        download.reset_mock()
        process.reset_mock()
        pd.DataFrame(columns=all_names).to_csv(output)
        reader.download_and_process(start, end, reprocess_files=True)
        assert download.call_count == 1
        assert process.call_count == 1

    def test_downloader_uses_and_cleans_os_temporary_directory(self, tmp_path, mocker):
        reader = OMNILowRes(data_dir=tmp_path)
        observed_directories = []

        def download(temporary_dir, filename):
            observed_directories.append(temporary_dir)
            assert temporary_dir.is_dir()
            (temporary_dir / filename).write_text("raw")

        mocker.patch.object(reader, "_download", side_effect=download)
        mocker.patch.object(
            reader,
            "_process_single_file",
            return_value=pd.DataFrame(
                {name: [1.0] for name in (variable.name for variable in LOW_RES_VARIABLES)},
                index=pd.DatetimeIndex(["2020-01-01T00:00Z"], name="timestamp"),
            ),
        )

        reader.download_and_process(datetime(2020, 1, 1), datetime(2020, 12, 30))

        assert len(observed_directories) == 1
        assert not observed_directories[0].exists()

    def test_corrupt_cache_is_reported_or_replaced(self, tmp_path, mocker):
        reader = OMNILowRes(data_dir=tmp_path)
        output = tmp_path / "OMNI_LOW_RES_2020.csv"
        output.write_bytes(b"\xff\xfe\x00broken")
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = datetime(2020, 1, 1, 1, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="Cannot read processed OMNI file"):
            reader.read(start, end, variables="speed")

        def replace_cache(_start, _end, reprocess_files=False):
            assert reprocess_files
            pd.DataFrame(
                {"speed": [402.0]},
                index=pd.DatetimeIndex([start], name="timestamp"),
            ).to_csv(output)

        replacement = mocker.patch.object(reader, "download_and_process", side_effect=replace_cache)
        result = reader.read(start, end, download=True, variables="speed")

        replacement.assert_called_once()
        assert result.loc[pd.Timestamp(start), "speed"] == 402.0

    def test_unreadable_cache_after_failed_upgrade_has_clear_error(self, tmp_path, mocker):
        reader = OMNILowRes(data_dir=tmp_path)
        output = tmp_path / "OMNI_LOW_RES_2020.csv"
        output.write_bytes(b"\xff\xfe")
        mocker.patch.object(reader, "download_and_process")

        with pytest.raises(ValueError, match="remains unreadable after attempted cache upgrade"):
            reader.read(
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                datetime(2020, 1, 1, 1, tzinfo=timezone.utc),
                download=True,
                variables="speed",
            )

    def test_year_file_selection_includes_only_required_boundary_neighbor(self, tmp_path):
        reader = OMNILowRes(data_dir=tmp_path)

        regular, _ = reader._get_processed_file_list(
            datetime(2020, 6, 1, tzinfo=timezone.utc),
            datetime(2020, 12, 31, 20, 59, tzinfo=timezone.utc),
        )
        boundary, _ = reader._get_processed_file_list(
            datetime(2020, 6, 1, tzinfo=timezone.utc),
            datetime(2020, 12, 31, 21, 0, tzinfo=timezone.utc),
        )

        assert [path.name for path in regular] == ["OMNI_LOW_RES_2020.csv"]
        assert [path.name for path in boundary] == ["OMNI_LOW_RES_2020.csv", "OMNI_LOW_RES_2021.csv"]

    def test_range_entirely_before_mission_start_has_clear_error(self, omni_low_res):
        with pytest.raises(ValueError, match="ends before OMNI data begin in 1963"):
            omni_low_res.read(datetime(1950, 1, 1), datetime(1951, 1, 1))

    def test_nonnumeric_hourly_field_has_clear_error(self, omni_low_res, tmp_path):
        words = ["2020", "1", "0", *(["1"] * (len(LOW_RES_VARIABLES) - 2))]
        words[10] = "not-a-number"
        raw_file = tmp_path / "nonnumeric.dat"
        raw_file.write_text(" ".join(words) + "\n")

        with pytest.raises(ValueError, match="contains nonnumeric fields"):
            omni_low_res._process_single_file(raw_file)

    def test_remove_processed_file(self):
        os.remove(Path(TEST_DIR) / "data/OMNI_LOW_RES_2020.csv")
