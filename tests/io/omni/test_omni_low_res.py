# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from datetime import datetime, timezone
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

    def test_remove_processed_file(self):
        os.remove(Path(TEST_DIR) / "data/OMNI_LOW_RES_2020.csv")
