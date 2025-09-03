# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: S101

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from swvo.io.hp.gfz import Hp30GFZ, Hp60GFZ, HpGFZ


class TestHpGFZ:
    @pytest.fixture
    def mock_hp_data(self):
        test_dates = pd.date_range(
            start=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end=datetime(2020, 12, 31, tzinfo=timezone.utc),
            freq="30min",
        )
        test_data = pd.DataFrame(
            {
                "t": test_dates,
                "hp30": [15.0] * len(test_dates),
            }
        )
        test_data.index = test_dates
        return test_data

    @pytest.fixture
    def hp30gfz(self, tmp_path):
        data_dir = tmp_path / "hp_data"
        return Hp30GFZ(data_dir=data_dir)

    @pytest.fixture
    def hp60gfz(self, tmp_path):
        data_dir = tmp_path / "hp_data"
        return Hp60GFZ(data_dir=data_dir)

    def test_hp30gfz_initialization(self, hp30gfz):
        assert hp30gfz.index == "hp30"
        assert hp30gfz.index_number == "30"
        assert isinstance(hp30gfz.data_dir, Path)

    def test_hp60gfz_initialization(self, hp60gfz):
        assert hp60gfz.index == "hp60"
        assert hp60gfz.index_number == "60"
        assert isinstance(hp60gfz.data_dir, Path)

    def test_invalid_index(self):
        with pytest.raises(ValueError, match="Encountered invalid index:.*"):
            HpGFZ("hp45")

    def test_missing_env_var(self):
        with pytest.raises(
            ValueError,
            match="Necessary environment variable RT_HP_GFZ_STREAM_DIR not set!",
        ):
            Hp30GFZ()

    def test_read_with_download(self, hp30gfz, mocker, mock_hp_data):
        end_time = datetime(2020, 12, 31)  # noqa: DTZ001
        start_time = datetime(2020, 1, 1)  # noqa: DTZ001

        mocker.patch("pathlib.Path.exists", return_value=False)
        mocker.patch.object(hp30gfz, "download_and_process")
        mocker.patch.object(hp30gfz, "_read_single_file", return_value=mock_hp_data)

        df = hp30gfz.read(start_time, end_time, download=True)

        assert not df.empty
        assert df.index.tz == timezone.utc
        assert "hp30" in df.columns

    def test_start_year_behind(self, hp30gfz, mocker, mock_hp_data):
        start_time = datetime(1980, 1, 1)  # noqa: DTZ001
        end_time = datetime(2020, 12, 31)  # noqa: DTZ001

        mocker.patch("pathlib.Path.exists", return_value=True)
        mocker.patch.object(hp30gfz, "_read_single_file", return_value=mock_hp_data)

        with patch("logging.Logger.warning") as mock_warning:
            _ = hp30gfz.read(start_time, end_time)
            mock_warning.assert_any_call(
                "Start date chosen falls behind the mission starting year. Moving start date to first"
                " available mission files..."
            )

    def test_process_single_file(self, hp30gfz, tmp_path, mocker):
        file_content = """# Header
    2020 01 01 0000 0.0 0.0 0.0 15.0
    2020 01 01 0030 0.0 0.0 0.0 16.0
    2020 01 01 0100 0.0 0.0 0.0 17.0"""

        m = mock_open(read_data=file_content)

        mocker.patch("builtins.open", m)
        df = hp30gfz._process_single_file(tmp_path, ["test_file.txt"])

        assert len(df) == 3
        assert df.index[0] == pd.Timestamp("2020-01-01 00:00:00", tzinfo=timezone.utc)
        assert df.iloc[0]["hp30"] == 15.0
        assert "hp30" in df.columns

    def test_get_processed_file_list(self, hp30gfz):
        start_time = datetime(2020, 1, 1)  # noqa: DTZ001
        end_time = datetime(2021, 12, 31)  # noqa: DTZ001

        file_paths, time_intervals = hp30gfz._get_processed_file_list(start_time, end_time)

        assert len(file_paths) == 2
        assert all(isinstance(p, Path) for p in file_paths)
        assert len(time_intervals) == 2
        assert time_intervals[0][0].year == 2020
        assert time_intervals[1][0].year == 2021

    def test_invalid_time_range(self, hp30gfz):
        end_time = datetime(2020, 1, 1)  # noqa: DTZ001
        start_time = datetime(2020, 12, 31)  # noqa: DTZ001

        with pytest.raises(AssertionError):
            hp30gfz.read(start_time, end_time)

    def test_download_and_process(self, hp30gfz, mocker):
        start_time = datetime(2020, 1, 1)  # noqa: DTZ001
        end_time = datetime(2020, 12, 31)  # noqa: DTZ001

        mocked = mocker.patch("wget.download")
        mocker.patch("shutil.rmtree")
        mocker.patch.object(hp30gfz, "_process_single_file", return_value=pd.DataFrame())

        hp30gfz.download_and_process(start_time, end_time)

        mocked.assert_called()

    @pytest.fixture
    def sample_csv_data(self):
        test_dates = pd.date_range(
            start=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end=datetime(2020, 1, 1, 1, tzinfo=timezone.utc),
            freq="30min",
        )
        return pd.DataFrame({"t": test_dates, "hp30": [15.0, 16.0, 17.0]})

    @pytest.mark.parametrize(
        ("index_name", "values"),
        [
            ("hp30", [15.0, 16.0, 17.0]),
            ("hp60", [20.0, 21.0, 22.0]),
        ],
    )
    def test_read_single_file(self, hp30gfz, hp60gfz, tmp_path, index_name, values):
        instance = hp30gfz if index_name == "hp30" else hp60gfz

        test_dates = pd.date_range(
            start=datetime(2020, 1, 1, tzinfo=timezone.utc),
            end=datetime(2020, 1, 1, 1, tzinfo=timezone.utc),
            freq="30min",
        ).strftime("%Y-%m-%d %H:%M:%S")
        test_data = pd.DataFrame({"t": test_dates, index_name: values})

        test_file = tmp_path / "test_data.csv"
        test_data.to_csv(test_file, index=False, header=False)

        result_df = instance._read_single_file(test_file)  # noqa: SLF001

        assert isinstance(result_df, pd.DataFrame)
        assert index_name in result_df.columns
        assert "t" not in result_df.columns
        assert len(result_df) == 3
        assert isinstance(result_df.index, pd.DatetimeIndex)
        assert result_df[index_name].tolist() == values
        assert result_df.index[0] == pd.Timestamp("2020-01-01 00:00:00", tzinfo=timezone.utc)
        assert result_df.index[-1] == pd.Timestamp("2020-01-01 01:00:00", tzinfo=timezone.utc)
