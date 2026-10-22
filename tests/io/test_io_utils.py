# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime, timezone

import pandas as pd
import pytest

from swvo.io.utils import any_nans, construct_updated_data_frame, nan_percentage, start_of_next_month


def test_any_nans_single_dataframe_with_nans():
    df = pd.DataFrame({"A": [1.0, 2.0, None], "B": [4.0, None, 6.0]})
    assert any_nans(df) is True


def test_any_nans_single_dataframe_without_nans():
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    assert any_nans(df) is False


def test_any_nans_list_of_dataframes_with_nans():
    df1 = pd.DataFrame({"A": [1.0, None, 3.0], "B": [4.0, 5.0, 6.0]})
    df2 = pd.DataFrame({"X": [1.0, 2.0, 3.0], "Y": [None, 5.0, 6.0]})
    assert any_nans([df1, df2]) is True


def test_any_nans_list_of_dataframes_without_nans():
    df1 = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    df2 = pd.DataFrame({"X": [1.0, 2.0, 3.0], "Y": [7.0, 5.0, 6.0]})
    assert any_nans([df1, df2]) is False


def test_nan_percentage_with_nans(caplog):
    df = pd.DataFrame({"A": [1.0, None, 3.0], "B": [4.0, None, 6.0]})
    with caplog.at_level(logging.INFO):
        percentage = nan_percentage(df)
        assert percentage == pytest.approx(33.33, rel=1e-2)
        assert "Percentage of NaNs in data frame: 33.33%" in caplog.text


def test_nan_percentage_without_nans(caplog):
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    with caplog.at_level(logging.INFO):
        percentage = nan_percentage(df)
        assert percentage == 0.0
        assert "Percentage of NaNs in data frame: 0.00%" in caplog.text


def test_any_nans_invalid_input():
    with pytest.raises(TypeError):
        any_nans("invalid input")


def test_any_nans_empty_list():
    assert any_nans([]) is False


@pytest.mark.parametrize(
    "time, expected",
    [
        (datetime(2024, 1, 1), datetime(2024, 2, 1)),
        (datetime(2024, 2, 29, 23, 59, 59), datetime(2024, 3, 1)),
        (datetime(2024, 11, 15, 12, 30), datetime(2024, 12, 1)),
        (datetime(2024, 12, 31, 23, 59, 59), datetime(2025, 1, 1)),
    ],
)
def test_start_of_next_month(time, expected):
    assert start_of_next_month(time) == expected


def test_start_of_next_month_keeps_timezone():
    time = datetime(2024, 12, 15, 6, tzinfo=timezone.utc)

    result = start_of_next_month(time)

    assert result == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert result.tzinfo == timezone.utc


def _omni_tail_and_ace_frames():
    index = pd.date_range("2026-09-30 00:00", periods=4, freq="1h", tz="UTC")
    old = pd.DataFrame(
        {
            "speed": [400.0, None, None, None],
            "bz_gsm": [1.0, None, None, None],
            "sym-h": [-5.0, -6.0, -7.0, -8.0],
            "file_name": ["omni.csv"] * 4,
            "model": ["omni", None, None, None],
        },
        index=index,
    )
    new = pd.DataFrame({"speed": [410.0, 420.0, 430.0, 440.0], "bz_gsm": [2.0, 3.0, 4.0, 5.0]}, index=index)
    new["file_name"] = "ace.csv"
    return old, new


def test_construct_updated_data_frame_value_columns_fills_nan_tail():
    old, new = _omni_tail_and_ace_frames()

    result = construct_updated_data_frame(old, new, "ace", value_columns=["speed", "bz_gsm"])[0]

    assert result["speed"].tolist() == [400.0, 420.0, 430.0, 440.0]
    assert result["model"].tolist() == ["omni", "ace", "ace", "ace"]
    assert result["file_name"].tolist() == ["omni.csv", "ace.csv", "ace.csv", "ace.csv"]
    assert result["sym-h"].tolist() == [-5.0, -6.0, -7.0, -8.0]


def test_construct_updated_data_frame_without_value_columns_keeps_rows_with_file_name():
    old, new = _omni_tail_and_ace_frames()

    result = construct_updated_data_frame(old, new, "ace")[0]

    assert result["speed"].isna().tolist() == [False, True, True, True]
    assert result["file_name"].tolist() == ["omni.csv"] * 4
