# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import pandas as pd
import logging
from swvo.io.utils import any_nans, nan_percentage


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
