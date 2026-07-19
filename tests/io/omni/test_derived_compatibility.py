# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Simon Mischel
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pandas as pd
import pytest

from swvo.io.dst import DSTOMNI
from swvo.io.f10_7 import F107OMNI
from swvo.io.kp import KpOMNI
from swvo.io.omni import OMNILowRes
from swvo.io.symh import SymhOMNI


@pytest.mark.parametrize("reader_class", [DSTOMNI, F107OMNI, KpOMNI, SymhOMNI])
@pytest.mark.parametrize(
    "start,end",
    [
        (datetime(2020, 1, 1), datetime(2020, 1, 1, tzinfo=timezone.utc)),
        (datetime(2020, 1, 2), datetime(2020, 1, 1, tzinfo=timezone.utc)),
    ],
)
def test_focused_omni_readers_reject_equal_or_reversed_mixed_timezone_ranges(
    tmp_path,
    reader_class,
    start,
    end,
):
    reader = reader_class(data_dir=tmp_path)

    with pytest.raises(ValueError, match="start_time must be before end_time"):
        reader.read(start, end, download=False)


@pytest.mark.parametrize(
    "reader_class,variable,expected",
    [
        (
            DSTOMNI,
            "dst",
            pd.date_range("2020-01-02 01:00", "2020-01-02 05:00", freq="h", tz="UTC"),
        ),
        (
            KpOMNI,
            "kp",
            pd.date_range("2020-01-02 00:00", "2020-01-02 06:00", freq="3h", tz="UTC"),
        ),
        (
            F107OMNI,
            "f107",
            pd.date_range("2020-01-02 00:00", "2020-01-03 00:00", freq="24h", tz="UTC"),
        ),
    ],
)
def test_hourly_focused_readers_preserve_boundary_padding_in_parent(
    tmp_path,
    mocker,
    reader_class,
    variable,
    expected,
):
    source_path = tmp_path / "complete.csv"
    source_index = pd.date_range("2020-01-01", "2020-01-04", freq="h", tz="UTC")
    pd.DataFrame({"timestamp": source_index, variable: 1.0}).to_csv(source_path, index=False)

    reader = reader_class(data_dir=tmp_path)
    mocker.patch.object(
        reader,
        "_get_processed_file_list",
        return_value=([source_path], [(datetime(2020, 1, 1), datetime(2021, 1, 1))]),
    )

    result = reader.read(
        datetime(2020, 1, 2, 1, 30),
        datetime(2020, 1, 2, 4, 30, tzinfo=timezone.utc),
        download=False,
    )

    pd.testing.assert_index_equal(result.index, expected, check_names=False)
    assert result.columns.tolist() == [variable, "file_name"]
    assert result[variable].eq(1.0).all()
    assert result["file_name"].eq(source_path).all()


@pytest.mark.parametrize(
    "reader_class,variable",
    [(DSTOMNI, "dst"), (KpOMNI, "kp"), (F107OMNI, "f107")],
)
def test_hourly_focused_readers_delegate_raw_times_to_parent(
    tmp_path,
    mocker,
    reader_class,
    variable,
):
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2, tzinfo=timezone.utc)
    parent_result = pd.DataFrame(
        {variable: [1.0], "file_name": [tmp_path / "source.csv"]},
        index=pd.DatetimeIndex(["2020-01-01T00:00:00Z"]),
    )
    parent_read = mocker.patch.object(OMNILowRes, "read", return_value=parent_result)

    reader_class(data_dir=tmp_path).read(start, end, download=True)

    parent_read.assert_called_once_with(start, end, download=True, variables=variable)
