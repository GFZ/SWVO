# SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import pytest

from swvo.io.dst import DSTOMNI
from swvo.io.f10_7 import F107OMNI
from swvo.io.kp import KpOMNI
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
