from datetime import datetime, timedelta

import numpy as np

from swvo.io.RBMDataSet.identify_orbits import _identify_orbits  # type: ignore[reportPrivateUsage]


def test_identify_orbits_clean_abs_sin():
    """Test orbital identification with a perfect abs(sin) wave."""

    x = np.arange(0, 3 * np.pi, 3 * np.pi / 100)
    distance = np.abs(np.sin(x))
    time = [datetime(2026, 1, 1) + timedelta(seconds=i) for i in range(len(x))]

    orbits = _identify_orbits(time, distance, minimal_distance=3, apply_smoothing=False)

    assert len(orbits) == 6

    assert orbits[0].start == 0
    assert orbits[0].end == 17
    assert orbits[0].direction == "outbound"

    assert orbits[1].start == 18
    assert orbits[1].end == 33
    assert orbits[1].direction == "inbound"

    assert orbits[2].start == 34
    assert orbits[2].end == 50
    assert orbits[2].direction == "outbound"

    assert orbits[3].start == 51
    assert orbits[3].end == 67
    assert orbits[3].direction == "inbound"

    assert orbits[4].start == 68
    assert orbits[4].end == 83
    assert orbits[4].direction == "outbound"

    assert orbits[5].start == 84
    assert orbits[5].end == 99
    assert orbits[5].direction == "inbound"


def test_identify_orbits_noisy_all_extrema() -> None:
    np.random.seed(42)
    x = np.arange(0, 3 * np.pi, 3 * np.pi / 100)
    distance = np.abs(np.sin(x)) + np.random.normal(0, 0.01, 100)
    time = [datetime(2026, 1, 1) + timedelta(seconds=i) for i in range(100)]

    orbits = _identify_orbits(time, distance, 20, apply_smoothing=True)

    assert len(orbits) == 6
