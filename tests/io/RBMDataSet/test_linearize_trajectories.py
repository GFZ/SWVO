from datetime import datetime, timedelta

import numpy as np
import pytest

from swvo.io.RBMDataSet.identify_orbits import Trajectory, _identify_orbits
from swvo.io.RBMDataSet.linearize_trajectories import _linearize_trajectories


def test_linearize_trajectories_monotonicity():
    """Verify that the resulting x-axis is strictly increasing even with inbound segments."""
    distance = np.array([0.0, 5.0, 10.0, 7.0, 3.0, 0.0])
    time = [datetime(2026, 1, 1) + timedelta(minutes=i) for i in range(len(distance))]

    trajectories = _identify_orbits(time, distance, 1, apply_smoothing=False)

    lin_x, _ = _linearize_trajectories(time, distance, trajectories)

    diffs = np.diff(lin_x)
    assert np.all(diffs >= 0), f"X-axis is not monotonic: {lin_x}"
    assert not np.isnan(lin_x).any(), "Result contains NaNs"


def test_linearize_trajectories_abs_sin():
    """Test with a more complex rectified sine wave."""
    x = np.arange(0, 3 * np.pi, 3 * np.pi / 100)
    distance = np.abs(np.sin(x))
    time = [datetime(2026, 1, 1) + timedelta(seconds=i) for i in range(len(distance))]

    trajectories = _identify_orbits(time, distance, 1, apply_smoothing=False)

    lin_x, _ = _linearize_trajectories(time, distance, trajectories)

    assert len(lin_x) == len(distance)
    assert lin_x[0] == 0
    assert lin_x[-1] > lin_x[9]


def test_linearize_trajectories_with_nans():
    """Ensure the interpolation logic handles NaNs in the input distance."""
    distance = np.array([1.0, np.nan, 3.0, 4.0])
    time = [datetime(2026, 1, 1) + timedelta(seconds=i) for i in range(4)]

    trajectories = [Trajectory(0, 3, "outbound")]

    lin_x, _ = _linearize_trajectories(time, distance, trajectories)

    assert not np.any(np.isnan(lin_x))
    assert lin_x[1] > lin_x[0]


def test_linearize_trajectories_single_point():
    """Edge case: ensure it handles very short trajectories without crashing."""
    distance = np.array([1.0])
    time = [datetime(2026, 1, 1)]
    trajectories = [Trajectory(0, 0, "outbound")]

    lin_x, _ = _linearize_trajectories(time, distance, trajectories)

    assert len(lin_x) == 1
    assert not np.any(np.isnan(lin_x))


@pytest.mark.visual
def test_linearize_trajectories_visual() -> None:
    x = np.arange(0, 3 * np.pi, 3 * np.pi / 200)
    distance = np.abs(np.sin(x))
    time = [datetime(2026, 1, 1) + timedelta(seconds=i) for i in range(len(distance))]

    trajectories = _identify_orbits(time, distance, 1, apply_smoothing=False)

    lin_x, _ = _linearize_trajectories(time, distance, trajectories)

    from matplotlib import pyplot as plt

    f, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(time, distance)
    axs[1].plot(lin_x, distance)
    f.savefig("linearize_trajectorieis_test.png")


@pytest.mark.visual
def test_linearize_trajectories_visual_with_noise() -> None:
    np.random.seed(42)
    x = np.arange(0, 3 * np.pi, 3 * np.pi / 100)
    distance = np.abs(np.sin(x)) + np.random.normal(0, 0.01, 100)
    time = [datetime(2026, 1, 1) + timedelta(seconds=i) for i in range(len(distance))]

    trajectories = _identify_orbits(time, distance, 20, apply_smoothing=False)

    lin_x, _ = _linearize_trajectories(time, distance, trajectories)

    from matplotlib import pyplot as plt

    f, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(time, distance)
    axs[1].plot(lin_x, distance)
    f.savefig("linearize_trajectorieis_with_noise_test.png")
