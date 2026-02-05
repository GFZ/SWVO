from __future__ import annotations

import typing
from datetime import datetime
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import make_splrep  # type: ignore[reportUnknownVariableType]
from scipy.signal import find_peaks  # type: ignore[reportUnknownVariableType]

if typing.TYPE_CHECKING:
    from swvo.io.RBMDataSet import RBMDataSet, RBMNcDataSet


class Trajectory(NamedTuple):
    start: int
    end: int
    direction: Literal["inbound", "outbound"]

def _identify_orbits(
        time: list[datetime],
        distance: NDArray[np.floating],
        minimal_distance: int,
        *,
        apply_smoothing: bool) -> list[Trajectory]:

    distance_filled = pd.Series(distance).interpolate(method="linear", limit_direction="both").to_numpy()

    if apply_smoothing:
        timestamps = [t.timestamp() for t in time]
        distance_filled = make_splrep(timestamps, distance_filled, s=0)(timestamps)  # type: ignore[reportUnknownVariableType]
        distance_filled = typing.cast("NDArray[np.floating]", distance_filled)

    peaks, _ = find_peaks(distance_filled, distance=minimal_distance)  # type: ignore[reportUnknownVariableType]
    troughs, _ = find_peaks(-distance_filled, distance=minimal_distance)  # type: ignore[reportUnknownVariableType]
    extrema = np.sort(np.concatenate((peaks, troughs)))  # type: ignore[reportUnknownVariableType]
    extrema = typing.cast("NDArray[np.int32]", extrema)

    diffs = np.diff(distance_filled)
    in_out_bound_label = "inbound" if np.median(diffs[0:extrema[0]]) < 0 else "outbound"
    orbits: list[Trajectory] = [Trajectory(0, int(extrema[0]), in_out_bound_label)]

    for i in range(1, len(extrema)):

        # print(diffs[extrema[i - 1]:extrema[i]])
        in_out_bound_label = "inbound" if np.median(diffs[extrema[i - 1]:extrema[i]]) < 0 else "outbound"

        orbits.append(
            Trajectory(extrema[i - 1] + 1, extrema[i], in_out_bound_label)
        )

    in_out_bound_label = "inbound" if np.median(diffs[extrema[-1]:]) < 0 else "outbound"
    orbits.append(Trajectory(extrema[-1] + 1, len(distance) - 1, in_out_bound_label))

    return orbits

def identify_orbits(
        self: RBMDataSet | RBMNcDataSet,
        orbit_type: Literal["R", "L*"] = "R",
        minimal_distance: int = 10,
        *,
        apply_smoothing: bool = True) -> list[Trajectory]:

    dist = self.R0 if orbit_type == "R" else self.Lstar[:, -1]

    return _identify_orbits(self.datetime, dist, minimal_distance, apply_smoothing=apply_smoothing)