from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from swvo.io.RBMDataSet import RBMDataSet, RBMNcDataSet
    from swvo.io.RBMDataSet.identify_orbits import Trajectory


def _linearize_trajectories(
    time: list[datetime],
    distance: np.ndarray,
    trajectories: list[Trajectory],
) -> tuple[NDArray[np.floating], list[datetime]]:
    dist_filled = pd.Series(distance).interpolate(method="linear", limit_direction="both").to_numpy()

    n = len(distance)
    lin_x_axis = np.full(n, np.nan)
    bend_time_axis = np.full(n, np.nan)

    max_r_global = np.nanmax(distance)
    min_r_global = np.nanmin(distance)

    # Convert datetime to timestamps for interpolation
    time_ts = np.array([t.timestamp() for t in time])

    for it, traj in enumerate(trajectories):
        idx = slice(traj.start, traj.end + 1)
        traj_r = dist_filled[idx]

        if len(traj_r) == 0:
            continue

        # Local Min/Max
        max_r_traj = np.max(traj_r) if (it != 0 and it != len(trajectories) - 1) else max_r_global
        min_r_traj = np.min(traj_r) if (it != 0 and it != len(trajectories) - 1) else min_r_global

        r_range = max_r_traj - min_r_traj if max_r_traj != min_r_traj else 1.0

        # Calculate offset
        if it > 0:
            diff_end_last = lin_x_axis[traj.start - 1] - lin_x_axis[traj.start - 2]
            start_offset = lin_x_axis[traj.start - 1]
        else:
            diff_end_last = 0
            start_offset = 0

        # Create the linearized X mapping
        if traj.direction == "inbound":
            lin_x_axis[idx] = start_offset + diff_end_last + (max_r_traj - traj_r) / r_range / 2
        else:
            lin_x_axis[idx] = start_offset + diff_end_last + (traj_r - min_r_traj) / r_range / 2

        # Bend time axis (interpolation)
        if len(traj_r) > 1:
            interp_query = np.linspace(it / 2, (it + 1) / 2, len(traj_r))
            f = interp1d(interp_query, time_ts[idx], fill_value="extrapolate")
            bend_time_axis[idx] = f(lin_x_axis[idx])
        else:
            bend_time_axis[idx] = time_ts[traj.start]

    # Clean up the end of the axis
    if n > 1:
        last_diff = lin_x_axis[-2] - lin_x_axis[-3] if n > 2 else 0
        lin_x_axis[-1] = lin_x_axis[-2] + last_diff

    return lin_x_axis, [datetime.fromtimestamp(ts) for ts in bend_time_axis]


def linearize_trajectories(
    self: RBMDataSet | RBMNcDataSet,
    trajectories: list[Trajectory],
    orbit_type: Literal["R", "L*"] = "R",
) -> tuple[NDArray[np.floating], list[datetime]]:
    dist = self.R0 if orbit_type == "R" else self.Lstar[:, -1]

    return _linearize_trajectories(self.datetime, dist, trajectories)
