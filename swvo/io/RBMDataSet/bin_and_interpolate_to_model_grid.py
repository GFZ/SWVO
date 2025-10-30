# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from icecream import ic
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm

if TYPE_CHECKING:
    from swvo.io.RBMDataSet import RBMDataSet, RBMDataSetElPaso


def bin_and_interpolate_to_model_grid(
    self: RBMDataSet | RBMDataSetElPaso,
    sim_time: list[datetime],
    grid_R: NDArray[np.float64],
    grid_mu_V: NDArray[np.float64],
    grid_K: NDArray[np.float64],
    grid_P: NDArray[np.float64] | None = None,
    debug_plot_settings: DebugPlotSettings | None = None,
    target_var_name: Literal["PSD", "density"] = "PSD",
    mu_or_V: Literal["Mu", "V"] = "V",
) -> NDArray[np.float64]:
    # make sure everything is 4D

    if grid_R.ndim == 3:
        grid_R = grid_R[np.newaxis, ...]
    if grid_mu_V.ndim == 3:
        grid_mu_V = grid_mu_V[np.newaxis, ...]
    if grid_K.ndim == 3:
        grid_K = grid_K[np.newaxis, ...]

    target_var_init = getattr(self, target_var_name)

    # 1. interpolate to V-K

    if grid_R.shape[2] > 1 and grid_R.shape[3] > 1:

        if target_var_init.ndim == 1:
            target_var_init = target_var_init[:, np.newaxis, np.newaxis]

        mu_or_V_arr = self.InvMu if mu_or_V == "Mu" else self.InvV
        if grid_mu_V.shape[2] > 1:
            psd_interp = _interpolate_in_V_K(target_var_init, mu_or_V_arr, self.InvK, grid_mu_V, grid_K)
        else:
            psd_interp = target_var_init

        # sanity check
        if np.min(target_var_init) > np.min(psd_interp) or np.max(target_var_init) < np.max(psd_interp):
            msg = "Found inconsitency in V-K interpolation. Aborting..."
            raise (ValueError(msg))
    else:

        if target_var_init.ndim == 1: # plasmasphere
            target_var_init = target_var_init[:, np.newaxis, np.newaxis]

        psd_interp = target_var_init

    # 2. Bin in space

    R_or_Lstar_arr = self.R0 if grid_P is not None else self.Lstar[:, -1]

    psd_binned_in_space = _bin_in_space(psd_interp, self.P, R_or_Lstar_arr, grid_R, grid_P)
    # sanity check
    if np.min(target_var_init) > np.min(psd_binned_in_space) or np.max(target_var_init) < np.max(psd_binned_in_space):
        msg = "Found inconsitency in space binning. Aborting..."
        raise (ValueError(msg))

    # 3. Bin in time
    psd_binned_in_time = _bin_in_time(self.datetime, sim_time, psd_binned_in_space)
    # sanity check
    if np.min(target_var_init) > np.min(psd_binned_in_time) or np.max(target_var_init) < np.max(psd_binned_in_time):
        msg = "Found inconsitency in time binning. Aborting..."
        raise (ValueError(msg))

    if debug_plot_settings:
        if debug_plot_settings.target_K is not None:
            plot_debug_figures(
                self,
                psd_binned_in_time,
                sim_time,
                grid_P,
                grid_R,
                grid_mu_V,
                grid_K,
                mu_or_V,
                debug_plot_settings,
            )
        else: plot_debug_figures_plasmasphere(
            self,
            psd_binned_in_time,
            sim_time,
            grid_P,
            grid_R,
            debug_plot_settings,
        )

    return psd_binned_in_time


def _linear_interp(
    PSD_left: float,
    PSD_right: float,
    target_value: float,
    left_value: float,
    right_value: float,
) -> float:
    a = (target_value - left_value) / (right_value - left_value)
    return PSD_left + a * (PSD_right - PSD_left)


def _get_time_bins(timestamps: list[float]) -> list[float]:
    dt = timestamps[1] - timestamps[0]

    bins = [timestamps[0] - dt / 2]
    for i in range(len(timestamps)):
        bins.append(bins[i] + dt)

    return bins


def _get_time_indices(data_timestamps: list[float], time_bins: list[float]) -> NDArray[np.float32]:
    time_indices = np.digitize(data_timestamps, time_bins)
    time_indices = time_indices - 1
    time_indices = np.where(time_indices == len(time_bins) - 1, -1, time_indices)

    return time_indices


def _bin_in_time(
    data_time: NDArray[np.object_],
    sim_time: NDArray[np.object_],
    data_psd: NDArray[np.float64],
) -> NDArray[np.float64]:
    psd_binned = np.full(
        (
            len(sim_time),
            data_psd.shape[1],
            data_psd.shape[2],
            data_psd.shape[3],
            data_psd.shape[4],
        ),
        np.nan,
    )
    sim_timestamps = [t.timestamp() for t in sim_time]
    data_timestamps = [t.timestamp() for t in data_time]
    time_indices = _get_time_indices(data_timestamps, _get_time_bins(sim_timestamps))

    for i, _ in tqdm(enumerate(sim_time)):
        psd_binned[i, ...] = np.power(10, np.nanmean(np.log10(data_psd[time_indices == i, ...]), axis=0))

    return psd_binned


def _bin_in_space(
    psd_in: NDArray[np.float64],
    P_data: NDArray[np.float64],
    R_data: NDArray[np.float64],
    grid_R: NDArray[np.float64],
    grid_P: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    print("\tBin in space...")

    if grid_P is not None:
        grid_P_1d = grid_P[:, 0, 0, 0]
        grid_R_1d = grid_R[0, :, 0, 0]

        psd_binned = np.full(
            (
                psd_in.shape[0],
                grid_P.shape[0],
                grid_P.shape[1],
                psd_in.shape[1],
                psd_in.shape[2],
            ),
            0.0,
        )
        number_of_observations = np.full(
            (
                psd_in.shape[0],
                grid_P.shape[0],
                grid_P.shape[1],
                psd_in.shape[1],
                psd_in.shape[2],
            ),
            0,
        )

    else:
        grid_P_1d = None
        grid_R_1d = grid_R[0, :, 0, 0]

        psd_binned = np.full((psd_in.shape[0], 1, grid_R.shape[1], psd_in.shape[1], psd_in.shape[2]), 0.0)
        number_of_observations = np.full((psd_in.shape[0], 1, grid_R.shape[1], psd_in.shape[1], psd_in.shape[2]), 0)

    for it in range(psd_in.shape[0]):
        if np.all(np.isnan(psd_in[it, :, :])):
            continue

        # find correct P-R-cell
        dR = grid_R_1d[1] - grid_R_1d[0]
        if R_data[it] - dR / 2 < grid_R_1d[0] or R_data[it] + dR / 2 > grid_R_1d[-1]:
            # out of bounds
            continue

        r_idx = np.argmin(np.abs(R_data[it] - grid_R_1d))

        if grid_P_1d is not None:
            raw_difference_p = np.abs(P_data[it] - grid_P_1d)
            min_difference_p = np.where(
                raw_difference_p <= np.pi,
                raw_difference_p,
                2 * np.pi - raw_difference_p,
            )
            p_idx = np.argmin(min_difference_p)

            number_of_observations[it, p_idx, r_idx, :, :] += np.where(np.isnan(psd_in[it, :, :]), 0, 1)
            psd_binned[it, p_idx, r_idx, :, :] += np.where(np.isnan(psd_in[it, :, :]), 0, np.log10(psd_in[it, :, :]))

        else:
            number_of_observations[it, 0, r_idx, :, :] += np.where(np.isnan(psd_in[it, :, :]), 0, 1)
            psd_binned[it, 0, r_idx, :, :] += np.where(np.isnan(psd_in[it, :, :]), 0, np.log10(psd_in[it, :, :]))

        # # ic(number_of_observations[it, :, :, 0, 0])
        # ic(np.power(10, np.nanmax(psd_binned[it, :, :, 0, 0])))
        # ic(np.power(10, np.nanmax(psd_binned[it, :, :, 0, 0] / number_of_observations[it, :, :, 0, 0])))

    psd_binned = np.where(psd_binned == 0, np.nan, psd_binned)

    return np.power(10, psd_binned / number_of_observations)


def _interpolate_in_V_K(
    psd_in: NDArray[np.float64],
    V_data: NDArray[np.float64],
    K_data: NDArray[np.float64],
    grid_V: NDArray[np.float64],
    grid_K: NDArray[np.float64],
) -> NDArray[np.float64]:
    print("\tInterpolate in V and K...")

    grid_K_1d = grid_K[0, 0, 0, :]

    func = partial(_parallel_func_VK, grid_K_1d, grid_V, K_data, V_data, psd_in)

    with Pool(12) as p:
        rs = p.map_async(func, range(psd_in.shape[0]))

        # display progress bar if verbose
        total_elements = rs._number_left
        with tqdm(total=total_elements) as t:
            while True:
                if rs.ready():
                    break
                t.n = total_elements - rs._number_left
                t.refresh()
                time.sleep(1)

    result = rs.get()
    if isinstance(result, Exception):
        raise result

    return np.asarray(result)


def _parallel_func_VK(
    grid_K_1d: NDArray[np.float64],
    grid_V: NDArray[np.float64],
    K_data: NDArray[np.float64],
    V_data: NDArray[np.float64],
    psd_in: NDArray[np.float64],
    it: int,
) -> NDArray[np.float64]:
    psd_interp = np.full((grid_V.shape[2], grid_V.shape[3]), np.nan)

    for iK, K_val in enumerate(grid_K_1d):
        grid_V_1d = grid_V[0, 0, :, iK]
        for iV, V_val in enumerate(grid_V_1d):
            K_finite = np.isfinite(K_data[it, :])
            K_sorted = 1 if np.all(np.diff(K_data[it, K_finite]) >= 0) else -1

            if np.all(K_data[it, :] == np.nan):
                continue

            if np.all(psd_in[it, :, :] == np.nan):
                continue

            # search for sourrounding 4 corners
            # take negative values, as K_data is in descending order

            K_idx_left = np.searchsorted(K_sorted * K_data[it, :], K_sorted * K_val, side="right") - 1
            K_idx_right = K_idx_left + 1

            if K_idx_left == -1 or K_idx_right >= K_data.shape[1]:
                # out of bounds
                continue

            V_finite = np.isfinite(V_data[it, :, K_idx_left])
            V_sorted = 1 if np.all(np.diff(V_data[it, V_finite, K_idx_left]) >= 0) else -1

            V_idx_left_left = np.searchsorted(V_sorted * V_data[it, :, K_idx_left], V_sorted * V_val, side="right") - 1
            V_idx_left_right = V_idx_left_left + 1

            if V_idx_left_left == -1 or V_idx_left_right >= V_data.shape[1]:
                # out of bounds
                continue

            V_sorted = 1 if np.all(np.diff(V_data[it, :, K_idx_right]) >= 0) else -1

            V_idx_right_left = (
                np.searchsorted(
                    V_sorted * V_data[it, :, K_idx_right],
                    V_sorted * V_val,
                    side="right",
                )
                - 1
            )
            V_idx_right_right = V_idx_right_left + 1

            if V_idx_right_left == -1 or V_idx_right_right >= V_data.shape[1]:
                # out of bounds
                continue

            PSD_left = np.power(
                10,
                _linear_interp(
                    np.log10(psd_in[it, V_idx_left_left, K_idx_left]),
                    np.log10(psd_in[it, V_idx_left_right, K_idx_left]),
                    np.log10(V_val),
                    np.log10(V_data[it, V_idx_left_left, K_idx_left]),
                    np.log10(V_data[it, V_idx_left_right, K_idx_left]),
                ),
            )

            PSD_right = np.power(
                10,
                _linear_interp(
                    np.log10(psd_in[it, V_idx_right_left, K_idx_right]),
                    np.log10(psd_in[it, V_idx_right_right, K_idx_right]),
                    np.log10(V_val),
                    np.log10(V_data[it, V_idx_right_left, K_idx_right]),
                    np.log10(V_data[it, V_idx_right_right, K_idx_right]),
                ),
            )

            psd_interp[iV, iK] = np.power(
                10,
                _linear_interp(
                    np.log10(PSD_left),
                    np.log10(PSD_right),
                    np.log10(K_val),
                    np.log10(K_data[it, K_idx_left]),
                    np.log10(K_data[it, K_idx_right]),
                ),
            )

    return psd_interp


@dataclass
class DebugPlotSettings:
    folder_path: Path
    satellite_name: str
    target_V: float | None = None
    target_K: float | None = None


def plot_debug_figures_plasmasphere(
    data_set: RBMDataSet,
    psd_binned: NDArray[np.float64],
    sim_time: NDArray[np.object_],
    grid_P: NDArray[np.float64] | None,
    grid_R: NDArray[np.float64],
    debug_plot_settings: DebugPlotSettings,
):

    print("\tPlot debug features...")

    from icecream import ic

    dt = sim_time[1] - sim_time[0]

    fig = plt.figure(figsize=(19.20, 8))
    plt.rcParams["axes.axisbelow"] = False

    R_or_Lstar_arr = data_set.R0

    for it, sim_time_curr in enumerate(tqdm(sim_time)):
        sat_time_idx = np.argwhere(np.abs(np.asarray(data_set.datetime) - sim_time_curr) <= dt / 2)

        R_idx = np.argwhere(np.abs(grid_R[0, :, 0, 0] - R_or_Lstar_arr[sat_time_idx]))


        ax0 = fig.add_subplot(121, projection="polar")
        ax1 = fig.add_subplot(122)

        # plot satellite trajectory on PxR grid
        # [x_sat, y_sat] = pol2cart(self.P, self.R)

        # ic(data_set.P[sat_time_idx])
        # ic(R_or_Lstar_arr[sat_time_idx])

        ax0.scatter(data_set.P[sat_time_idx], R_or_Lstar_arr[sat_time_idx], c=np.log10(data_set.density[sat_time_idx]), marker="D", vmin=0,
            vmax=4,
            cmap="jet",)
        ax0.set_ylim(1, 6.6)
        ax0.set_title("Orbit")
        ax0.set_rlim([0, 6.6])
        ax0.set_theta_offset(np.pi)

        grid_X = grid_R[:, :, 0, 0] * np.cos(grid_P[:, :, 0, 0])
        grid_Y = grid_R[:, :, 0, 0] * np.sin(grid_P[:, :, 0, 0])

        pc = ax1.pcolormesh(
            grid_X,
            grid_Y,
            np.squeeze(np.log10(psd_binned[it, :, :, :, :])),
            vmin=0,
            vmax=4,
            cmap="jet",
            edgecolors="k",
            linewidth=0.1,
        )
        ax1.set_title("Assimilation input")
        ax1.set_xlim(np.max(grid_R), -np.max(grid_R))
        ax1.set_ylim(np.max(grid_R), -np.max(grid_R))
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")

        fig.colorbar(pc, ax=ax1)

        fig.savefig(Path(debug_plot_settings.folder_path) / f"{debug_plot_settings.satellite_name}_{sim_time_curr}.png")

        # ic(np.log10(psd_binned[it,:,:,V_idx,K_idx]))

        fig.clf()

        if np.any(data_set.P[sat_time_idx] < 0.1):
            ic(psd_binned[it, 0, :, :, :])
            asdf



def plot_debug_figures(
    data_set: RBMDataSet,
    psd_binned: NDArray[np.float64],
    sim_time: NDArray[np.object_],
    grid_P: NDArray[np.float64] | None,
    grid_R: NDArray[np.float64],
    grid_V: NDArray[np.float64],
    grid_K: NDArray[np.float64],
    mu_or_V: Literal["Mu", "V"],
    debug_plot_settings: DebugPlotSettings,
):
    print("\tPlot debug features...")

    dt = sim_time[1] - sim_time[0]

    fig = plt.figure(figsize=(19.20, 5))
    plt.rcParams["axes.axisbelow"] = False

    data_set_V_or_Mu = data_set.InvMu if mu_or_V == "Mu" else data_set.InvV

    R_or_Lstar_arr = data_set.R0 if grid_P is not None else data_set.Lstar[:, -1]

    for it, sim_time_curr in enumerate(tqdm(sim_time)):
        sat_time_idx = np.argwhere(np.abs(np.asarray(data_set.datetime) - sim_time_curr) <= dt / 2)

        R_idx = np.argwhere(np.abs(grid_R[0, :, 0, 0] - R_or_Lstar_arr[sat_time_idx]))

        K_idx = np.argmin(np.abs(grid_K[0, R_idx, 0, :] - debug_plot_settings.target_K))
        V_idx = np.argmin(np.abs(grid_V[0, R_idx, :, K_idx] - debug_plot_settings.target_V))

        V_lim_min = np.log10(0.9 * np.min([np.nanmin(data_set_V_or_Mu), np.min(grid_V)]))
        V_lim_max = np.log10(1.1 * np.max([np.nanmax(data_set_V_or_Mu), np.max(grid_V)]))

        K_lim_min = np.log10(0.9 * np.min([np.nanmin(data_set.InvK), np.min(grid_K)]))
        K_lim_max = np.log10(1.1 * np.max([np.nanmax(data_set.InvK), np.max(grid_K)]))

        ax0 = fig.add_subplot(131, projection="polar")
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)

        # plot satellite trajectory on PxR grid
        # [x_sat, y_sat] = pol2cart(self.P, self.R)

        ax0.scatter(data_set.P[sat_time_idx], R_or_Lstar_arr[sat_time_idx], c="k", marker="D")
        ax0.set_ylim(1, 6.6)
        ax0.set_title("Orbit")
        ax0.set_theta_offset(np.pi)

        ax1.vlines(
            [np.log10(np.min(grid_V)), np.log10(np.max(grid_V))],
            np.log10(np.min(grid_K)),
            np.log10(np.max(grid_K)),
        )
        ax1.hlines(
            [np.log10(np.min(grid_K)), np.log10(np.max(grid_K))],
            np.log10(np.min(grid_V)),
            np.log10(np.max(grid_V)),
        )
        ax1.scatter(
            np.log10(grid_V[0, R_idx, :, :]),
            np.log10(grid_K[0, R_idx, :, :]),
            c="b",
            s=10,
        )

        for iV in range(data_set_V_or_Mu.shape[1]):
            sc = ax1.scatter(
                np.log10(data_set_V_or_Mu[sat_time_idx, iV, :]),
                np.log10(data_set.InvK[sat_time_idx, :]),
                c=np.log10(data_set.PSD[sat_time_idx, iV, :]),
                marker="D",
                vmin=-1,
                vmax=3,
                cmap="jet",
            )

        # sc = ax1.scatter(np.log10(data_set_V_or_Mu[sat_time_idx,0,:]), np.log10(data_set.InvK[sat_time_idx,:]),
        #                     c=np.log10(data_set.PSD[sat_time_idx,0,:]), marker="D", vmin=-1, vmax=3, cmap="jet")
        # sc = ax1.scatter(np.log10(data_set_V_or_Mu[sat_time_idx,-1,:]), np.log10(data_set.InvK[sat_time_idx,:]),
        #                     c=np.log10(data_set.PSD[sat_time_idx,-1,:]), marker="D", vmin=-1, vmax=3, cmap="jet")

        ax1.scatter(
            np.log10(grid_V[0, R_idx, V_idx, K_idx]),
            np.log10(grid_K[0, R_idx, V_idx, K_idx]),
            c="r",
            s=15,
            marker="x",
        )
        ax1.set_title("V-K of satellite and simulation grid")
        ax1.set_xlim(V_lim_min, V_lim_max)
        ax1.set_ylim(K_lim_min, K_lim_max)
        ax1.set_xlabel("log10 V")
        ax1.set_ylabel("log10 K")

        fig.colorbar(sc, ax=ax1)

        if grid_P:
            grid_X = grid_R[:, :, 0, 0] * np.cos(grid_P[:, :, 0, 0])
            grid_Y = grid_R[:, :, 0, 0] * np.sin(grid_P[:, :, 0, 0])

            pc = ax2.pcolormesh(
                grid_X,
                grid_Y,
                np.any(np.isfinite(psd_binned[it, :, :, :, :]), axis=(2, 3)),
                vmin=-1,
                vmax=5,
                cmap="jet",
                edgecolors="k",
                linewidth=0.1,
            )
            ax2.set_title("Assimilation input")
            ax2.set_xlim(np.max(grid_R), -np.max(grid_R))
            ax2.set_ylim(np.max(grid_R), -np.max(grid_R))
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")

            fig.colorbar(pc, ax=ax2)
        else:
            grid_X, grid_Y = np.meshgrid(sim_time, grid_R[0, :, 0, 0])
            print(np.log10(psd_binned[:, 0, :, V_idx, K_idx]))
            pc = ax2.pcolormesh(
                grid_X,
                grid_Y,
                np.log10(psd_binned[:, 0, :, V_idx, K_idx]).T,
                vmin=-1,
                vmax=5,
                cmap="jet",
                edgecolors=None,
                linewidth=0.1,
                shading="nearest",
            )
            ax2.set_title("Assimilation input")
            ax2.set_ylim(0, np.max(grid_R))
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Lstar")

            fig.colorbar(pc, ax=ax2)

        fig.savefig(Path(debug_plot_settings.folder_path) / f"{debug_plot_settings.satellite_name}_{sim_time_curr}.png")

        # ic(np.log10(psd_binned[it,:,:,V_idx,K_idx]))

        fig.clf()
