# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import time
from collections.abc import Iterable
from enum import Enum
from functools import partial
from multiprocessing import Pool

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from swvo.io.RBMDataSet import RBMDataSet


class TargetType(Enum):
    TargetPairs = 0
    TargetMeshGrid = 1


def _linear_interp(
    flux_left: float,
    flux_right: float,
    target_value: float,
    left_value: float,
    right_value: float,
) -> float:
    a = (target_value - left_value) / (right_value - left_value)
    return flux_left + a * (flux_right - flux_left)


def _interp_flux_parallel(
    flux: NDArray[np.float64],
    energy: NDArray[np.float64],
    alpha_eq_model: NDArray[np.float64],
    targets: list[tuple[float, float]],
    it: int,
) -> list[float]:
    result: list[float] = []

    for _, (target_en_single, target_al_single) in enumerate(targets):
        # find left and right alpha indices
        # first find the two al levels, where en points must exist

        al_right_idx = np.searchsorted(
            alpha_eq_model[it, :], target_al_single, side="right"
        )
        al_left_idx = al_right_idx - 1

        if al_right_idx == 0 or al_right_idx >= len(alpha_eq_model[it, :]):
            result.append(np.nan)
            continue

        finite_idx = np.argwhere(
            np.isfinite(energy[it, :]) & np.isfinite(flux[it, :, al_left_idx])
        )
        if finite_idx.size == 0:
            result.append(np.nan)
            continue

        energy_interp = np.squeeze(energy[it, finite_idx])
        flux_interp = np.squeeze(flux[it, finite_idx, al_left_idx])
        assert np.all(np.diff(energy_interp) > 0)

        flux_left = float(
            np.interp(
                target_en_single, energy_interp, flux_interp, left=np.nan, right=np.nan
            )
        )

        finite_idx = np.argwhere(
            np.isfinite(energy[it, :]) & np.isfinite(flux[it, :, al_right_idx])
        )
        if finite_idx.size == 0:
            result.append(np.nan)
            continue

        energy_interp = np.squeeze(energy[it, finite_idx])
        flux_interp = np.squeeze(flux[it, finite_idx, al_right_idx])
        assert np.all(np.diff(energy_interp) > 0)

        flux_right = float(
            np.interp(
                target_en_single, energy_interp, flux_interp, left=np.nan, right=np.nan
            )
        )

        result.append(
            _linear_interp(
                flux_left,
                flux_right,
                target_al_single,
                alpha_eq_model[it, al_left_idx],
                alpha_eq_model[it, al_right_idx],
            )
        )

    return result


def interp_flux(
    self: RBMDataSet,
    target_en: float | list[float] | NDArray[np.float64],
    target_al: float | list[float],
    target_type: TargetType,
    n_threads: int = 10,
) -> NDArray[np.float64]:
    if not isinstance(target_en, Iterable):
        target_en = [target_en]
    if not isinstance(target_al, Iterable):
        target_al = [target_al]

    if target_type == TargetType.TargetPairs:
        assert len(target_en) == len(
            target_al
        ), "For TargetType.Pairs, the target vectors must have the same size!"

        result_arr = np.empty((len(self.time), len(target_en)))
        targets = list(zip(target_en, target_al))
    else:
        result_arr = np.empty((len(self.time), len(target_en), len(target_al)))
        targets = list(itertools.product(target_en, target_al))

    func = partial(
        _interp_flux_parallel,
        self.Flux,
        self.energy_channels,
        self.alpha_eq_model,
        targets,
    )

    with Pool(n_threads) as p:
        rs = p.map_async(func, range(len(self.time)))

        # display progress bar if verbose
        if self._verbose:
            total_elements = rs._number_left
            with tqdm(total=total_elements) as t:
                while True:
                    if rs.ready():
                        break
                    t.n = total_elements - rs._number_left
                    t.refresh()
                    time.sleep(1)
        else:
            rs.wait()

    parallel_results = rs.get()

    if isinstance(parallel_results, Exception):
        raise parallel_results

    for i in range(result_arr.shape[0]):
        if target_type == TargetType.TargetPairs:
            for t, _ in enumerate(targets):
                result_arr[i, t] = parallel_results[i][t]
        else:
            for ie, ia in itertools.product(
                range(len(target_en)), range(len(target_al))
            ):
                result_arr[i, ie, ia] = parallel_results[i][ie * len(target_al) + ia]

    return result_arr


def _interp_psd_parallel(
    psd, mu_or_V, K, target_type, result_arr, target_1, target_K, it
):
    if target_type == TargetType.TargetPairs:
        targets = zip(target_1, target_K)
    else:
        targets = np.meshgrid(target_1, target_K)

        targets_1_grid = targets[0]
        targets_2_grid = targets[1]

    for i in range(targets_1_grid.shape[0]):
        for j in range(targets_1_grid.shape[1]):
            # find left and right alpha indices
            # first find the two al levels, where en points must exist

            K_right_idx = np.searchsorted(K[it, :], targets_2_grid[i, j], side="right")
            K_left_idx = K_right_idx - 1

            if K_right_idx == 0 or K_right_idx >= len(K[it, :]):
                continue

            psd_left = np.interp(
                targets_1_grid[i, j],
                mu_or_V[it, :, K_left_idx],
                psd[it, :, K_left_idx],
                left=np.nan,
                right=np.nan,
            )
            psd_right = np.interp(
                targets_1_grid[i, j],
                mu_or_V[it, :, K_right_idx],
                psd[it, :, K_right_idx],
                left=np.nan,
                right=np.nan,
            )

            result_arr[it, i] = _linear_interp(
                psd_left,
                psd_right,
                targets_2_grid[i, j],
                K[it, K_left_idx],
                K[it, K_right_idx],
            )

            # if target_type == TargetType.TargetPairs:
            #     result_arr[it,:] = result_tmp
            # else:
            #     result_arr[it,:,:] = result_tmp.reshape(len(target_en), len(target_al))


def interp_psd(
    self: RBMDataSet,
    target_K: list[float],
    target_type: TargetType,
    target_mu: list[float] = None,
    target_V: list[float] = None,
    n_threads: int=10,
) -> NDArray[np.float64]:
    if not isinstance(target_K, Iterable):
        target_K = [target_K]

    target_1 = None
    target_1_type = ""

    if target_mu is not None:
        if not isinstance(target_mu, Iterable):
            target_mu = [target_mu]

        target_1 = target_mu
        target_1_type = "mu"

    if target_V is not None:
        if not isinstance(target_V, Iterable):
            target_V = [target_V]

        if target_1_type == "mu":
            raise ValueError("You can only specify a mu OR V target and not both!")
        target_1_type = "V"
        target_1 = target_V

    assert (
        target_1_type != ""
    ), "Neither a mu nor a V target have been specified! Specify one of them."

    m = ParallelManager()
    m.start()

    if target_type == TargetType.TargetPairs:
        assert len(target_1) == len(
            target_K
        ), "For TargetType.Pairs, the target vectors must have the same size!"

        result_arr = m.np_empty((len(self.time), len(target_1)))
    else:
        result_arr = m.np_empty((len(self.time), len(target_1), len(target_K)))

    data_dim_1 = self.InvMu if target_1_type == "mu" else self.InvV

    func = partial(
        _interp_psd_parallel,
        self.PSD,
        data_dim_1,
        self.InvK,
        target_type,
        result_arr,
        target_1,
        target_K,
    )

    # display progress bar if verbose
    if self._verbose:
        process_map(func, range(len(self.time)), chunksize=len(self.time) // 10)
    else:
        with Pool(n_threads) as p:
            p.map(func, range(len(self.time)))
    return np.array(result_arr)
