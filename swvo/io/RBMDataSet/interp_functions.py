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
from typing import Literal

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
    target_type: TargetType|Literal["TargetPairs", "TargetMesh"],
    n_threads: int = 10,
) -> NDArray[np.float64]:
    if not isinstance(target_en, Iterable):
        target_en = [target_en]
    if not isinstance(target_al, Iterable):
        target_al = [target_al]

    if isinstance(target_type, str):
        target_type = TargetType[target_type]

    if target_type == TargetType.TargetPairs:
        assert len(target_en) == len(  # ty:ignore[invalid-argument-type]
            target_al  # ty:ignore[invalid-argument-type]
        ), "For TargetType.Pairs, the target vectors must have the same size!"

        result_arr = np.empty((len(self.time), len(target_en)))  # ty:ignore[invalid-argument-type]
        targets = list(zip(target_en, target_al))
    else:
        result_arr = np.empty((len(self.time), len(target_en), len(target_al)))  # ty:ignore[invalid-argument-type]
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
            total_elements = rs._number_left  # ty:ignore[unresolved-attribute]
            with tqdm(total=total_elements) as t:
                while True:
                    if rs.ready():
                        break
                    t.n = total_elements - rs._number_left  # ty:ignore[unresolved-attribute]
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
                range(len(target_en)), range(len(target_al))  # ty:ignore[invalid-argument-type]
            ):
                result_arr[i, ie, ia] = parallel_results[i][ie * len(target_al) + ia]  # ty:ignore[invalid-argument-type]

    return result_arr


def _interp_psd_parallel(psd: NDArray[np.float64],
                         invmu: NDArray[np.float64],
                         invk: NDArray[np.float64],
                         targets: list[tuple[float, float]],
                         it: int) -> list[float]:
    """
    Interpolate PSD at time index `it` to (mu_target, K_target) pairs in `targets`.

    Shapes per time slice:
      psd[it]   -> (nE, nA)
      invmu[it] -> (nE, nA)
      invk[it]  -> (nA,)
    """
    out: list[float] = []

    # ---- 0) Extract this time slice
    psd_i  = psd[it, :, :]      # (nE, nA)
    mu_i   = invmu[it, :, :]    # (nE, nA)
    K_row  = invk[it, :]        # (nA,)

    # ---- 1) Drop NaN K bins and the corresponding columns in PSD/mu
    finite_k = np.isfinite(K_row)
    if not np.any(finite_k):
        # No valid K at this time -> all NaN
        return [np.nan] * len(targets)

    K_use   = K_row[finite_k]           # (nA_valid,)
    psd_use = psd_i[:, finite_k]        # (nE, nA_valid)
    mu_use  = mu_i[:,  finite_k]        # (nE, nA_valid)

    # If after masking we have fewer than 2 K points, we cannot bracket
    if K_use.size < 2:
        return [np.nan] * len(targets)

    # ---- 2) Ensure K ascending for searchsorted; if descending, flip columns
    if K_use[1] < K_use[0]:
        K_use   = K_use[::-1]
        psd_use = psd_use[:, ::-1]
        mu_use  = mu_use[:,  ::-1]

    # ---- 3) For each (mu*, K*) target: 1D along mu, then linear across K
    for _, (mu_t, K_t) in enumerate(targets):

        # 3a) Bracket in K
        k_right = np.searchsorted(K_use, K_t, side='right')
        k_left  = k_right - 1
        if k_right == 0 or k_right >= K_use.size:
            out.append(np.nan)
            continue

        # 3b) Interp along mu at LEFT K
        mu_L  = mu_use[:,  k_left]
        psd_L = psd_use[:, k_left]
        okL   = np.isfinite(mu_L) & np.isfinite(psd_L)
        if not np.any(okL):
            out.append(np.nan); continue

        xL = np.asarray(mu_L[okL],  dtype=float)
        yL = np.asarray(psd_L[okL], dtype=float)
        if xL.size < 2:
            out.append(np.nan); continue
        if not np.all(np.diff(xL) > 0):
            order = np.argsort(xL)
            xL, yL = xL[order], yL[order]
            xL, idx = np.unique(xL, return_index=True)
            yL = yL[idx]
            if xL.size < 2:
                out.append(np.nan); continue

        psd_left = float(np.interp(mu_t, xL, yL, left=np.nan, right=np.nan))

        # 3c) Interp along mu at RIGHT K
        mu_R  = mu_use[:,  k_right]
        psd_R = psd_use[:, k_right]
        okR   = np.isfinite(mu_R) & np.isfinite(psd_R)
        if not np.any(okR):
            out.append(np.nan); continue

        xR = np.asarray(mu_R[okR],  dtype=float)
        yR = np.asarray(psd_R[okR], dtype=float)
        if xR.size < 2:
            out.append(np.nan); continue
        if not np.all(np.diff(xR) > 0):
            order = np.argsort(xR)
            xR, yR = xR[order], yR[order]
            xR, idx = np.unique(xR, return_index=True)
            yR = yR[idx]
            if xR.size < 2:
                out.append(np.nan); continue

        psd_right = float(np.interp(mu_t, xR, yR, left=np.nan, right=np.nan))

        if not np.isfinite(psd_left) or not np.isfinite(psd_right):
            out.append(np.nan); continue

        # 3d) Linear across K to K_t
        val = _linear_interp(psd_left, psd_right, K_t, K_use[k_left], K_use[k_right])
        out.append(val)

    return out


def interp_psd(self: RBMDataSet,
               target_mu: float | list[float] | NDArray[np.float64],
               target_K:  float | list[float] | NDArray[np.float64],
               target_type: TargetType|Literal["TargetPairs", "TargetMesh"],
               n_threads: int = 10) -> NDArray[np.float64]:
    """
    Interpolate PSD to requested (mu, K) targets for every time.

    Output shapes (matching interp_flux semantics):
      - TargetPairs     -> (time, N)
      - TargetMeshGrid  -> (time, n_mu, n_K)
    """

    if not isinstance(target_mu, Iterable):
        target_mu = [target_mu]
    if not isinstance(target_K, Iterable):
        target_K = [target_K]

    if isinstance(target_type, str):
        target_type = TargetType[target_type]

    if target_type == TargetType.TargetPairs:
        assert len(target_mu) == len(target_K), \
            "For TargetType.Pairs, mu and K vectors must have the same size!"  # ty:ignore[invalid-argument-type]
        result_arr = np.empty((len(self.time), len(target_mu)))  # ty:ignore[invalid-argument-type]
        targets = list(zip(target_mu, target_K))
    else:
        result_arr = np.empty((len(self.time), len(target_mu), len(target_K)))  # ty:ignore[invalid-argument-type]
        targets = list(itertools.product(target_mu, target_K))

    # ensure needed fields are loaded (triggers lazy loader if any)
    _ = self.PSD; _ = self.InvMu; _ = self.InvK

    # parallel over time (same pattern as interp_flux)
    func = partial(_interp_psd_parallel, self.PSD, self.InvMu, self.InvK, targets)

    with Pool(n_threads) as p:
        rs = p.map_async(func, range(len(self.time)))

        if self._verbose:
            total_elements = rs._number_left  # ty:ignore[unresolved-attribute]
            with tqdm(total=total_elements) as t:
                while True:
                    if rs.ready(): break
                    t.n = (total_elements - rs._number_left)  # ty:ignore[unresolved-attribute]
                    t.refresh()
                    time.sleep(1)
        else:
            rs.wait()

    parallel_results = rs.get()
    if isinstance(parallel_results, Exception):
        raise parallel_results

    # pack results back like interp_flux
    if target_type == TargetType.TargetPairs:
        for i in range(result_arr.shape[0]):
            for t, _ in enumerate(targets):
                result_arr[i, t] = parallel_results[i][t]
    else:
        n_mu, n_K = len(target_mu), len(target_K)  # ty:ignore[invalid-argument-type]
        for i in range(result_arr.shape[0]):
            for im, iK in itertools.product(range(n_mu), range(n_K)):
                result_arr[i, im, iK] = parallel_results[i][im * n_K + iK]

    return result_arr
