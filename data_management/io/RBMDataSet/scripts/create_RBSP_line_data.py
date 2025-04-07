from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from numpy.typing import NDArray

from RBMDataSet import (
    InstrumentEnum,
    MfmEnum,
    RBMDataSet,
    RBMDataSetManager,
    SatelliteEnum,
    SatelliteLike,
    TargetType,
)


def create_RBSP_line_data(
    start_time: datetime,
    end_time: datetime,
    data_server_path: Path,
    target_en: float | Iterable[float],
    target_al: float | Iterable[float],
    target_type: TargetType | Literal["TargetPairs", "TargetMeshGrid"],
    energy_offset_threshold: float = 0.1,
    instruments: list[InstrumentEnum] | None = None,
    satellites: list[SatelliteLike] | SatelliteLike | None = None,
    mfm: MfmEnum = MfmEnum.T89,
    *,
    adjust_targets: bool = True,
    verbose: bool = True,
):
    # Instruments represents also the priority of the instrument for overlapping energies. The first instrument will be prefered.

    instruments = instruments or [
        InstrumentEnum.HOPE,
        InstrumentEnum.MAGEIS,
        InstrumentEnum.REPT,
    ]
    satellites = satellites or [SatelliteEnum.RBSPA, SatelliteEnum.RBSPB]

    # pass and check args
    if isinstance(data_server_path, str):
        data_server_path = Path(data_server_path)
    if not isinstance(target_al, Iterable):
        target_al = [target_al]
    if not isinstance(target_en, Iterable):
        target_en = [target_en]
    if not isinstance(satellites, Iterable) or isinstance(satellites, str):
        satellites = [satellites]
    if isinstance(target_type, str):
        target_type = TargetType[target_type]

    if target_type == TargetType.TargetPairs:
        assert len(target_en) == len(
            target_al
        ), "For TargetType.Pairs, the target vectors must have the same size!"

    result_arr = []
    list_instruments_used = []

    for satellite in satellites:
        rbm_data: list[RBMDataSet] = []

        for i, instrument in enumerate(instruments):
            rbm_data.append(
                RBMDataSetManager.load(
                    start_time,
                    end_time,
                    data_server_path,
                    satellite,
                    instrument,
                    mfm,
                    verbose=verbose,
                )
            )

            # strip of time dimention
            if rbm_data[i].energy_channels.shape[0] == len(rbm_data[i].time):
                rbm_data[i].energy_channels_no_time = np.nanmean(
                    rbm_data[i].energy_channels, axis=0
                )
            else:
                rbm_data[i].energy_channels_no_time = rbm_data[i].energy_channels
            if rbm_data[i].alpha_local.shape[0] == len(rbm_data[i].time):
                rbm_data[i].alpha_local_no_time = np.nanmean(
                    rbm_data[i].alpha_local, axis=0
                )
            else:
                rbm_data[i].alpha_local_no_time = rbm_data[i].alpha_local

        for e, target_en_single in enumerate(target_en):
            if verbose:
                print(f"Energy offset for target [{e}] ({target_en_single:.2e} MeV):")

            energy_offsets = np.empty((len(instruments),))

            for i, instrument in enumerate(instruments):
                energy_offsets[i] = np.nanmin(
                    np.abs(rbm_data[i].energy_channels_no_time - target_en_single),
                    axis=None,
                )

                if verbose:
                    print(
                        f"\t{instrument.name}:\tabsolute: {energy_offsets[i]:.2e}, \trelative: {energy_offsets[i]/target_en_single:.2e}"
                    )

                # initiate the RBMDataSet for the result
                if e == 0 and i == 0:
                    rbm_data_set_result = deepcopy(rbm_data[i])

                    if target_type == TargetType.TargetPairs:
                        rbm_data_set_result.line_data_flux = np.empty(
                            (len(rbm_data_set_result.time), len(target_en))
                        )
                        rbm_data_set_result.line_data_energy = np.empty(
                            (len(target_en),)
                        )
                        rbm_data_set_result.line_data_alpha_local = np.empty(
                            (len(target_al),)
                        )
                    elif target_type == TargetType.TargetMeshGrid:
                        rbm_data_set_result.line_data_flux = np.empty(
                            (
                                len(rbm_data_set_result.time),
                                len(target_en),
                                len(target_al),
                            )
                        )
                        rbm_data_set_result.line_data_energy = np.empty(
                            (len(target_en),)
                        )
                        rbm_data_set_result.line_data_alpha_local = np.empty(
                            (len(target_al),)
                        )

            energy_offsets_relative = energy_offsets / target_en_single
            if np.all(np.abs(energy_offsets_relative) > energy_offset_threshold):
                raise ValueError(
                    f"For the given energy target ({target_en_single:.2e} MeV), no suitable energy channel was found for a threshold of {energy_offset_threshold:.02f}!"
                )

            min_offset_instrument = np.argmax(
                np.abs(energy_offsets_relative) <= energy_offset_threshold
            )
            list_instruments_used.append(instruments[min_offset_instrument])

            if verbose:
                print(
                    f"Choosing {instruments[min_offset_instrument].value} with an offset of {energy_offsets_relative[min_offset_instrument]*100:.1f}%\n"
                )

            closest_en_idx = np.nanargmin(
                np.abs(
                    rbm_data[min_offset_instrument].energy_channels_no_time
                    - target_en_single
                )
            )
            rbm_data_set_result.line_data_energy[e] = rbm_data[
                min_offset_instrument
            ].energy_channels_no_time[closest_en_idx]

            if target_type == TargetType.TargetPairs:
                closest_al_idx = np.nanargmin(
                    np.abs(
                        rbm_data[min_offset_instrument].alpha_local_no_time
                        - target_al[e]
                    )
                )
                rbm_data_set_result.line_data_alpha_local[e] = rbm_data[
                    min_offset_instrument
                ].alpha_local_no_time[closest_al_idx]

                if adjust_targets:
                    rbm_data_set_result.line_data_flux[:, e] = rbm_data[
                        min_offset_instrument
                    ].Flux[:, closest_en_idx, closest_al_idx]
                else:
                    rbm_data_set_result.line_data_flux[:, e] = np.squeeze(
                        rbm_data[min_offset_instrument].interp_flux(
                            target_en_single, target_al[e], TargetType.TargetPairs
                        )
                    )

            elif target_type == TargetType.TargetMeshGrid:
                for a, target_al_single in enumerate(target_al):
                    closest_al_idx = np.nanargmin(
                        np.abs(
                            rbm_data[min_offset_instrument].alpha_local_no_time
                            - target_al_single
                        )
                    )
                    rbm_data_set_result.line_data_alpha_local[a] = rbm_data[
                        min_offset_instrument
                    ].alpha_local_no_time[closest_al_idx]

                    if adjust_targets:
                        rbm_data_set_result.line_data_flux[:, e, a] = rbm_data[
                            min_offset_instrument
                        ].Flux[:, closest_en_idx, closest_al_idx]
                    else:
                        rbm_data_set_result.line_data_flux[:, e, a] = np.squeeze(
                            rbm_data[min_offset_instrument].interp_flux(
                                target_en_single,
                                target_al_single,
                                TargetType.TargetPairs,
                            )
                        )

        result_arr.append(rbm_data_set_result)

    return result_arr, list_instruments_used
