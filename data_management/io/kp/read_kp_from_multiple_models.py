"""Function to read Kp from multiple models."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import pandas as pd

from data_management.io.kp import KpEnsemble, KpNiemegk, KpOMNI, KpSWPC

KpModel = KpEnsemble | KpNiemegk | KpOMNI | KpSWPC


def _read_from_model(  # noqa: PLR0913
    model: KpModel,
    start_time: datetime,
    end_time: datetime,
    synthetic_now_time: datetime,
    reduce_ensemble: str,
    *,
    download: bool,
) -> list[pd.DataFrame] | pd.DataFrame:
    # Read from historical models
    if isinstance(model, (KpOMNI, KpNiemegk)):
        data_one_model, model_label = _read_historical_model(
            model,
            start_time,
            end_time,
            synthetic_now_time,
            download=download,
        )

    # Forecasting models are called with synthetic now time
    if isinstance(model, KpSWPC):
        logging.info(f"Reading swpc from {synthetic_now_time} to {end_time}")
        data_one_model = [
            model.read(synthetic_now_time.replace(hour=0, minute=0, second=0), end_time, download=download),
        ]
        model_label = "swpc"

    if isinstance(model, KpEnsemble):
        data_one_model = _read_latest_ensemble_files(model, synthetic_now_time, end_time)

        model_label = "ensemble"
        num_ens_members = len(data_one_model)

        if num_ens_members > 0 and reduce_ensemble is not None:
            data_one_model = _reduce_ensembles(data_one_model, reduce_ensemble)

    return data_one_model, model_label


def _read_historical_model(
    model: KpOMNI | KpNiemegk,
    start_time: datetime,
    end_time: datetime,
    synthetic_now_time: datetime,
    *,
    download: bool,
) -> tuple[pd.DataFrame, str]:
    if isinstance(model, KpOMNI):
        model_label = "omni"
    elif isinstance(model, KpNiemegk):
        model_label = "niemegk"
    else:
        msg = "Encountered invalide model type in read historical model!"
        raise TypeError(msg)

    logging.info(f"Reading {model_label} from {start_time} to {end_time}")

    data_one_model = model.read(start_time, end_time, download=download)
    # set nan for 'future' values
    data_one_model.loc[synthetic_now_time:end_time, "kp"] = np.nan
    logging.info(f"Setting NaNs in {model_label} from {synthetic_now_time} to {end_time}")

    return data_one_model, model_label


def _read_latest_ensemble_files(
    model: KpEnsemble,
    synthetic_now_time: datetime,
    end_time: datetime,
) -> list[pd.DataFrame]:
    # we are trying to read the most recent file; it this fails, we go 1 hour back and see if this file is present

    target_time = synthetic_now_time
    data_one_model = model.read(target_time, end_time)

    while len(data_one_model) == 0 and target_time > (synthetic_now_time - timedelta(days=3)):
        target_time -= timedelta(hours=1)

        # ONLY READ MIDNIGHT FILE FOR NOW; OTHER FILES BREAK
        target_time = target_time.replace(hour=0, minute=0, second=0)

        data_one_model = model.read(target_time, end_time)

    logging.info(f"Reading PAGER Kp ensemble from {target_time} to {end_time}")

    return data_one_model


def _reduce_ensembles(data_ensembles: list[pd.DataFrame], method: Literal["mean"]) -> pd.DataFrame:
    if method == "mean":
        kp_mean_ensembles = []

        for it, _ in enumerate(data_ensembles[0].index):
            data_curr_time = [
                data_one_ensemble.loc[data_one_ensemble.index[it], "kp"] for data_one_ensemble in data_ensembles
            ]

            kp_mean_ensembles.append(np.mean(data_curr_time))

        data_reduced = pd.DataFrame(index=data_ensembles[0].index, data={"kp": kp_mean_ensembles})
    else:
        msg = "This reduction method has not been implemented yet!"
        raise NotImplementedError(msg)

    return data_reduced


def _construct_updated_data_frame(
    data: list[pd.DataFrame],
    data_one_model: list[pd.DataFrame],
    model_label: str,
) -> list[pd.DataFrame]:
    if isinstance(data_one_model, pd.DataFrame):
        data_one_model = [data_one_model]

    # extend the data we have read so far to match the new ensemble numbers
    if len(data) == 1 and len(data_one_model) > 1:
        data = data * len(data_one_model)
    elif len(data) != len(data_one_model):
        msg = f"Tried to combine models with different ensemble numbers: {len(data)} and {len(data_one_model)}"
        raise ValueError(msg)

    for i, _ in enumerate(data_one_model):
        data_one_model[i]["model"] = model_label
        data_one_model[i].loc[data_one_model[i]["kp"].isna(), "model"] = None
        data[i] = data[i].combine_first(data_one_model[i])

    return data


def _any_nans(data: list[pd.DataFrame]) -> bool:
    return any(df["kp"].isna().sum() > 0 for df in data)


def read_kp_from_multiple_models(  # noqa: PLR0913
    start_time: datetime,
    end_time: datetime,
    model_order: list | None = None,
    reduce_ensemble: Literal["mean"] | None = None,
    synthetic_now_time: datetime | None = None,
    *,
    download: bool = False,
) -> pd.DataFrame | list[pd.DataFrame]:
    """Read Kp data from multiple models.

    The model order represents the priorities of models.
    The first model in the model order is read. If there are still NaNs in the resulting data,
    the next model will be read. And so on. In the case of reading ensemble predictions, a list
    will be returned, otherwise a plain data frame will be returned.

    :param start_time: Start time of the data request.
    :type start_time: datetime
    :param end_time: End time of the data request.
    :type end_time: datetime
    :param model_order: Order in which data will be read from the models, defaults to [OMNI, Niemegk, Ensemble, SWPC]
    :type model_order: list | None, optional
    :param reduce_ensemble: The method to reduce ensembles to a single time series, defaults to None
    :type reduce_ensemble: Literal[&quot;mean&quot;] | None, optional
    :param synthetic_now_time: Time, which represents &quot;now&quot;.
    After this time, no data will be taken from historical models (OMNI, Niemegk), defaults to None
    :type synthetic_now_time: datetime | None, optional
    :param download: Flag which decides whether new data should be downloaded, defaults to False
    :type download: bool, optional
    :return: A data frame or a list of data frames containing data for the requested period.
    :rtype: pd.DataFrame | list[pd.DataFrame]
    """
    if synthetic_now_time is None:
        synthetic_now_time = datetime.now(timezone.utc)

    if model_order is None:
        model_order = [KpOMNI(), KpNiemegk(), KpEnsemble(), KpSWPC()]
        logging.warning("No model order specified, using default order: OMNI, Niemegk, Ensemble, SWPC")

    data_out = [pd.DataFrame()]

    for model in model_order:
        data_one_model, model_label = _read_from_model(
            model,
            start_time,
            end_time,
            synthetic_now_time,
            reduce_ensemble,
            download=download,
        )

        data_out = _construct_updated_data_frame(data_out, data_one_model, model_label)
        if not _any_nans(data_out):
            break

    if len(data_out) == 1:
        data_out = data_out[0]

    return data_out
