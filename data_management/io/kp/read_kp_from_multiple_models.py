"""Function to read Kp from multiple models."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Literal
import warnings

import numpy as np
import pandas as pd

from data_management.io.kp import KpEnsemble, KpNiemegk, KpOMNI, KpSWPC
from data_management.io.utils import any_nans, construct_updated_data_frame

KpModel = KpEnsemble | KpNiemegk | KpOMNI | KpSWPC


def read_kp_from_multiple_models(  # noqa: PLR0913
    start_time: datetime,
    end_time: datetime,
    model_order: list[KpModel] | None = None,
    reduce_ensemble: Literal["mean", "median"] | None = None,
    synthetic_now_time: datetime | None = None,
    *,
    download: bool = False,
) -> pd.DataFrame | list[pd.DataFrame]:
    """Read Kp data from multiple models.

    The model order determines the priority of models. Data is read from the first model in the
    model order. If there are still NaNs in the resulting data, the next model is read, and so on.
    For ensemble predictions, a list of data frames is returned; otherwise, a single data frame
    is returned.

    Parameters
    ----------
    start_time : datetime
        The start time of the data request.
    end_time : datetime
        The end time of the data request.
    model_order : list or None, optional
        The order in which data will be read from the models. Defaults to [OMNI, Niemegk, Ensemble, SWPC].
    reduce_ensemble : {"mean", None}, optional
        The method to reduce ensembles to a single time series. Defaults to None.
    synthetic_now_time : datetime or None, optional
        Represents "now". After this time, no data will be taken from historical models
        (OMNI, Niemegk). Defaults to None.
    download : bool, optional
        Flag to decide whether new data should be downloaded. Defaults to False.

    Returns
    -------
    pd.DataFrame or list of pd.DataFrame
        A data frame or a list of data frames containing data for the requested period.

    """
    if synthetic_now_time is None:
        synthetic_now_time = min(datetime.now(timezone.utc), end_time)

    if model_order is None:
        model_order = [KpOMNI(), KpNiemegk(), KpEnsemble(), KpSWPC()]
        logging.warning(
            "No model order specified, using default order: OMNI, Niemegk, Ensemble, SWPC"
        )

    data_out = [pd.DataFrame()]

    for model in model_order:
        data_one_model = _read_from_model(
            model,
            start_time,
            end_time,
            synthetic_now_time,
            reduce_ensemble,
            download=download,
        )
        data_out = construct_updated_data_frame(data_out, data_one_model, model.LABEL)
        if not any_nans(data_out):
            break

    if len(data_out) == 1:
        data_out = data_out[0]

    return data_out


def _read_from_model(  # noqa: PLR0913
    model: KpModel,
    start_time: datetime,
    end_time: datetime,
    synthetic_now_time: datetime,
    reduce_ensemble: str,
    *,
    download: bool,
) -> list[pd.DataFrame] | pd.DataFrame:
    """Reads Kp data from a given model within the specified time range.

    Parameters
    ----------
    model : KpModel
        The model from which to read the Kp data.
    start_time : datetime
        The start time of the data range.
    end_time : datetime
        The end time of the data range.
    synthetic_now_time : datetime
        Represents "now". Used for defining boundaries for historical or forecast data.
    reduce_ensemble : str
        The method to reduce ensemble data (e.g., "mean"). If None, ensemble members are not reduced.
    download : bool, optional
        Whether to download new data or not.

    Returns
    -------
    list[pd.DataFrame] | pd.DataFrame
        A single data frame or a list of data frames containing the model data.

    """
    # Read from historical models
    if isinstance(model, (KpOMNI, KpNiemegk)):
        data_one_model = _read_historical_model(
            model,
            start_time,
            end_time,
            synthetic_now_time,
            download=download,
        )

    # Forecasting models are called with synthetic now time
    if isinstance(model, KpSWPC):
        logging.info(
            f"Reading swpc from {synthetic_now_time.replace(hour=0, minute=0, second=0)} to {end_time}\noriginal synthetic_now_time: {synthetic_now_time}"
        )
        data_one_model = [
            model.read(
                synthetic_now_time.replace(hour=0, minute=0, second=0),
                end_time,
                download=download,
            )
        ]

    if isinstance(model, KpEnsemble):
        data_one_model = _read_latest_ensemble_files(
            model, synthetic_now_time, end_time
        )

        num_ens_members = len(data_one_model)

        if num_ens_members > 0 and reduce_ensemble is not None:
            data_one_model = _reduce_ensembles(data_one_model, reduce_ensemble)

    return data_one_model


def _read_historical_model(
    model: KpOMNI | KpNiemegk,
    start_time: datetime,
    end_time: datetime,
    synthetic_now_time: datetime,
    *,
    download: bool,
) -> pd.DataFrame:
    """Reads Kp data from historical models (KpOMNI or KpNiemegk) within the specified time range.

    Parameters
    ----------
    model : KpOMNI | KpNiemegk
        The historical model from which to read the data.
    start_time : datetime
        The start time of the data range.
    end_time : datetime
        The end time of the data range.
    synthetic_now_time : datetime
        Represents "now". Data after this time is set to NaN.
    download : bool, optional
        Whether to download new data or not.

    Returns
    -------
    pd.DataFrame
        A data frame containing the model data with future values (after synthetic_now_time) set to NaN.

    Raises
    ------
    TypeError
        If the provided model is not an instance of KpOMNI or KpNiemegk.

    """
    if not isinstance(model, (KpOMNI, KpNiemegk)):
        msg = "Encountered invalide model type in read historical model!"
        raise TypeError(msg)

    logging.info(f"Reading {model.LABEL} from {start_time} to {end_time}")

    data_one_model = model.read(start_time, end_time, download=download)
    # set nan for 'future' values
    data_one_model.loc[synthetic_now_time + timedelta(hours=3) : end_time, "kp"] = (
        np.nan
    )
    logging.info(
        f"Setting NaNs in {model.LABEL} from {synthetic_now_time + timedelta(hours=3)} to {end_time}"
    )

    return data_one_model


def _read_latest_ensemble_files(
    model: KpEnsemble,
    synthetic_now_time: datetime,
    end_time: datetime,
) -> list[pd.DataFrame]:
    """
    Reads the most recent Kp ensemble data file available from the specified model.

    If the file for the target time is not found, the function iterates backward in hourly
    increments, up to 3 days, until a valid file is located.

    Ensures that the last index of every dataframe is the next higher multiple of 3 hours
    than the target time.

    Parameters
    ----------
    model : KpEnsemble
        The ensemble model from which to read the data.
    synthetic_now_time : datetime
        Represents "now". The function starts searching for files from this time.
    end_time : datetime
        The end time of the data range.

    Returns
    -------
    list[pd.DataFrame]
        A list of data frames containing ensemble data for the specified range.
    """
    target_time = synthetic_now_time
    data_one_model = pd.DataFrame(data={"kp": []})

    while target_time > (synthetic_now_time - timedelta(days=3)):
        target_time = target_time.replace(minute=0, second=0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                data_one_model = model.read(target_time, end_time)
                break
        except UserWarning as e:
            if "No ensemble files found" in str(e):
                target_time -= timedelta(hours=1)
                continue
        
    logging.info(f"Read PAGER Kp ensemble from {target_time} to {end_time}")

    # Ensure the last index of every DataFrame is the next higher multiple of 3 hours than target_time
    adjusted_data = []
    for df in data_one_model:
        if not df.empty:
            if df.index[-1] < end_time and (df.index[-1] - end_time) < timedelta(hours=3):
                df.loc[df.index[-1] + timedelta(hours=3)] = df.loc[df.index[-1]]
        adjusted_data.append(df)
    return adjusted_data


def _reduce_ensembles(
    data_ensembles: list[pd.DataFrame], method: Literal["mean", "median"]
) -> pd.DataFrame:
    """
    Reduce a list of data frames representing ensemble data to a single data frame using the provided method.

    Parameters
    ----------
    data_ensembles : list[pd.DataFrame]
        A list of data frames containing ensemble data.
    method : {"mean", "median"}
        The method to reduce the ensemble data.

    Returns
    -------
    pd.DataFrame
        A data frame containing the reduced ensemble data.

    Raises
    ------
    NotImplementedError
        If the provided reduction method is not implemented.

    """
    if method == "mean":
        kp_mean_ensembles = []

        for it, _ in enumerate(data_ensembles[0].index):
            data_curr_time = [
                data_one_ensemble.loc[data_one_ensemble.index[it], "kp"]
                for data_one_ensemble in data_ensembles
            ]

            kp_mean_ensembles.append(np.mean(data_curr_time))

        data_reduced = pd.DataFrame(
            index=data_ensembles[0].index, data={"kp": kp_mean_ensembles}
        )

    elif method == "median":
        kp_median_ensembles = []

        for it, _ in enumerate(data_ensembles[0].index):
            data_curr_time = [
                data_one_ensemble.loc[data_one_ensemble.index[it], "kp"]
                for data_one_ensemble in data_ensembles
            ]

            kp_median_ensembles.append(np.median(data_curr_time))

        data_reduced = pd.DataFrame(
            index=data_ensembles[0].index, data={"kp": kp_median_ensembles}
        )

    else:
        msg = f"This reduction method has not been implemented yet: {method}!"
        raise NotImplementedError(msg)

    return data_reduced
