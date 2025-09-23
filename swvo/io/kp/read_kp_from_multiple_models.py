# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""Function to read Kp from multiple models."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import pandas as pd

from swvo.io.exceptions import ModelError
from swvo.io.kp import KpEnsemble, KpNiemegk, KpOMNI, KpSWPC
from swvo.io.utils import any_nans, construct_updated_data_frame

KpModel = KpEnsemble | KpNiemegk | KpOMNI | KpSWPC

logging.captureWarnings(True)


def read_kp_from_multiple_models(  # noqa: PLR0913
    start_time: datetime,
    end_time: datetime,
    model_order: Sequence[KpModel] | None = None,
    reduce_ensemble: Literal["mean", "median"] | None = None,
    historical_data_cutoff_time: datetime | None = None,
    *,
    download: bool = False,
    recurrence: bool = False,
    rec_model_order: Sequence[KpOMNI | KpNiemegk] | None = None,
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
    model_order : Sequence or None, optional
        The order in which data will be read from the models. Defaults to [OMNI, Niemegk, Ensemble, SWPC].
    reduce_ensemble : {"mean", "median"} or None, optional
        The method to reduce ensembles to a single time series. Can be "mean", "median", or None. Defaults to None.
    historical_data_cutoff_time : datetime or None, optional
        Represents "now". After this time, no data will be taken from historical models
        (OMNI, Niemegk). Defaults to None.
    download : bool, optional
        Flag to decide whether new data should be downloaded. Defaults to False.
        Also applies to recurrence filling.
    recurrence : bool, optional
        If True, fill missing values using 27-day recurrence from historical models (OMNI, Niemegk).
        Defaults to False.
    rec_model_order : Sequence[KpOMNI | KpNiemegk], optional
        The order in which historical models will be used for 27-day recurrence filling.
        Defaults to [OMNI, Niemegk].


    Returns
    -------
    Union[:class:`pandas.DataFrame`, list[:class:`pandas.DataFrame`]]
        A data frame or a list of data frames containing data for the requested period.

    """
    if historical_data_cutoff_time is None:
        historical_data_cutoff_time = min(datetime.now(timezone.utc), end_time)

    if model_order is None:
        model_order = [KpOMNI(), KpNiemegk(), KpEnsemble(), KpSWPC()]
        logging.warning("No model order specified, using default order: OMNI, Niemegk, Ensemble, SWPC")

    data_out = [pd.DataFrame()]

    for model in model_order:
        if not isinstance(model, KpModel):
            raise ModelError(f"Unknown or incompatible model: {type(model).__name__}")
        data_one_model = _read_from_model(
            model,
            start_time,
            end_time,
            historical_data_cutoff_time,
            reduce_ensemble,
            download=download,
        )
        data_out = construct_updated_data_frame(data_out, data_one_model, model.LABEL)
        if not any_nans(data_out):
            break

    if recurrence:
        if rec_model_order is None:
            rec_model_order = [m for m in model_order if isinstance(m, (KpOMNI, KpNiemegk))]
        for i, df in enumerate(data_out):
            if not df.empty:
                data_out[i] = _recursive_fill_27d_historical(df, download, rec_model_order)

    if len(data_out) == 1:
        data_out = data_out[0]

    return data_out


def _read_from_model(  # noqa: PLR0913
    model: KpModel,
    start_time: datetime,
    end_time: datetime,
    historical_data_cutoff_time: datetime,
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
    historical_data_cutoff_time : datetime
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
            historical_data_cutoff_time,
            download=download,
        )

    # Forecasting models are called with synthetic now time
    if isinstance(model, KpSWPC):
        logging.info(
            f"Reading swpc from {historical_data_cutoff_time.replace(hour=0, minute=0, second=0)} to {end_time}\noriginal historical_data_cutoff_time: {historical_data_cutoff_time}"
        )
        data_one_model = [
            model.read(
                historical_data_cutoff_time.replace(hour=0, minute=0, second=0),
                end_time,
                download=download,
            )
        ]

    if isinstance(model, KpEnsemble):
        data_one_model = _read_latest_ensemble_files(model, historical_data_cutoff_time, end_time)

        num_ens_members = len(data_one_model)

        if num_ens_members > 0 and reduce_ensemble is not None:
            data_one_model = _reduce_ensembles(data_one_model, reduce_ensemble)

    return data_one_model


def _read_historical_model(
    model: KpOMNI | KpNiemegk,
    start_time: datetime,
    end_time: datetime,
    historical_data_cutoff_time: datetime,
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
    historical_data_cutoff_time : datetime
        Represents "now". Data after this time is set to NaN.
    download : bool, optional
        Whether to download new data or not.

    Returns
    -------
    pd.DataFrame
        A data frame containing the model data with future values (after historical_data_cutoff_time) set to NaN.

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
    data_one_model.loc[historical_data_cutoff_time + timedelta(hours=3) : end_time, "kp"] = np.nan
    logging.info(f"Setting NaNs in {model.LABEL} from {historical_data_cutoff_time + timedelta(hours=3)} to {end_time}")

    return data_one_model


def _read_latest_ensemble_files(
    model: KpEnsemble,
    historical_data_cutoff_time: datetime,
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
    historical_data_cutoff_time : datetime
        Represents "now". The function starts searching for files from this time.
    end_time : datetime
        The end time of the data range.

    Returns
    -------
    list[pd.DataFrame]
        A list of data frames containing ensemble data for the specified range.
    """
    target_time = historical_data_cutoff_time
    data_one_model = pd.DataFrame(data={"kp": []})

    while target_time > (historical_data_cutoff_time - timedelta(days=3)):
        target_time = target_time.replace(minute=0, second=0)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="No ensemble files found")
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


def _reduce_ensembles(data_ensembles: list[pd.DataFrame], method: Literal["mean", "median"]) -> pd.DataFrame:
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
                data_one_ensemble.loc[data_one_ensemble.index[it], "kp"] for data_one_ensemble in data_ensembles
            ]

            kp_mean_ensembles.append(np.mean(data_curr_time))

        data_reduced = pd.DataFrame(index=data_ensembles[0].index, data={"kp": kp_mean_ensembles})

    elif method == "median":
        kp_median_ensembles = []

        for it, _ in enumerate(data_ensembles[0].index):
            data_curr_time = [
                data_one_ensemble.loc[data_one_ensemble.index[it], "kp"] for data_one_ensemble in data_ensembles
            ]

            kp_median_ensembles.append(np.median(data_curr_time))

        data_reduced = pd.DataFrame(index=data_ensembles[0].index, data={"kp": kp_median_ensembles})

    else:
        msg = f"This reduction method has not been implemented yet: {method}!"
        raise NotImplementedError(msg)

    return data_reduced


def _recursive_fill_27d_historical(df, download, historical_models):
    """Recursively fill missing values in using OMNI/Niemegk for (`date` - 27 days).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to fill with gaps.
    download : bool
        Download new data or not.
    historical_models : list[KpOMNI | KpNiemegk]
        List of historical models to use for filling gaps.
    value_col : str, optional
        _description_, by default "kp"

    Returns
    -------
    pd.DataFrame
        DataFrame with gaps filled using 27d recurrence.
    """
    df = df.copy()
    value_col = "kp"
    missing = df.index[df[value_col].isna()]
    tried = set()
    while len(missing) > 0:
        fill_map = {}
        for idx in missing:
            prev_idx = idx - timedelta(days=27)
            if prev_idx not in tried:
                # Try each historical model in priority order
                for model in historical_models:
                    prev_data = model.read(
                        prev_idx - timedelta(days=3), prev_idx + timedelta(days=3), download=download
                    )
                    if not prev_data.empty and not pd.isna(prev_data.loc[prev_idx, value_col]):
                        fill_map[idx] = (
                            prev_data.loc[prev_idx, value_col],
                            model.LABEL,
                            prev_data.loc[prev_idx, "file_name"],
                        )
                        break
                tried.add(prev_idx)
        for idx, (val, label, fname) in fill_map.items():
            df.loc[idx, value_col] = val
            df.loc[idx, "model"] = f"{label}_recurrence"
            df.loc[idx, "file_name"] = fname
        # Update missing for next recursion
        missing = df.index[df[value_col].isna()]
        if not fill_map:
            break
    return df
