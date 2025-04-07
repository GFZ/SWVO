from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import pandas as pd

from data_management.io.solar_wind import SWACE, SWOMNI, SWSWIFTEnsemble, DSCOVR
from data_management.io.utils import any_nans, construct_updated_data_frame

SWModel = DSCOVR | SWACE | SWOMNI | SWSWIFTEnsemble


def read_solar_wind_from_multiple_models(  # noqa: PLR0913
    start_time: datetime,
    end_time: datetime,
    model_order: list[SWModel] | None = None,
    reduce_ensemble: str | None = None,
    synthetic_now_time: datetime | None = None,
    *,
    download: bool = False,
) -> pd.DataFrame | list[pd.DataFrame]:
    """
    Read solar wind data from multiple models.

    The model order represents the priorities of models. The first model in the model order is read. If there are still NaNs in the resulting data, the next model will be read. And so on. In the case of reading ensemble predictions, a list will be returned, otherwise a plain data frame will be returned.

    Parameters
    ----------
    start_time : datetime
        Start time of the data request.
    end_time : datetime
        End time of the data request.
    model_order : list, optional
        Order in which data will be read from the models. Defaults to [OMNI, ACE, SWIFT].
    reduce_ensemble : {'mean'}, optional
        The method to reduce ensembles to a single time series. Defaults to None.
    synthetic_now_time : datetime, optional
        Time which represents "now". After this time, no data will be taken from historical models (OMNI, ACE). Defaults to None.
    download : bool, optional
        Flag which decides whether new data should be downloaded. Defaults to False.

    Returns
    -------
    pd.DataFrame or list of pandas.DataFrame
        A data frame or a list of data frames containing data for the requested period.
    """

    if synthetic_now_time is None:
        synthetic_now_time = min(datetime.now(timezone.utc), end_time)

    if model_order is None:
        model_order = [SWOMNI(), DSCOVR(), SWACE(), SWSWIFTEnsemble()]
        logging.warning("No model order specified, using default order: OMNI, ACE, SWIFT ensemble")

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
    model: SWModel,
    start_time: datetime,
    end_time: datetime,
    synthetic_now_time: datetime,
    reduce_ensemble: str,
    *,
    download: bool,
) -> list[pd.DataFrame] | pd.DataFrame:
    """Reads SW data from a given model within the specified time range.

    Parameters
    ----------
    model : SWModel
        The model from which to read the SW data.
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
    if isinstance(model, (DSCOVR, SWACE, SWOMNI)):
        data_one_model = _read_historical_model(
            model,
            start_time,
            end_time,
            synthetic_now_time,
            download=download,
        )

    # Forecasting models are called with synthetic now time
    if isinstance(model, SWSWIFTEnsemble):
        data_one_model = _read_latest_ensemble_files(model, synthetic_now_time, end_time)

        num_ens_members = len(data_one_model)

        if num_ens_members > 0 and reduce_ensemble is not None:
            data_one_model = _reduce_ensembles(data_one_model, reduce_ensemble)

    return data_one_model


def _read_historical_model(
    model: DSCOVR | SWACE | SWOMNI,
    start_time: datetime,
    end_time: datetime,
    synthetic_now_time: datetime,
    *,
    download: bool,
) -> tuple[pd.DataFrame, str]:
    
    """Reads SW data from historical models (DSCOVR, SWACE or SWOMNI) within the specified time range.

    Parameters
    ----------
    model : DSCOVR | SWACE | SWOMNI
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
        If the provided model is not an instance of DSCOVR, SWACE or SWOMNI.

    """
    if isinstance(model, SWOMNI):
        model_label = "omni"
    elif isinstance(model, SWACE):
        model_label = "ace"
    elif isinstance(model, DSCOVR):
        model_label = "dscovr"
    else:
        msg = "Encountered invalide model type in read historical model!"
        raise TypeError(msg)

    logging.info(f"Reading {model.LABEL} from {start_time} to {end_time}")

    data_one_model = model.read(start_time, end_time, download=download)
    # set nan for 'future' values
    data_one_model.loc[synthetic_now_time+timedelta(minutes=1):end_time] = np.nan
    logging.info(f"Setting NaNs in {model.LABEL} from {synthetic_now_time} to {end_time}")

    return data_one_model


def _read_latest_ensemble_files(
    model: SWSWIFTEnsemble,
    synthetic_now_time: datetime,
    end_time: datetime,
) -> list[pd.DataFrame]:
    # we are trying to read the most recent file; it this fails, we go one step back (1 day) and see if this file is present

    """
    Reads the most recent SW ensemble data file available from the specified model.

    If the file for the target time is not found, the function iterates backward in hourly increments, up to 5 days, until a valid file is located.

    Parameters
    ----------
    model : SWSWIFTEnsemble
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

    target_time = min(synthetic_now_time, end_time)
    data_one_model = []

    while target_time > (synthetic_now_time - timedelta(days=5)):
        data_one_model = model.read(target_time, end_time)

        if len(data_one_model) == 0:
            target_time -= timedelta(days=1)
            continue

        data_one_model = _interpolate_to_common_indices(target_time, end_time, synthetic_now_time, data_one_model)
        break

    logging.info(f"Reading SWIFT ensemble from {target_time} to {end_time}")

    return data_one_model


def _interpolate_to_common_indices(
    target_time: datetime, end_time: datetime, synthetic_now_time: datetime, data: list[pd.DataFrame]
) -> list[pd.DataFrame]:
    """
    Interpolate the data to a common index with a 1-minute frequency.

    Parameters
    ----------
    target_time : datetime
        The start time for the interpolation.
    end_time : datetime
        The end time for the interpolation.
    synthetic_now_time : datetime
        The "now" time, used for truncating data after interpolation.
    data : list[pd.DataFrame]
        The list of data frames to interpolate.

    Returns
    -------
    list[pd.DataFrame]
        The list of interpolated data frames with a common index.
    """

    for ie, _ in enumerate(data):
        df_common_index = pd.DataFrame(
            index=pd.date_range(
                datetime(target_time.year, target_time.month, target_time.day, tzinfo=timezone.utc),
                datetime(end_time.year, end_time.month, end_time.day, 23, 59, 59, tzinfo=timezone.utc),
                freq=timedelta(minutes=1),
                tz="UTC",
            ),
        )
        df_common_index.index.name = data[ie].index.name

        for colname, col in data[ie].items():
            if col.dtype == "object":
                # this is the filename column
                df_common_index[colname] = col.iloc[0]
            else:
                df_common_index[colname] = np.interp(df_common_index.index, data[ie].index, col)

        data[ie] = df_common_index
        data[ie] = data[ie].truncate(
            before=synthetic_now_time - timedelta(minutes=0.999999), after=end_time + timedelta(minutes=0.999999)
        )

    return data


def _reduce_ensembles(data_ensembles: list[pd.DataFrame], method: Literal["mean"]) -> pd.DataFrame:
    """Reduce a list of data frames representing ensemble data to a single data frame using the provided method."""
    msg = "This reduction method has not been implemented yet!"
    raise NotImplementedError(msg)