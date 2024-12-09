from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import pandas as pd

from data_management.io.solar_wind import SWACE, SWOMNI, SWSWIFTEnsemble, DSCOVR

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

    The model order represents the priorities of models.
    The first model in the model order is read. If there are still NaNs in the resulting data,
    the next model will be read. And so on. In the case of reading ensemble predictions, a list
    will be returned, otherwise a plain data frame will be returned.

    :param start_time: Start time of the data request.
    :type start_time: datetime
    :param end_time: End time of the data request.
    :type end_time: datetime
    :param model_order: Order in which data will be read from the models, defaults to [OMNI, ACE, SWIFT]
    :type model_order: list | None, optional
    :param reduce_ensemble: The method to reduce ensembles to a single time series, defaults to None
    :type reduce_ensemble: Literal[&quot;mean&quot;] | None, optional
    :param synthetic_now_time: Time, which represents &quot;now&quot;.
    After this time, no data will be taken from historical models (OMNI, ACE), defaults to None
    :type synthetic_now_time: datetime | None, optional
    :param download: Flag which decides whether new data should be downloaded, defaults to False
    :type download: bool, optional
    :return: A data frame or a list of data frames containing data for the requested period.
    :rtype: pd.DataFrame | list[pd.DataFrame]
    """
    if synthetic_now_time is None:
        synthetic_now_time = datetime.now(timezone.utc)

    if model_order is None:
        model_order = [SWOMNI(), DSCOVR(), SWACE(), SWSWIFTEnsemble()]
        logging.warning("No model order specified, using default order: OMNI, ACE, SWIFT ensemble")

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


def _read_from_model(  # noqa: PLR0913
    model: SWModel,
    start_time: datetime,
    end_time: datetime,
    synthetic_now_time: datetime,
    reduce_ensemble: str,
    *,
    download: bool,
) -> list[pd.DataFrame] | pd.DataFrame:
    # Read from historical models
    if isinstance(model, (DSCOVR, SWACE, SWOMNI)):
        data_one_model, model_label = _read_historical_model(
            model,
            start_time,
            end_time,
            synthetic_now_time,
            download=download,
        )

    # Forecasting models are called with synthetic now time
    if isinstance(model, SWSWIFTEnsemble):
        data_one_model = _read_latest_ensemble_files(model, synthetic_now_time, end_time)

        model_label = "swift"
        num_ens_members = len(data_one_model)

        if num_ens_members > 0 and reduce_ensemble is not None:
            data_one_model = _reduce_ensembles(data_one_model, reduce_ensemble)

    return data_one_model, model_label


def _read_historical_model(
    model: DSCOVR | SWACE | SWOMNI,
    start_time: datetime,
    end_time: datetime,
    synthetic_now_time: datetime,
    *,
    download: bool,
) -> tuple[pd.DataFrame, str]:
    if isinstance(model, SWOMNI):
        model_label = "omni"
    elif isinstance(model, SWACE):
        model_label = "ace"
    elif isinstance(model, DSCOVR):
        model_label = "dscovr"
    else:
        msg = "Encountered invalide model type in read historical model!"
        raise TypeError(msg)

    logging.info(f"Reading {model_label} from {start_time} to {end_time}")

    data_one_model = model.read(start_time, end_time, download=download)
    # set nan for 'future' values
    data_one_model.loc[synthetic_now_time:end_time] = np.nan
    logging.info(f"Setting NaNs in {model_label} from {synthetic_now_time} to {end_time}")

    return data_one_model, model_label


def _read_latest_ensemble_files(
    model: SWSWIFTEnsemble,
    synthetic_now_time: datetime,
    end_time: datetime,
) -> list[pd.DataFrame]:
    # we are trying to read the most recent file; it this fails, we go one step back (1 day) and see if this file is present

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


def _construct_updated_data_frame(
    data: list[pd.DataFrame],
    data_one_model: list[pd.DataFrame],
    model_label: str,
) -> list[pd.DataFrame]:
    """
    Construct an updated data frame providing the previous data frame and the data frame of the current model call.

    Also adds the model label to the data frame.
    """
    if isinstance(data_one_model, list) and data_one_model == []:  # nothing to update
        return data

    if isinstance(data_one_model, pd.DataFrame):
        data_one_model = [data_one_model]

    # extend the data we have read so far to match the new ensemble numbers
    if len(data) == 1 and len(data_one_model) > 1:
        data = data * len(data_one_model)
    elif len(data) != len(data_one_model):
        msg = f"Tried to combine models with different ensemble numbers: {len(data)} and {len(data_one_model)}!"
        raise ValueError(msg)

    for i, _ in enumerate(data_one_model):
        data_one_model[i]["model"] = model_label
        data_one_model[i].loc[data_one_model[i].isna().any(axis=1), "model"] = None
        data[i] = data[i].combine_first(data_one_model[i])

    return data


def _any_nans(data: list[pd.DataFrame]) -> bool:
    return any((df.isna().any(axis=None) > 0) for df in data)
