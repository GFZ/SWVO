"""Function to read Hp from multiple models."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import pandas as pd

from data_management.io.hp import Hp30Ensemble, Hp30GFZ, Hp60Ensemble, Hp60GFZ
from data_management.io.utils import any_nans, construct_updated_data_frame
from data_management.io.decorators import add_time_docs, add_attributes_to_class_docstring, add_methods_to_class_docstring



HpModel = Hp30Ensemble | Hp30GFZ | Hp60Ensemble | Hp60GFZ


def read_hp_from_multiple_models(  # noqa: PLR0913
    start_time: datetime,
    end_time: datetime,
    model_order: list[HpModel] | None = None,
    hp_index: str = "hp30",
    reduce_ensemble: Literal["mean", "median"] | None = None,
    synthetic_now_time: datetime | None = None,
    *,
    download: bool = False,
) -> pd.DataFrame | list[pd.DataFrame]:
    """
    Read Hp data from multiple models.

    The model order represents the priorities of models.
    The first model in the model order is read. If there are still NaNs in the resulting data,
    the next model will be read. And so on. In the case of reading ensemble predictions, a list
    will be returned, otherwise a plain data frame will be returned.

    Parameters
    ----------
    start_time : datetime
        Start time of the data request.
    end_time : datetime
        End time of the data request.
    model_order : list, optional
        Order in which data will be read from the models, defaults to [OMNI, Niemegk, Ensemble, SWPC].
    reduce_ensemble : {"mean"}, optional
        The method to reduce ensembles to a single time series, defaults to None.
    synthetic_now_time : datetime, optional
        Time, which represents "now". After this time, no data will be taken from historical models (OMNI, Niemegk), defaults to None.
    download : bool, optional
        Flag which decides whether new data should be downloaded, defaults to False.

    Returns
    -------
    pd.DataFrame | list of pd.DataFrame
        A data frame or a list of data frames containing data for the requested period.
    """
    if synthetic_now_time is None:
        synthetic_now_time = datetime.now(timezone.utc)

    hp_index = hp_index.lower()

    if model_order is None:
        logging.warning("No model order specified, using default order: GFZ, Ensemble")
        if hp_index == "hp30":
            model_order = [Hp30GFZ(), Hp30Ensemble()]
        elif hp_index == "hp60":
            model_order = [Hp60GFZ(), Hp60Ensemble()]
        else:
            msg = f"Requested {hp_index} index does not exist! Possible options: hp30, hp60"
            raise ValueError(msg)
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


@add_time_docs(None)
def _read_from_model(  # noqa: PLR0913
    model: HpModel,
    start_time: datetime,
    end_time: datetime,
    synthetic_now_time: datetime,
    reduce_ensemble: str,
    *,
    download: bool,
) -> list[pd.DataFrame] | pd.DataFrame:
    """

    Parameters
    ----------
    start_time : datetime
        _description_
    end_time : datetime
        _description_
    synthetic_now_time : datetime
        _description_
    reduce_ensemble : str
        _description_
    download : bool
        _description_

    Returns
    -------
    list[pd.DataFrame] | pd.DataFrame
        _description_
    """
    # Read from historical models
    if isinstance(model, (Hp30GFZ, Hp60GFZ)):
        data_one_model = _read_historical_model(
            model,
            start_time,
            end_time,
            model.index,
            synthetic_now_time,
            download=download,
        )

    if isinstance(model, (Hp30Ensemble, Hp60Ensemble)):
        data_one_model = _read_latest_ensemble_files(
            model, model.index, synthetic_now_time, end_time
        )

        num_ens_members = len(data_one_model)

        if num_ens_members > 0 and reduce_ensemble is not None:
            data_one_model = _reduce_ensembles(
                data_one_model, reduce_ensemble, model.index
            )

    return data_one_model


def _read_historical_model(
    model: Hp30GFZ | Hp60GFZ,
    start_time: datetime,
    end_time: datetime,
    hp_index: str,
    synthetic_now_time: datetime,
    *,
    download: bool,
) -> pd.DataFrame:
    """
    Reads historical data from a specified model within a given time range and sets NaN for future values.

    Parameters:
    -----------

    model : Hp30GFZ | Hp60GFZ
        The model from which to read historical data.
    start_time :datetime
        The start time for the data reading.
    end_time : datetime
        The end time for the data reading.
    hp_index : str
        The index to be used for setting NaN values.
    synthetic_now_time : datetime
        The synthetic current time to determine future values.
    download : bool
        Flag indicating whether to download the data.

    Returns:
    --------
    pd.DataFrame
        DataFrame with the historical data.

    Raises:
    -------
    TypeError
        If the model is not an instance of Hp30GFZ or Hp60GFZ.
    """

    if not isinstance(model, (Hp30GFZ, Hp60GFZ)):
        msg = "Encountered invalide model type in read historical model!"
        raise TypeError(msg)

    logging.info(f"Reading {model.LABEL} from {start_time} to {end_time}")

    data_one_model = model.read(start_time, end_time, download=download)

    # set nan for 'future' values
    data_one_model.loc[synthetic_now_time:end_time, hp_index] = np.nan
    logging.info(
        f"Setting NaNs in {model.LABEL} from {synthetic_now_time} to {end_time}"
    )

    return data_one_model


def _read_latest_ensemble_files(
    model: Hp30Ensemble | Hp60Ensemble,
    hp_index: str,
    synthetic_now_time: datetime,
    end_time: datetime,
) -> list[pd.DataFrame]:
    # we are trying to read the most recent file; it this fails, we go 1 hour back and see if this file is present

    """
    Reads ensemble data from a specified model within a given time range.

    Parameters:
    -----------

    model : Hp30Ensemble | Hp60Ensemble
        The model from which to read ensemble data.
    hp_index : str
        The index to be used for setting NaN values.
    synthetic_now_time : datetime
        The synthetic current time to determine future values.
    end_time : datetime
        The end time for the data reading.

    Returns:
    --------
    tuple[pd.DataFrame]
        List of DataFrames with the ensemble data.

    Raises:
    -------
    FileNotFoundError
    """
    target_time = synthetic_now_time

    data_one_model = pd.DataFrame(data={hp_index: []})

    while target_time > (synthetic_now_time - timedelta(days=3)):
        # ONLY READ MIDNIGHT FILE FOR NOW; OTHER FILES BREAK
        target_time = target_time.replace(hour=0, minute=0, second=0)

        try:
            data_one_model = model.read(target_time, end_time)
            break
        except FileNotFoundError:
            target_time -= timedelta(hours=1)
            continue

    logging.info(f"Reading PAGER Hp ensemble from {target_time} to {end_time}")

    return data_one_model


def _reduce_ensembles(
    data_ensembles: list[pd.DataFrame], method: Literal["mean", "median"], hp_index: str
) -> pd.DataFrame:
    """Reduce a list of data frames representing ensemble data to a single data frame using the provided method.

    Parameters:
    -----------
    data_ensembles : list[pd.DataFrame]
        List of data frames representing ensemble data.
    method : str
        The method to reduce the ensemble data.
    hp_index : str
        Hp index.

    Returns:
    --------
    pd.DataFrame
        DataFrame with the reduced ensemble data.

    Raises:
    -------
    NotImplementedError
        If the method is not implemented.

    """
    if method == "mean":
        hp_mean_ensembles = []

        for it, _ in enumerate(data_ensembles[0].index):
            data_curr_time = [
                data_one_ensemble.loc[data_one_ensemble.index[it], hp_index]
                for data_one_ensemble in data_ensembles
            ]

            hp_mean_ensembles.append(np.mean(data_curr_time))

        data_reduced = pd.DataFrame(
            index=data_ensembles[0].index, data={hp_index: hp_mean_ensembles}
        )

    elif method == "median":
        hp_median_ensembles = []

        for it, _ in enumerate(data_ensembles[0].index):
            data_curr_time = [
                data_one_ensemble.loc[data_one_ensemble.index[it], hp_index]
                for data_one_ensemble in data_ensembles
            ]

            hp_median_ensembles.append(np.median(data_curr_time))

        data_reduced = pd.DataFrame(
            index=data_ensembles[0].index, data={hp_index: hp_median_ensembles}
        )

    else:
        msg = f"This reduction method has not been implemented yet: {method}!"
        raise NotImplementedError(msg)

    return data_reduced
