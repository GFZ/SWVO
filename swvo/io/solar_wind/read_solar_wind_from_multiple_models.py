# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

"""Function to read Solar Wind from multiple models."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from swvo.io.exceptions import ModelError
from swvo.io.solar_wind import (
    AVERAGE_VALUES_TO_FILL,
    DSCOVR,
    SWACE,
    SWENLIL,
    SWENLIL_BKG,
    SWENLIL_CME,
    SWIMAP,
    SWMIDL,
    SWOMNI,
    SWSWIFTEnsemble,
)
from swvo.io.utils import (
    any_nans,
    construct_updated_data_frame,
    enforce_utc_timezone,
)

logger = logging.getLogger(__name__)

SWModel = DSCOVR | SWACE | SWOMNI | SWMIDL | SWIMAP | SWSWIFTEnsemble | SWENLIL_BKG | SWENLIL_CME

ENLIL_MAX_LOOKBACK_DAYS = 5

logging.captureWarnings(True)


def read_solar_wind_from_multiple_models(
    start_time: datetime,
    end_time: datetime,
    model_order: Sequence[SWModel] | None = None,
    reduce_ensemble: Optional[Literal["mean", "median"]] = None,
    historical_data_cutoff_time: datetime | None = None,
    *,
    download: bool = False,
    fill_average: bool = False,
    do_interpolation: bool = False,
) -> pd.DataFrame | list[pd.DataFrame]:
    """
    Read solar wind data from multiple models.

    The model order represents the priorities of models. The first model in the model order is read. If there are still NaNs in the resulting data, the next model will be read. And so on. In the case of reading ensemble predictions, a list will be returned, otherwise a plain data frame will be returned.

    SWIFT and ENLIL are both forecast models and only ever provide data after `historical_data_cutoff_time`. Each in turn covers as much of the forecast window as it reaches and hands the remainder to the next one in the order. In the default order that chain is SWIFT, then `SWENLIL_CME`, then `SWENLIL_BKG`: where SWIFT runs out before `end_time` the latest ENLIL CME run carries on from there, a period with no CME run at all is left to `SWENLIL_BKG`, and whatever none of them reach stays NaN unless `fill_average` is set, which closes the tail with ten year averages. Ensembles come from SWIFT whenever it supplies one, in which case ENLIL contributes only its latest run, copied out to match the SWIFT members. ENLIL's own CME runs become the ensemble instead when SWIFT supplies nothing, either because it is absent from the model order or because it had no data for the requested period. The one exception is ENLIL placed *before* SWIFT, where SWIFT's ensemble size is not yet known and ENLIL therefore stays a single run. ENLIL rows are labelled `enlil_cme` or `enlil_bkg` in the `model` column, after the run mode they came from.

    Parameters
    ----------
    start_time : datetime
        Start time of the data request. Floored to the minute, since the returned data lives on a
        one minute grid whose phase this sets.
    end_time : datetime
        End time of the data request.
    model_order : list, optional
        Order in which data will be read from the models.
        Defaults to [OMNI, IMAP, ACE, SWIFT, ENLIL CME, ENLIL BKG].
    reduce_ensemble : Literal["mean", "median"], optional
        The method to reduce ensembles to a single time series. Defaults to None.
    historical_data_cutoff_time : datetime, optional
        Time which represents "now". After this time, no data will be taken from historical models (OMNI, ACE). Defaults to None.
    download : bool, optional
        Flag which decides whether new data should be downloaded. Defaults to False.
    fill_average : bool, optional
        If True, keep the final dataframe through the requested end time for average-based filling.
        Defaults to False.
    do_interpolation : bool, optional
        If True, apply spline interpolation to short gaps (<= 3 hours) in historical data.
        Defaults to False.

    Returns
    -------
    Union[:class:`pandas.DataFrame`, list[:class:`pandas.DataFrame`]]
        A data frame or a list of data frames containing data for the requested period.

    Raises
    ------
    ModelError
        If an unknown or incompatible model is provided in the model order.
    AssertionError
        If `reduce_ensemble` is not None, "mean" or "median".
    """

    assert reduce_ensemble in (None, "mean", "median"), "reduce_ensemble must be None, `mean` or `median`"

    if start_time > end_time:
        msg = "start_time must be before end_time"
        raise ValueError(msg)

    start_time = enforce_utc_timezone(start_time)
    end_time = enforce_utc_timezone(end_time)

    # Everything here lives on a one minute grid whose phase comes from start_time, while the
    # models themselves are sampled on whole minutes. A start_time carrying seconds would offset
    # the whole grid and leave it unable to line up with any model, yielding an empty frame.
    if start_time.second or start_time.microsecond:
        floored_start_time = start_time.replace(second=0, microsecond=0)
        logger.info(f"Flooring start_time {start_time} to {floored_start_time} to align it to the one minute grid")
        start_time = floored_start_time

    if historical_data_cutoff_time is not None:
        historical_data_cutoff_time = enforce_utc_timezone(historical_data_cutoff_time)

    if historical_data_cutoff_time is None:
        historical_data_cutoff_time = min(datetime.now(timezone.utc), end_time)

    if model_order is None:
        model_order = [SWOMNI(), SWIMAP(), SWACE(), SWSWIFTEnsemble(), SWENLIL_CME(), SWENLIL_BKG()]
        logger.warning(
            "No model order specified, using default order: SWOMNI, SWIMAP, SWACE, SWSWIFTEnsemble, "
            "SWENLIL_CME, SWENLIL_BKG"
        )

    swift_index = next((i for i, m in enumerate(model_order) if isinstance(m, SWSWIFTEnsemble)), None)
    forecast_indices = [i for i, m in enumerate(model_order) if isinstance(m, (SWSWIFTEnsemble, SWENLIL))]
    last_forecast_index = forecast_indices[-1] if forecast_indices else -1

    data_out = [pd.DataFrame()]
    forecast_models_tried = 0
    forecast_models_with_data = 0

    for model_index, model in enumerate(model_order):
        if not isinstance(model, SWModel):
            raise ModelError(f"Unknown or incompatible model: {type(model).__name__}")
        active_model = model
        try:
            data_one_model = _read_from_model(
                model,
                start_time,
                end_time,
                historical_data_cutoff_time,
                reduce_ensemble,  # ty: ignore[invalid-argument-type]
                len(data_out),
                download=download,
                do_interpolation=do_interpolation,
                swift_read_later=swift_index is not None and swift_index > model_index,
            )
        except ValueError as e:
            # Only DSCOVR has an ACE fallback (it stops being available past a fixed date). A
            # ValueError from any other model is a real failure of that model and must surface,
            # not be reported as a DSCOVR problem and silently answered with ACE data.
            if not isinstance(model, DSCOVR):
                raise

            logger.warning(f"Failed to read DSCOVR data because: {e}. Falling back to ACE.")
            # switch to SWACE if SWACE is already in the model_order, otherwise create a new instance of SWACE with "./data" as the data directory
            # also log this fallback action
            active_model = next((m for m in model_order if isinstance(m, SWACE)), None)

            if active_model is not None:
                logger.info("Falling back to SWACE model in the model order.")
            else:
                active_model = SWACE(Path("./data"))
                logger.info("Falling back to a new instance of SWACE model with default data directory './data'.")
            data_one_model = _read_from_model(
                active_model,
                start_time,
                end_time,
                historical_data_cutoff_time,
                reduce_ensemble,  # ty: ignore[invalid-argument-type]
                len(data_out),
                download=download,
                do_interpolation=do_interpolation,
                swift_read_later=swift_index is not None and swift_index > model_index,
            )

        if isinstance(active_model, (SWSWIFTEnsemble, SWENLIL)):
            forecast_models_tried += 1
            if _has_valid_data(data_one_model):
                forecast_models_with_data += 1
            else:
                logger.info(f"{active_model.LABEL} data not available for future dates")

        data_out = construct_updated_data_frame(data_out, data_one_model, active_model.LABEL)

        # A forecast model can stop short of end_time and pin the index there, which would leave
        # the next forecast model nowhere to write and end the loop before it is even read.
        if isinstance(active_model, (SWSWIFTEnsemble, SWENLIL)) and model_index < last_forecast_index:
            data_out = _extend_to_full_index(data_out, start_time, end_time)

        if not any_nans(data_out):
            break

    # Ensure continuous dataframe and handle forecast model unavailability
    data_out = _ensure_continuous_dataframe(
        data_out,
        start_time,
        end_time,
        historical_data_cutoff_time,
        forecast_models_tried == 0 or forecast_models_with_data > 0,
        truncate=not fill_average,
    )

    if fill_average:
        logger.info("Filling future values with 10-year average values.")
        for i, df in enumerate(data_out):
            if df.empty:
                continue
            numeric_cols = [
                col for col in AVERAGE_VALUES_TO_FILL if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
            ]
            if not numeric_cols:
                continue
            all_numeric_nan_mask = df[numeric_cols].isna().all(axis=1)
            rows_to_fill = all_numeric_nan_mask
            if rows_to_fill.any():
                for col in numeric_cols:
                    df.loc[rows_to_fill, col] = AVERAGE_VALUES_TO_FILL[col]
                df.loc[rows_to_fill, "model"] = "10_year_average_filled"
                df.loc[rows_to_fill, "file_name"] = "10_year_average_filled"
            data_out[i] = df

    if len(data_out) == 1:
        data_out = data_out[0]

    return data_out


def _has_valid_data(data: list[pd.DataFrame] | pd.DataFrame) -> bool:
    """Whether a model returned at least one numeric value.

    Parameters
    ----------
    data : list[pd.DataFrame] | pd.DataFrame
        The data returned by a single model.

    Returns
    -------
    bool
        False if the data is empty or holds nothing but NaNs.
    """
    for df in data if isinstance(data, list) else [data]:
        if df.empty:
            continue
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and not df[numeric_cols].isna().all().all():
            return True

    return False


def _read_from_model(
    model: SWModel,
    start_time: datetime,
    end_time: datetime,
    historical_data_cutoff_time: datetime,
    reduce_ensemble: str,
    num_ensemble_members: int = 1,
    *,
    download: bool,
    do_interpolation: bool,
    swift_read_later: bool = False,
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
    historical_data_cutoff_time : datetime
        Represents "now". Used for defining boundaries for historical or forecast data.
    reduce_ensemble : str
        The method to reduce ensemble data (e.g., "mean"). If None, ensemble members are not reduced.
    num_ensemble_members : int, optional
        Number of ensemble members read so far. If this is larger than one, the ensemble size is
        already fixed and ENLIL has to match it by copying its run out to that many members.
    download : bool, optional
        Whether to download new data or not.
    do_interpolation : bool, optional
        If True, apply spline interpolation to short gaps (<= 3 hours) in historical data
    swift_read_later : bool, optional
        Whether SWIFT sits after this model in the order and so has not been read yet. Its
        ensemble size is then unknown, and ENLIL keeps to its latest run to stay combinable.

    Returns
    -------
    list[pd.DataFrame] | pd.DataFrame
        A single data frame or a list of data frames containing the model data.

    """
    # Read from historical models
    if isinstance(model, (DSCOVR, SWACE, SWOMNI, SWMIDL, SWIMAP)):
        data_one_model = _read_historical_model(
            model,
            start_time,
            end_time,
            historical_data_cutoff_time,
            download=download,
            do_interpolation=do_interpolation,
        )

    # Forecasting models are called with synthetic now time
    if isinstance(model, SWSWIFTEnsemble):
        data_one_model = _read_latest_ensemble_files(model, historical_data_cutoff_time, end_time)

        num_ens_members = len(data_one_model)

        if num_ens_members > 0 and reduce_ensemble is not None:
            data_one_model = _reduce_ensembles(data_one_model, reduce_ensemble)  # ty: ignore[invalid-argument-type]

    if isinstance(model, SWENLIL):
        data_one_model = _read_latest_enlil_run(
            model,
            start_time,
            end_time,
            historical_data_cutoff_time,
            download=download,
        )

        if len(data_one_model) > 0:
            # SWIFT owns the ensemble when it supplies one, and `num_ensemble_members` below
            # already reports that. Nothing is known about a SWIFT still to be read, though, so
            # ENLIL keeps to a single run rather than risk two ensembles that cannot be combined.
            if swift_read_later:
                data_one_model = [data_one_model[-1]]
            if reduce_ensemble is not None:
                data_one_model = [_reduce_ensembles(data_one_model, reduce_ensemble)]  # ty: ignore[invalid-argument-type]
            if num_ensemble_members > 1:
                data_one_model = [data_one_model[-1].copy() for _ in range(num_ensemble_members)]

    return data_one_model


def _read_historical_model(
    model: DSCOVR | SWACE | SWOMNI | SWMIDL | SWIMAP,
    start_time: datetime,
    end_time: datetime,
    historical_data_cutoff_time: datetime,
    *,
    download: bool,
    do_interpolation: bool,
) -> pd.DataFrame:
    """Reads SW data from historical models (DSCOVR, SWACE, SWOMNI, SWMIDL or SWIMAP) within the specified time range.

    Parameters
    ----------
    model : DSCOVR | SWACE | SWOMNI | SWMIDL | SWIMAP
        The historical model from which to read the data.
    start_time : datetime
        The start time of the data range.
    end_time : datetime
        The end time of the data range.
    historical_data_cutoff_time : datetime
        Represents "now". Data after this time is set to NaN.
    download : bool, optional
        Whether to download new data or not.
    do_interpolation : bool, optional
        If True, apply spline interpolation to short gaps (<= 3 hours) in historical data

    Returns
    -------
    pd.DataFrame
        A data frame containing the model data with future values (after historical_data_cutoff_time) set to NaN.

    Raises
    ------
    TypeError
        If the provided model is not an instance of DSCOVR, SWACE or SWOMNI.

    """
    logger.info(f"Reading {model.LABEL} from {start_time} to {end_time}")
    if isinstance(model, SWOMNI):
        data_one_model = model.read(start_time, end_time, download=download)
    else:
        data_one_model = model.read(start_time, end_time, download=download, propagation=True)

    # Create continuous index from start to end time
    continuous_index = pd.date_range(start=start_time, end=end_time, freq="1min", tz="UTC")

    # Always build the continuous, NaN-filled frame, even when the model returned zero rows
    continuous_df = pd.DataFrame(index=continuous_index)
    continuous_df.index.name = data_one_model.index.name
    for col in data_one_model.columns:
        if data_one_model[col].dtype in ["object", "str"]:
            continuous_df[col] = None
        else:
            continuous_df[col] = np.nan

    common_index = data_one_model.index.intersection(continuous_index)
    if len(common_index) > 0:
        for col in data_one_model.columns:
            continuous_df.loc[common_index, col] = data_one_model.loc[common_index, col]

    data_one_model = continuous_df

    historical_data = data_one_model.loc[:historical_data_cutoff_time]
    if not historical_data.empty:
        if do_interpolation:
            interpolated_historical = _interpolate_short_gaps(historical_data, max_gap_minutes=180)
            data_one_model.loc[:historical_data_cutoff_time] = interpolated_historical
            logger.info(
                f"Applied spline interpolation to short gaps (<= 3 hours) in {model.LABEL} historical data",
            )

    if historical_data_cutoff_time < end_time:
        data_one_model.loc[historical_data_cutoff_time + timedelta(minutes=1) : end_time] = np.nan
        logger.info(f"Setting NaNs in {model.LABEL} from {historical_data_cutoff_time} to {end_time}")

    return data_one_model


def _read_latest_ensemble_files(
    model: SWSWIFTEnsemble,
    historical_data_cutoff_time: datetime,
    end_time: datetime,
) -> list[pd.DataFrame]:
    """
    Reads the most recent SW ensemble data file available from the specified model.

    If the file for the target time is not found, the function iterates backward in hourly increments, up to 5 days, until a valid file is located.

    Parameters
    ----------
    model : SWSWIFTEnsemble
        The ensemble model from which to read the data.
    historical_data_cutoff_time : datetime
        Represents "now". The function starts searching for files from this time.
    end_time : datetime
        The end time of the data range.

    Returns
    -------
    list[pd.DataFrame]
        A list of data frames containing ensemble data for the specified range.
        Returns empty list if no data is available.
    """
    # Only try to read SWIFT data if historical cutoff is before end time
    if historical_data_cutoff_time >= end_time:
        return []

    target_time = min(historical_data_cutoff_time, end_time)
    data_one_model = []

    while target_time > (historical_data_cutoff_time - timedelta(days=5)) and target_time < end_time:
        try:
            data_one_model = model.read(target_time, end_time)
        except Exception as e:
            logger.warning(f"Failed to read SWIFT ensemble for {target_time}: {e}")
            target_time -= timedelta(days=1)
            continue

        if len(data_one_model) == 0:
            target_time -= timedelta(days=1)
            continue

        elif len(data_one_model) == 1:
            target_time -= timedelta(days=1)
            continue

        logger.info(f"SWIFT ends at {data_one_model[0].index[-1]}")

        data_one_model = _interpolate_to_common_indices(
            target_time, end_time, historical_data_cutoff_time, data_one_model
        )
        break

    if len(data_one_model) > 0:
        logger.info(f"Reading SWIFT ensemble from {target_time} to {end_time}")
    else:
        logger.info("No SWIFT ensemble data available for the requested time range")

    return data_one_model


def _read_latest_enlil_run(
    model: SWENLIL,
    start_time: datetime,
    end_time: datetime,
    historical_data_cutoff_time: datetime,
    *,
    download: bool,
) -> list[pd.DataFrame]:
    """
    Read the most recent ENLIL run available at `historical_data_cutoff_time`.

    A run is keyed by its date, so the search starts at the date of `historical_data_cutoff_time`
    and walks backwards a day at a time until runs of `model`'s own mode are found. Finding none
    leaves the forecast window to the next model in the order, which is how a `SWENLIL_BKG` placed
    after a `SWENLIL_CME` takes over on days that carry no CME run.

    Parameters
    ----------
    model : SWENLIL
        The ENLIL model to read from, either :class:`SWENLIL_CME` or :class:`SWENLIL_BKG`.
    start_time : datetime
        Start time of the data request, which fixes the one minute grid the data is put on.
    end_time : datetime
        End time of the data range.
    historical_data_cutoff_time : datetime
        Represents "now". ENLIL is a forecast, so nothing before this time is used.
    download : bool
        Whether to download new data or not.

    Returns
    -------
    list[pd.DataFrame]
        One data frame per run of the selected date, on a common one minute index.
        Empty if no run covers the requested forecast window.
    """
    if historical_data_cutoff_time >= end_time:
        return []

    run_date = historical_data_cutoff_time.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

    for _ in range(ENLIL_MAX_LOOKBACK_DAYS):
        read_runs = model.read(run_date, end_time, download=download)
        runs: list[pd.DataFrame] = read_runs if isinstance(read_runs, list) else [read_runs]
        runs = [df for df in runs if not df.empty]

        if runs:
            logger.info(f"Reading {len(runs)} ENLIL {model.MODE} run(s) of {run_date.date()} up to {end_time}")
            return _interpolate_enlil_to_common_index(runs, start_time, end_time, historical_data_cutoff_time)

        run_date -= timedelta(days=1)

    logger.info(f"No ENLIL {model.MODE} data available for the requested time range")
    return []


def _interpolate_enlil_to_common_index(
    data: list[pd.DataFrame],
    start_time: datetime,
    end_time: datetime,
    historical_data_cutoff_time: datetime,
) -> list[pd.DataFrame]:
    """
    Interpolate ENLIL runs onto the shared one minute grid of the request.

    ENLIL timestamps follow the model's own irregular cadence, so without this the data would
    not line up with the grid the other models are put on. All runs share one index so they can
    be reduced against each other.

    Parameters
    ----------
    data : list[pd.DataFrame]
        The runs to interpolate.
    start_time : datetime
        Start time of the data request, which fixes the phase of the one minute grid.
    end_time : datetime
        End time of the data request.
    historical_data_cutoff_time : datetime
        Represents "now". Data before this time is dropped.

    Returns
    -------
    list[pd.DataFrame]
        The interpolated runs, or an empty list if no run covers the forecast window.
    """
    grid = pd.date_range(start=start_time, end=end_time, freq="1min", tz="UTC")
    lower = max([historical_data_cutoff_time, *(df.index[0] for df in data)])
    upper = min([end_time, *(df.index[-1] for df in data)])
    common_index = grid[(grid >= lower) & (grid <= upper)]

    if len(common_index) == 0:
        logger.info("ENLIL runs do not overlap the requested forecast window")
        return []

    interpolated = []

    for df in data:
        df_common_index = pd.DataFrame(index=common_index)
        df_common_index.index.name = df.index.name

        for colname, col in df.items():
            if col.dtype in ["object", "str"]:
                # This is the filename column. It is read back from CSV as a string dtype, but the
                # other models carry theirs as object (SWIFT stores Path), and a model merged in
                # later has to be able to write its own file names into this same column.
                df_common_index[colname] = pd.Series(col.iloc[0], index=common_index, dtype=object)
            else:
                df_common_index[colname] = np.interp(common_index, df.index, col)

        interpolated.append(df_common_index)

    logger.info(f"ENLIL ends at {common_index[-1]}")

    return interpolated


def _extend_to_full_index(
    data_out: list[pd.DataFrame],
    start_time: datetime,
    end_time: datetime,
) -> list[pd.DataFrame]:
    """
    Extend data frames to the full one minute grid of the request.

    A model can only fill rows that already exist in the frame it is merged into, so a forecast
    model stopping short of `end_time` would otherwise prevent later models from covering the rest.

    Parameters
    ----------
    data_out : list[pd.DataFrame]
        The data frames read so far.
    start_time : datetime
        Start time of the data request.
    end_time : datetime
        End time of the data request.

    Returns
    -------
    list[pd.DataFrame]
        The data frames, each spanning at least the full requested period.
    """
    full_index = pd.date_range(start=start_time, end=end_time, freq="1min", tz="UTC")

    for i, df in enumerate(data_out):
        if df.empty or full_index.difference(df.index).empty:
            continue
        data_out[i] = df.reindex(df.index.union(full_index))

    return data_out


def _interpolate_to_common_indices(
    target_time: datetime,
    end_time: datetime,
    historical_data_cutoff_time: datetime,
    data: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    """
    Interpolate the data to a common index with a 1-minute frequency.

    Parameters
    ----------
    target_time : datetime
        The start time for the interpolation.
    end_time : datetime
        The end time for the interpolation.
    historical_data_cutoff_time : datetime
        The "now" time, used for truncating data after interpolation.
    data : list[pd.DataFrame]
        The list of data frames to interpolate.

    Returns
    -------
    list[pd.DataFrame]
        The list of interpolated data frames with a common index.
    """

    data_final_index = min(df.index[-1] for df in data if not df.empty)
    for ie, _ in enumerate(data):
        df_common_index = pd.DataFrame(
            index=pd.date_range(
                datetime(
                    target_time.year,
                    target_time.month,
                    target_time.day,
                    tzinfo=timezone.utc,
                ),
                datetime(
                    end_time.year,
                    end_time.month,
                    end_time.day,
                    23,
                    59,
                    59,
                    tzinfo=timezone.utc,
                ),
                freq=timedelta(minutes=1),
                tz="UTC",
            ),
        )
        df_common_index.index.name = data[ie].index.name

        for colname, col in data[ie].items():
            if col.dtype in ["object", "str"]:
                # this is the filename column
                df_common_index[colname] = col.iloc[0]
            else:
                df_common_index[colname] = np.interp(df_common_index.index, data[ie].index, col)
        logger.info(f"Post interpolation SWIFT ends at {data_final_index}")
        data[ie] = df_common_index
        data[ie] = data[ie].truncate(
            before=historical_data_cutoff_time - timedelta(minutes=0.999999),
            after=data_final_index + timedelta(minutes=0.999999),
        )

    return data


def _interpolate_short_gaps(df: pd.DataFrame, max_gap_minutes: int = 180) -> pd.DataFrame:
    """
    Interpolate short gaps in historical data using spline interpolation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with potential gaps
    max_gap_minutes : int, optional
        Maximum gap size in minutes to interpolate (default 180 = 3 hours)

    Returns
    -------
    pd.DataFrame
        Dataframe with short gaps interpolated
    """
    if df.empty:
        return df

    df_interpolated = df.copy()
    interpolated_indices = set()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df_interpolated[col]

        is_nan = series.isna()
        nan_groups = (is_nan != is_nan.shift()).cumsum()

        for group_id in nan_groups[is_nan].unique():
            nan_mask = (nan_groups == group_id) & is_nan
            gap_size = nan_mask.sum()

            if gap_size <= max_gap_minutes:
                # Get indices of the gap
                gap_start_idx = nan_mask.idxmax()
                gap_end_idx = nan_mask[::-1].idxmax()

                # Find valid data points around the gap for interpolation
                valid_before = series.loc[:gap_start_idx].dropna()
                valid_after = series.loc[gap_end_idx:].dropna()

                # Need at least 2 points before and after for spline interpolation
                if len(valid_before) >= 2 and len(valid_after) >= 2:
                    # Take last 10 points before and first 10 points after for context
                    context_before = valid_before.tail(10)
                    context_after = valid_after.head(10)

                    x_context = np.concatenate(
                        [
                            np.arange(len(context_before)) - len(context_before),
                            np.arange(gap_size) + 1,
                            np.arange(len(context_after)) + gap_size + 1,
                        ]
                    )
                    y_context = np.concatenate(
                        [
                            context_before.values,
                            np.full(gap_size, np.nan),
                            context_after.values,
                        ]
                    )

                    valid_mask = ~np.isnan(y_context)
                    if np.sum(valid_mask) >= 4:  # Need at least 4 points for spline
                        try:
                            spline = UnivariateSpline(
                                x_context[valid_mask],
                                y_context[valid_mask],
                                s=0,
                                k=min(3, np.sum(valid_mask) - 1),
                            )
                            gap_x = np.arange(gap_size) + 1
                            interpolated_values = spline(gap_x)
                            df_interpolated.loc[nan_mask, col] = np.round(interpolated_values, 1)
                            interpolated_indices.update(df_interpolated.index[nan_mask])

                        except Exception as e:
                            logger.warning(f"Spline interpolation failed for column {col}: {e}")
                            interpolated_mask = df_interpolated[col].isna() & nan_mask
                            df_interpolated.loc[interpolated_mask, col] = df_interpolated[col].interpolate(
                                method="linear"
                            )[interpolated_mask]
                            interpolated_indices.update(df_interpolated.index[interpolated_mask])

    # Mark interpolated values in file_name and model columns
    if interpolated_indices:
        df_interpolated.loc[list(interpolated_indices), "file_name"] = "interpolated"
        df_interpolated.loc[list(interpolated_indices), "model"] = "interpolated"

    return df_interpolated


def _ensure_continuous_dataframe(
    data_out: list[pd.DataFrame],
    start_time: datetime,
    end_time: datetime,
    historical_data_cutoff_time: datetime,
    forecast_data_available: bool,
    truncate: bool = True,
) -> list[pd.DataFrame]:
    """
    Ensure the dataframe is continuous from start to end time, handling gaps and forecast unavailability.

    Parameters
    ----------
    data_out : list[pd.DataFrame]
        The current data frames
    start_time : datetime
        Start time of the data request
    end_time : datetime
        End time of the data request
    historical_data_cutoff_time : datetime
        Time representing "now"
    forecast_data_available : bool
        Whether forecast data (SWIFT or ENLIL) is available for future dates

    Returns
    -------
    list[pd.DataFrame]
        Continuous data frames with proper NaN filling
    """
    if not data_out or all(df.empty for df in data_out):
        return data_out

    future_data_all_nan = False
    if historical_data_cutoff_time < end_time:
        for df in data_out:
            if not df.empty:
                future_data = df.loc[historical_data_cutoff_time:end_time]
                if not future_data.empty:
                    numeric_cols = future_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        future_data_all_nan = future_data[numeric_cols].isna().all().all()
                    break

    # Determine actual end time based on forecast availability
    if ((not forecast_data_available or future_data_all_nan) and (historical_data_cutoff_time < end_time)) and truncate:
        actual_end_time = historical_data_cutoff_time
        logger.info(
            f"Since no forecast model is available for future dates, final dataframe truncated to {historical_data_cutoff_time}"
        )
    else:
        actual_end_time = end_time

    continuous_index = pd.date_range(start=start_time, end=actual_end_time, freq="1min", tz="UTC")

    for i, df in enumerate(data_out):
        if df.empty:
            continue

        continuous_df = pd.DataFrame(index=continuous_index)
        continuous_df.index.name = df.index.name

        for col in df.columns:
            if df[col].dtype in ["object", "str"]:
                continuous_df[col] = None
            else:
                continuous_df[col] = np.nan

        # Fill in the available data
        common_index = df.index.intersection(continuous_index)
        if len(common_index) > 0:
            for col in df.columns:
                continuous_df.loc[common_index, col] = df.loc[common_index, col]

        data_out[i] = continuous_df

    return data_out


def _reduce_ensembles(data_ensembles: list[pd.DataFrame], method: Literal["mean", "median"]) -> pd.DataFrame:
    """Reduce a list of data frames representing ensemble data to a single data frame using the provided method.

    Parameters
    ----------
    data_ensembles : list[pd.DataFrame]
        List of dataframes to reduce.
    method : Literal["mean", "median"]
        Method to reduce the ensembles.

    Returns
    -------
    pd.DataFrame
        Reduced data frame.

    Raises
    ------
    ValueError
        If the method is not recognized.
    ValueError
        If the data frames have different lengths.
    """

    lengths = [len(d) for d in data_ensembles if not d.empty]
    if len(set(lengths)) > 1:
        raise ValueError("Ensemble data frames have different lengths, cannot reduce.")

    combined_df = pd.concat(data_ensembles)
    if method == "mean":
        reduced_df = combined_df.groupby(combined_df.index).mean(numeric_only=True)
    elif method == "median":
        reduced_df = combined_df.groupby(combined_df.index).median(numeric_only=True)

    # For non-numeric columns, take the first non-null value
    non_numeric_cols = combined_df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        reduced_df[col] = combined_df[col].groupby(combined_df.index).first()

    reduced_df = reduced_df.sort_index()
    return reduced_df
