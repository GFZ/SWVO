# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

"""Function to read Solar Wind from multiple models."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from swvo.io.exceptions import ModelError
from swvo.io.solar_wind import DSCOVR, SWACE, SWOMNI, SWSWIFTEnsemble
from swvo.io.utils import any_nans, construct_updated_data_frame

SWModel = DSCOVR | SWACE | SWOMNI | SWSWIFTEnsemble

logging.captureWarnings(True)


def read_solar_wind_from_multiple_models(  # noqa: PLR0913
    start_time: datetime,
    end_time: datetime,
    model_order: Sequence[SWModel] | None = None,
    reduce_ensemble: str | None = None,
    historical_data_cutoff_time: datetime | None = None,
    *,
    synthetic_now_time: datetime | None = None,  # deprecated
    download: bool = False,
    recurrence: bool = False,
    rec_model_order: list[DSCOVR | SWACE | SWOMNI] | None = None,
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
    historical_data_cutoff_time : datetime, optional
        Time which represents "now". After this time, no data will be taken from historical models (OMNI, ACE). Defaults to None.
    download : bool, optional
        Flag which decides whether new data should be downloaded. Defaults to False.
        Also applies to recurrence filling.
    recurrence : bool, optional
        If True, fill missing values using 27-day recurrence from historical models (OMNI, ACE, SWIFT).
        Defaults to False.
    rec_model_order : list[DSCOVR | SWACE | SWOMNI], optional
        The order in which historical models will be used for 27-day recurrence filling.
        Defaults to [DSCOVR, SWACE, SWOMNI].

    Returns
    -------
    Union[:class:`pandas.DataFrame`, list[:class:`pandas.DataFrame`]]
        A data frame or a list of data frames containing data for the requested period.
    """
    if synthetic_now_time is not None:
        warnings.warn(
            "`synthetic_now_time` is deprecated and will be removed in a future version. "
            "Use `historical_data_cutoff_time` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if historical_data_cutoff_time is None:
            historical_data_cutoff_time = synthetic_now_time

    if historical_data_cutoff_time is None:
        historical_data_cutoff_time = min(datetime.now(timezone.utc), end_time)

    if model_order is None:
        model_order = [SWOMNI(), DSCOVR(), SWACE(), SWSWIFTEnsemble()]
        logging.warning("No model order specified, using default order: OMNI, ACE, SWIFT ensemble")

    data_out = [pd.DataFrame()]
    swift_data_available = True

    for model in model_order:
        if not isinstance(model, SWModel):
            raise ModelError(f"Unknown or incompatible model: {type(model).__name__}")
        data_one_model = _read_from_model(
            model,
            start_time,
            end_time,
            historical_data_cutoff_time,
            reduce_ensemble,
            download=download,
        )

        # Check if SWIFT ensemble returned empty data
        if isinstance(model, SWSWIFTEnsemble):
            if (
                not data_one_model
                or (isinstance(data_one_model, list) and len(data_one_model) == 0)
                or (isinstance(data_one_model, pd.DataFrame) and data_one_model.empty)
            ):
                swift_data_available = False
                logging.info("SWIFT ensemble data not available for future dates")
            else:
                # Check if SWIFT data is all NaN
                swift_has_valid_data = False
                if isinstance(data_one_model, list):
                    for df in data_one_model:
                        if not df.empty:
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0 and not df[numeric_cols].isna().all().all():
                                swift_has_valid_data = True
                                break
                elif isinstance(data_one_model, pd.DataFrame) and not data_one_model.empty:
                    numeric_cols = data_one_model.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0 and not data_one_model[numeric_cols].isna().all().all():
                        swift_has_valid_data = True

                if not swift_has_valid_data:
                    swift_data_available = False
                    logging.info("SWIFT ensemble data contains only NaN values for future dates")

        data_out = construct_updated_data_frame(data_out, data_one_model, model.LABEL)
        if not any_nans(data_out):
            break

    # Apply 27-day recurrence if requested

    if recurrence:
        if rec_model_order is None:
            rec_model_order = [m for m in model_order if isinstance(m, (DSCOVR, SWACE, SWOMNI))]
        for i, df in enumerate(data_out):
            if not df.empty:
                data_out[i] = _recursive_fill_27d_historical(df, download, rec_model_order)

    # Ensure continuous dataframe and handle SWIFT unavailability
    data_out = _ensure_continuous_dataframe(
        data_out,
        start_time,
        end_time,
        historical_data_cutoff_time,
        swift_data_available,
    )

    if len(data_out) == 1:
        data_out = data_out[0]
        _set_interpolated_flags(data_out, label="interpolated")

    else:
        for df in data_out:
            _set_interpolated_flags(df, label="interpolated")

    return data_out


def _set_interpolated_flags(df: pd.DataFrame, label: str) -> None:
    """
    Set appropriate flags in the 'model' column for interpolated data points.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame to process.
    label : str
        The label to set for interpolated data points.

    Returns
    -------
    None
        The function modifies the data frames in place.
    """
    if not df.empty and "file_name" in df.columns:
        # Handle interpolated data
        interpolated_indices = df.index[df["file_name"] == label]
        df.loc[interpolated_indices, "model"] = label


def _read_from_model(  # noqa: PLR0913
    model: SWModel,
    start_time: datetime,
    end_time: datetime,
    historical_data_cutoff_time: datetime,
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
    if isinstance(model, (DSCOVR, SWACE, SWOMNI)):
        data_one_model = _read_historical_model(
            model,
            start_time,
            end_time,
            historical_data_cutoff_time,
            download=download,
        )

    # Forecasting models are called with synthetic now time
    if isinstance(model, SWSWIFTEnsemble):
        data_one_model = _read_latest_ensemble_files(model, historical_data_cutoff_time, end_time)

        num_ens_members = len(data_one_model)

        if num_ens_members > 0 and reduce_ensemble is not None:
            data_one_model = _reduce_ensembles(data_one_model, reduce_ensemble)

    return data_one_model


def _read_historical_model(
    model: DSCOVR | SWACE | SWOMNI,
    start_time: datetime,
    end_time: datetime,
    historical_data_cutoff_time: datetime,
    *,
    download: bool,
) -> pd.DataFrame:
    """Reads SW data from historical models (DSCOVR, SWACE or SWOMNI) within the specified time range.

    Parameters
    ----------
    model : DSCOVR | SWACE | SWOMNI
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
        If the provided model is not an instance of DSCOVR, SWACE or SWOMNI.

    """
    logging.info(f"Reading {model.LABEL} from {start_time} to {end_time}")
    if isinstance(model, SWOMNI):
        data_one_model = model.read(start_time, end_time, download=download)
    else:
        data_one_model = model.read(start_time, end_time, download=download, propagation=True)

    # Create continuous index from start to end time
    continuous_index = pd.date_range(start=start_time, end=end_time, freq="1min", tz="UTC")

    if not data_one_model.empty:
        continuous_df = pd.DataFrame(index=continuous_index)
        continuous_df.index.name = data_one_model.index.name
        for col in data_one_model.columns:
            if data_one_model[col].dtype == "object":
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
            interpolated_historical = _interpolate_short_gaps(historical_data, max_gap_minutes=180)
            data_one_model.loc[:historical_data_cutoff_time] = interpolated_historical
            logging.info(
                f"Applied spline interpolation to short gaps (<= 3 hours) in {model.LABEL} historical data",
            )

        if historical_data_cutoff_time < end_time:
            data_one_model.loc[historical_data_cutoff_time + timedelta(minutes=1) : end_time] = np.nan
            logging.info(f"Setting NaNs in {model.LABEL} from {historical_data_cutoff_time} to {end_time}")

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

    while target_time > (historical_data_cutoff_time - timedelta(days=5)):
        try:
            data_one_model = model.read(target_time, end_time)
        except Exception as e:
            logging.warning(f"Failed to read SWIFT ensemble for {target_time}: {e}")
            target_time -= timedelta(days=1)
            continue

        if len(data_one_model) == 0:
            target_time -= timedelta(days=1)
            continue

        data_one_model = _interpolate_to_common_indices(
            target_time, end_time, historical_data_cutoff_time, data_one_model
        )
        break

    if len(data_one_model) > 0:
        logging.info(f"Reading SWIFT ensemble from {target_time} to {end_time}")
    else:
        logging.info("No SWIFT ensemble data available for the requested time range")

    return data_one_model


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
            if col.dtype == "object":
                # this is the filename column
                df_common_index[colname] = col.iloc[0]
            else:
                df_common_index[colname] = np.interp(df_common_index.index, data[ie].index, col)

        data[ie] = df_common_index
        data[ie] = data[ie].truncate(
            before=historical_data_cutoff_time - timedelta(minutes=0.999999),
            after=end_time + timedelta(minutes=0.999999),
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
                            logging.warning(f"Spline interpolation failed for column {col}: {e}")
                            interpolated_mask = df_interpolated[col].isna() & nan_mask
                            df_interpolated.loc[interpolated_mask, col] = df_interpolated[col].interpolate(
                                method="linear"
                            )[interpolated_mask]
                            interpolated_indices.update(df_interpolated.index[interpolated_mask])

    # Mark interpolated values in file_name and model columns
    if interpolated_indices:
        df_interpolated.loc[list(interpolated_indices), "file_name"] = "interpolated"

    return df_interpolated


def _ensure_continuous_dataframe(
    data_out: list[pd.DataFrame],
    start_time: datetime,
    end_time: datetime,
    historical_data_cutoff_time: datetime,
    swift_data_available: bool,
) -> list[pd.DataFrame]:
    """
    Ensure the dataframe is continuous from start to end time, handling gaps and SWIFT unavailability.

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
    swift_data_available : bool
        Whether SWIFT data is available for future dates

    Returns
    -------
    list[pd.DataFrame]
        Continuous data frames with proper NaN filling
    """
    if not data_out or all(df.empty for df in data_out):
        return data_out

    swift_data_all_nan = False
    if historical_data_cutoff_time < end_time:
        for df in data_out:
            if not df.empty:
                future_data = df.loc[historical_data_cutoff_time:end_time]
                if not future_data.empty:
                    numeric_cols = future_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        swift_data_all_nan = future_data[numeric_cols].isna().all().all()
                    break

    # Determine actual end time based on SWIFT availability
    if (not swift_data_available or swift_data_all_nan) and historical_data_cutoff_time < end_time:
        actual_end_time = historical_data_cutoff_time
        logging.info(
            f"Since SWIFT is not available for future dates, final dataframe truncated to {historical_data_cutoff_time}"
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
            if df[col].dtype == "object":
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


def _recursive_fill_27d_historical(
    df: pd.DataFrame, download: bool, historical_models: list[DSCOVR | SWACE | SWOMNI]
) -> pd.DataFrame:
    """Fill missing values using historical models for (`date` - 27 days).

    For continuous missing data blocks, copies the entire corresponding 27-day-back block.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to fill with gaps.
    download : bool
        Download new data or not.
    historical_models : list[DSCOVR | SWACE | SWOMNI]
        List of historical models to use for filling gaps.

    Returns
    -------
    pd.DataFrame
        DataFrame with gaps filled using 27d recurrence.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    value_cols = [col for col in numeric_cols if col not in ["file_name", "model"]]

    if not value_cols:
        return df

    # Find continuous blocks of missing data
    missing_mask = df[value_cols].isna().all(axis=1)

    if not missing_mask.any():
        return df

    # continuous blocks of missing data
    missing_blocks = []
    in_block = False
    block_start = None

    for idx in df.index:
        if missing_mask[idx]:
            if not in_block:
                block_start = idx
                in_block = True
        else:
            if in_block:
                missing_blocks.append((block_start, idx - timedelta(minutes=1)))
                in_block = False

    if in_block:
        missing_blocks.append((block_start, df.index[-1]))

    for block_start, block_end in missing_blocks:
        # Calculate 27-day-back period
        recurrence_start = block_start - timedelta(days=27)
        recurrence_end = block_end - timedelta(days=27)

        filled = False
        for model in historical_models:
            try:
                prev_data = model.read(
                    recurrence_start - timedelta(hours=1),
                    recurrence_end + timedelta(hours=1),
                    download=download,
                    propagation=True if not isinstance(model, SWOMNI) else False,
                )

                if prev_data.empty:
                    continue

                # Check if we have data for the recurrence period
                recurrence_mask = (prev_data.index >= recurrence_start) & (prev_data.index <= recurrence_end)
                recurrence_data = prev_data[recurrence_mask]

                if recurrence_data.empty:
                    continue

                # Check if recurrence data has valid values (not all NaN)
                has_valid_data = False
                for col in value_cols:
                    if col in recurrence_data.columns and not recurrence_data[col].isna().all():
                        has_valid_data = True
                        break

                if not has_valid_data:
                    continue

                current_block_mask = (df.index >= block_start) & (df.index <= block_end)

                for current_idx in df.index[current_block_mask]:
                    recurrence_idx = current_idx - timedelta(days=27)

                    if recurrence_idx in recurrence_data.index:
                        for col in value_cols:
                            if col in recurrence_data.columns and not pd.isna(recurrence_data.loc[recurrence_idx, col]):
                                df.loc[current_idx, col] = recurrence_data.loc[recurrence_idx, col]

                        # if all the numeric columns are still NaN, skip setting model and file_name
                        if df.loc[current_idx, value_cols].isna().all():
                            continue
                        df.loc[current_idx, "model"] = f"{model.LABEL}_recurrence_27d"
                        original_fname = recurrence_data.loc[recurrence_idx].get("file_name", "recurrence_27d")
                        df.loc[current_idx, "file_name"] = f"{original_fname}_recurrence_27d"

                filled = True
                logging.info(f"Filled missing block {block_start} to {block_end} using {model.LABEL} 27d recurrence")
                break

            except Exception as e:
                logging.warning(f"Failed to read {model.LABEL} for 27d recurrence block {block_start}-{block_end}: {e}")
                continue

        if not filled:
            logging.warning(f"Could not fill missing block {block_start} to {block_end} with 27d recurrence")

    return df


def _reduce_ensembles(data_ensembles: list[pd.DataFrame], method: Literal["mean"]) -> pd.DataFrame:
    """Reduce a list of data frames representing ensemble data to a single data frame using the provided method."""
    msg = "This reduction method has not been implemented yet!"
    raise NotImplementedError(msg)
