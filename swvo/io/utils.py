# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


def any_nans(data: list[pd.DataFrame] | pd.DataFrame) -> bool:
    """Calculate if a list of data frames contains any nans.

    Parameters
    ----------
    data : list[pd.DataFrame] | pd.DataFrame
        Data frame or list of data frames to process

    Returns
    -------
    bool
        Bool if any data frame of the list contains any nan values
    """
    if isinstance(data, list):
        for df in data:
            _ = nan_percentage(df)

    if isinstance(data, pd.DataFrame):
        _ = nan_percentage(data)

    return any((df.isna().any(axis=None) > 0) for df in data)


def nan_percentage(data: pd.DataFrame) -> float:
    """Calculate the percentage of NaN values in the data column of data frame and log it.

    Parameters
    ----------
    data : pd.DataFrame
        The data frame to process

    Returns
    -------
    float
        Nan percentage in the data frame
    """
    float_columns = data.select_dtypes(include=["float64", "float32"]).columns
    nan_percentage = (data[float_columns].isna().sum().sum() / (data.shape[0])) * 100
    logger.info(f"Percentage of NaNs in data frame: {nan_percentage:.2f}%")

    return nan_percentage


def construct_updated_data_frame(
    data: list[pd.DataFrame] | pd.DataFrame,
    data_one_model: list[pd.DataFrame] | pd.DataFrame,
    model_label: str,
) -> list[pd.DataFrame]:
    """
    Construct an updated data frame providing the previous data frame and the data frame of the current model call.

    Also adds the model label to the data frame.
    Parameters
    ----------
    data : list[pd.DataFrame] | pd.DataFrame
        The data frame or list of data frames to update.
    data_one_model : list[pd.DataFrame] | pd.DataFrame
        The data frame or list of data frames from the current model call.
    model_label : str
        The label of the model to add to the data frame.

    Returns
    -------
    list[pd.DataFrame]
        The updated data frame or list of data frames with the model label added.
    """
    if isinstance(data_one_model, list) and data_one_model == []:  # nothing to update
        return data

    if isinstance(data_one_model, pd.DataFrame):
        data_one_model = [data_one_model]

    if isinstance(data, pd.DataFrame):
        data = [data]

    # extend the data we have read so far to match the new ensemble numbers
    if len(data) == 1 and len(data_one_model) > 1:
        data = data * len(data_one_model)
    elif len(data) != len(data_one_model):
        msg = f"Tried to combine models with different ensemble numbers: {len(data)} and {len(data_one_model)}!"
        raise ValueError(msg)

    for i, _ in enumerate(data_one_model):
        if "model" in data_one_model[i].columns:
            mask_not_interpolated = data_one_model[i]["model"] != "interpolated"
            data_one_model[i].loc[mask_not_interpolated, "model"] = model_label
            mask_nan = data_one_model[i].isna().any(axis=1)
            data_one_model[i].loc[mask_nan, "model"] = None
        else:
            data_one_model[i]["model"] = model_label
            data_one_model[i].loc[data_one_model[i].isna().any(axis=1), "model"] = None
        if "file_name" in data_one_model[i].columns:
            data_one_model[i].loc[data_one_model[i]["file_name"].notna(), "model"] = model_label
            data_one_model[i].loc[data_one_model[i]["file_name"].isna(), "model"] = None
        if data[i].empty:
            data[i] = data_one_model[i]
        empty_idx = data[i].index[data[i].isna().all(axis=1)]

        data[i].loc[empty_idx] = (
            data[i].loc[empty_idx].combine_first(data_one_model[i].reindex(data[i].index).loc[empty_idx])
        )

    return data


def datenum(
    date_input: datetime | int,
    month: Optional[int] = None,
    year: Optional[int] = None,
    hour: int = 0,
    minute: int = 0,
    seconds: int = 0,
) -> float:
    """Convert a date to a MATLAB serial date number.

    Parameters
    ----------
    date_input : datetime | int
        A datetime object or an integer representing the day of the month.
    month : int, optional
        The month of the date. Required if date_input is an integer.
    year : int, optional
        The year of the date. Required if date_input is an integer.
    hour : int
        The hour of the date, by default 0
    minute : int
        The minute of the date, by default 0
    seconds : int
        The seconds of the date, by default 0

    Returns
    -------
    float
        The MATLAB serial date number.

    Raises
    ------
    ValueError
        If the input is invalid, i.e., if date_input is an integer and month or year is not provided.
    """
    MATLAB_EPOCH = datetime.toordinal(datetime(1970, 1, 1, tzinfo=timezone.utc)) + 366

    if isinstance(date_input, datetime):
        dt = date_input.replace(tzinfo=timezone.utc)
    elif month is not None and year is not None:
        dt = datetime(
            year=year,
            month=month,
            day=date_input,
            hour=hour,
            minute=minute,
            second=seconds,
        ).replace(tzinfo=timezone.utc)
    else:
        raise ValueError("Invalid input. Provide either a datetime object or year, month, and day.")

    return dt.timestamp() / 86400 + MATLAB_EPOCH


def datestr(datenum: float) -> str:
    """
    Convert MATLAB datenum to a formatted date string.

    Parameters
    ----------
    datenum : float
        The MATLAB datenum to convert.

    Returns
    -------
    str
        The formatted date string in the format "YYYYMMDDHHMM00".
    """

    MATLAB_EPOCH = datetime.toordinal(datetime(1970, 1, 1, tzinfo=timezone.utc)) + 366
    unix_days = datenum - MATLAB_EPOCH
    unix_timestamp = unix_days * 86400

    dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
    formatted_date = dt.strftime("%Y%m%d%H%M")

    return formatted_date


def sw_mag_propagation(sw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Propagate the solar wind magnetic field to the bow shock and magnetopause.

    Parameters
    ----------
    sw_data : pd.DataFrame
        Data frame containing solar wind data with a 'speed' column.

    Returns
    -------
    pd.DataFrame
        Data frame with propagated solar wind data, indexed by time.
    """

    sw_data["t"] = [t.timestamp() for t in sw_data.index.to_pydatetime()]
    sw_data = sw_data.dropna(how="any")

    distance = 1.5e6
    shifted_time = distance / sw_data["speed"]

    shifted_time_smooth = gaussian_filter1d(np.array(shifted_time.values, dtype=np.float64), sigma=5)
    new_time_smooth = sw_data["t"] + shifted_time_smooth

    stdate = sw_data["t"].min()
    endate = new_time_smooth.max()

    full_time_range = pd.date_range(
        pd.to_datetime(sw_data["t"].min(), unit="s", utc=True).floor("min"),
        pd.to_datetime(new_time_smooth.max(), unit="s", utc=True).floor("min"),
        freq="1min",
        tz="UTC",
    )

    valid = (new_time_smooth >= stdate) & (new_time_smooth <= endate)
    sw_data = sw_data[valid]
    new_time_smooth = new_time_smooth[valid]
    valid = np.diff(new_time_smooth, prepend=new_time_smooth.iloc[0]) > 0
    sw_data = sw_data[valid]
    new_time_smooth = new_time_smooth[valid]

    sw_data["t"] = new_time_smooth
    sw_data = sw_data.dropna()
    sw_data["t"] = pd.to_datetime(sw_data["t"], unit="s", utc=True)

    sw_data.index = sw_data["t"]
    sw_data.index = sw_data.index.round("min")
    sw_data = sw_data[~sw_data.index.duplicated(keep="first")]
    sw_data = sw_data.reindex(full_time_range)
    sw_data = sw_data.drop(columns=["t"])

    return sw_data
