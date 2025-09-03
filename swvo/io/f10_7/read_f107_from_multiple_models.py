# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from swvo.io.f10_7 import F107OMNI, F107SWPC
from swvo.io.utils import any_nans, construct_updated_data_frame

F107Model = F107OMNI | F107SWPC

logging.captureWarnings(True)


def read_f107_from_multiple_models(
    start_time: datetime,
    end_time: datetime,
    model_order: list[F107Model] | None = None,
    historical_data_cutoff_time: datetime | None = None,
    *,
    synthetic_now_time: datetime | None = None,  # deprecated
    download: bool = False,
) -> pd.DataFrame:
    """
    Read F107 data from multiple models.

    The model order represents the priorities of models. The first model in the
    model order is read. If there are still NaNs in the resulting data, the next
    model will be read, and so on. For ensemble predictions, a list will be
    returned; otherwise, a plain data frame will be returned.

    Parameters
    ----------
    start_time : datetime
        Start time of the data request.
    end_time : datetime
        End time of the data request.
    model_order : list or None, optional
        Order in which data will be read from the models. Defaults to [OMNI, SWPC].
    historical_data_cutoff_time : datetime or None, optional
        Time representing "now". After this time, no data will be taken from
        historical models (OMNI, SWPC). Defaults to None.
    download : bool, optional
        Flag indicating whether new data should be downloaded. Defaults to False.

    Returns
    -------
    :class:`pandas.DataFrame`
        A data frame containing data for the requested
        period.
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
        model_order = [F107OMNI(), F107SWPC()]
        logging.warning("No model order specified, using default order: OMNI, SWPC")

    data_out = pd.DataFrame()

    for model in model_order:
        if isinstance(model, F107OMNI):
            logging.info(f"Reading {model.LABEL} from {start_time} to {end_time}")
            data_one_model = model.read(start_time, end_time, download=download)

            index_range = pd.date_range(start=historical_data_cutoff_time, end=end_time, freq="d")
            data_one_model = data_one_model.reindex(data_one_model.index.union(index_range))

            data_one_model.loc[historical_data_cutoff_time:end_time, "f107"] = np.nan
            data_one_model = data_one_model.fillna({"file_name": np.nan})
            logging.info(f"Setting NaNs in {model.LABEL} from {historical_data_cutoff_time} to {end_time}")

        if isinstance(model, F107SWPC):
            logging.info(
                f"Reading swpc from {historical_data_cutoff_time.replace(hour=0, minute=0, second=0)} to {end_time}"
            )
            data_one_model = model.read(
                historical_data_cutoff_time.replace(hour=0, minute=0, second=0),
                end_time,
                download=download,
            )

        data_out = construct_updated_data_frame(data_out, data_one_model, model.LABEL)

        # if no NaNs are present anymore, we don't have to read backups
        if not any_nans(data_out):
            break

    if len(data_out) == 1:
        data_out = data_out[0]

    return data_out
