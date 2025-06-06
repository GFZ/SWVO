from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from data_management.io.dst import DSTOMNI, DSTWDC
from data_management.io.utils import any_nans, construct_updated_data_frame

DSTModel = DSTOMNI | DSTWDC


def read_dst_from_multiple_models(
    start_time: datetime,
    end_time: datetime,
    model_order: list[DSTModel] | None = None,
    synthetic_now_time: datetime | None = None,
    *,
    download: bool = False,
) -> pd.DataFrame:
    """
    Read DST data from multiple models.

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
        Order in which data will be read from the models. Defaults to [OMNI, WDC].
    synthetic_now_time : datetime or None, optional
        Time representing "now". After this time, no data will be taken from
        historical models (OMNI, WDC). Defaults to None.
    download : bool, optional
        Flag indicating whether new data should be downloaded. Defaults to False.

    Returns
    -------
    :class:`pandas.DataFrame`
        A data frame containing data for the requested period.
    """

    if synthetic_now_time is None:
        synthetic_now_time = min(datetime.now(timezone.utc), end_time)

    if model_order is None:
        model_order = [DSTOMNI(), DSTWDC()]
        logging.warning("No model order specified, using default order: OMNI, WDC")

    data_out = pd.DataFrame()

    for model in model_order:
        logging.info(f"Reading {model.LABEL} from {start_time} to {end_time}")
        data_one_model = model.read(start_time, end_time, download=download)

        index_range = pd.date_range(
            start=synthetic_now_time, end=end_time, freq="h"
        )
        data_one_model = data_one_model.reindex(
            data_one_model.index.union(index_range)
        )

        data_one_model.loc[data_one_model.index > synthetic_now_time, "dst"] = np.nan
        data_one_model = data_one_model.fillna({"file_name": np.nan})
        logging.info(
            f"Setting NaNs in {model.LABEL} from {synthetic_now_time} to {end_time}"
        )

        data_out = construct_updated_data_frame(data_out, data_one_model, model.LABEL)

        # if no NaNs are present anymore, we don't have to read backups
        if not any_nans(data_out):
            break

    if len(data_out) == 1:
        data_out = data_out[0]

    return data_out
