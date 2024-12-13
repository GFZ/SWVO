from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from data_management.io.f10_7 import F107OMNI, F107SWPC
from data_management.io.utils import any_nans, construct_updated_data_frame

F107Model = F107OMNI | F107SWPC

def read_f107_from_multiple_models(
    start_time: datetime,
    end_time: datetime,
    model_order: list[F107Model]|None = None,
    synthetic_now_time: datetime|None = None,
    *,
    download:bool=False,
) -> pd.DataFrame:
    """
    Read F107 data from multiple models.

    The model order represents the priorities of models.
    The first model in the model order is read. If there are still NaNs in the resulting data,
    the next model will be read. And so on. In the case of reading ensemble predictions, a list
    will be returned, otherwise a plain data frame will be returned.

    :param start_time: Start time of the data request.
    :type start_time: datetime
    :param end_time: End time of the data request.
    :type end_time: datetime
    :param model_order: Order in which data will be read from the models, defaults to [OMNI, SWPC]
    :type model_order: list | None, optional
    :param synthetic_now_time: Time, which represents &quot;now&quot;.
    After this time, no data will be taken from historical models (OMNI, SWPC), defaults to None
    :type synthetic_now_time: datetime | None, optional
    :param download: Flag which decides whether new data should be downloaded, defaults to False
    :type download: bool, optional
    :return: A data frame or a list of data frames containing data for the requested period.
    :rtype: pd.DataFrame
    """
    if synthetic_now_time is None:
        synthetic_now_time = datetime.now(timezone.utc)

    if model_order is None:
        model_order = [F107OMNI(), F107SWPC()]
        logging.warning("No model order specified, using default order: OMNI, SWPC")

    data_out = pd.DataFrame()

    for model in model_order:
        if isinstance(model, F107OMNI):
            logging.info(f"Reading {model.LABEL} from {start_time} to {end_time}")
            data_one_model = model.read(start_time, end_time, download=download)

            index_range = pd.date_range(start=synthetic_now_time, end=end_time, freq="d")
            data_one_model = data_one_model.reindex(data_one_model.index.union(index_range))

            data_one_model.loc[synthetic_now_time:end_time, "f107"] = np.nan
            data_one_model = data_one_model.fillna({"file_name": np.nan})
            logging.info(f"Setting NaNs in {model.LABEL} from {synthetic_now_time} to {end_time}")

        if isinstance(model, F107SWPC):
            logging.info(f"Reading swpc from {synthetic_now_time.replace(hour=0, minute=0, second=0)} to {end_time}")
            data_one_model = model.read(
                synthetic_now_time.replace(hour=0, minute=0, second=0), end_time, download=download,
            )

        data_out = construct_updated_data_frame(data_out, data_one_model, model.LABEL)

        # if no NaNs are present anymore, we don't have to read backups
        if not any_nans(data_out):
            break

    if len(data_out) == 1:
        data_out = data_out[0]

    return data_out
