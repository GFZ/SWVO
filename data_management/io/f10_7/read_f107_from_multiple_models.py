from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import logging
from typing import List, Type

from data_management.io.f10_7 import F107OMNI, F107SWPC


def read_f107_from_multiple_models(
    start_time: datetime,
    end_time: datetime,
    model_order: List[Type] = None,
    synthetic_now_time: datetime = datetime.now(timezone.utc),
    download=False,
):

    if model_order is None:
        model_order = [F107OMNI(), F107SWPC()]
        logging.warning("No model order specified, using default order: OMNI, SWPC")

    data_out = pd.DataFrame()

    for model in model_order:

        if isinstance(model, F107OMNI):
            print(f"Reading omni from {start_time} to {end_time}")
            data_one_model = model.read(start_time, end_time, download=download)
            model_label = "omni"
            
            data_one_model.replace(999.9, np.nan, inplace=True)
            index_range = pd.date_range(start=synthetic_now_time, end=end_time, freq='d')
            data_one_model = data_one_model.reindex(data_one_model.index.union(index_range))


            data_one_model.loc[synthetic_now_time:end_time, "f107"] = np.nan
            data_one_model.fillna({"file_name": np.nan}, inplace=True)
            logging.info(f"Setting NaNs in OMNI from {synthetic_now_time} to {end_time}")


        if isinstance(model, F107SWPC):
            print(f"Reading swpc from {synthetic_now_time.replace(hour=0, minute=0, second=0)} to {end_time}")
            data_one_model = model.read(synthetic_now_time.replace(hour=0, minute=0, second=0), end_time, download=download)
            model_label = "swpc"

        any_nans_found = False
        data_one_model["model"] = model_label
        data_one_model.loc[data_one_model["f107"].isna(), "model"] = None
        data_out = data_out.combine_first(data_one_model)

        if data_out["f107"].isna().sum() > 0:
            any_nans_found = True

        logging.info(f"Found {data_out['f107'].isna().sum()} NaNs in {model_label}")

        # if no NaNs are present anymore, we don't have to read backups
        if not any_nans_found:
            break

    if len(data_out) == 1:
        data_out = data_out[0]

    return data_out
