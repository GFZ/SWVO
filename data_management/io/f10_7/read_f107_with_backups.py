from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from data_management.io.f10_7 import F107OMNI, F107SWPC


def read_f107_with_backups(
    start_time: datetime,
    end_time: datetime,
    model_order: list = None,
    synthetic_now_time: datetime = datetime.now(timezone.utc),
    download=False,
):

    if model_order is None:
        model_order = [F107OMNI(), F107SWPC()]

    data_out = [pd.DataFrame()]

    for model in model_order:

        if isinstance(model, F107OMNI):
            print("Reading omni...")
            data_one_model = [model.read(start_time, end_time, download=download)]
            model_label = "omni"

        if isinstance(model, F107SWPC):
            print("Reading swpc...")
            data_one_model = [model.read(synthetic_now_time.replace(hour=0, minute=0, second=0), end_time, download=download)]
            model_label = "swpc"

        any_nans_found = False
        # we making it a list in case of ensemble members
        for i, _ in enumerate(data_one_model):
            data_one_model[i]["model"] = model_label
            data_one_model[i].loc[data_one_model[i]["f107"].isna(), "model"] = None
            data_out[i] = data_out[i].combine_first(data_one_model[i])

            if data_out[i]["f107"].isna().sum() > 0:
                any_nans_found = True

        # if no NaNs are present anymore, we don't have to read backups
        if not any_nans_found:
            break

    if len(data_out) == 1:
        data_out = data_out[0]

    return data_out
