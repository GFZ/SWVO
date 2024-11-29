from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import logging
from typing import List, Type

from data_management.io.solar_wind import SWACE, SWOMNI, SWSWIFTEnsemble


def read_solar_wind_from_multiple_models(
    start_time: datetime,
    end_time: datetime,
    model_order: List[Type] = None,
    reduce_ensemble=None,
    synthetic_now_time: datetime = datetime.now(timezone.utc),
    download=False,
):

    if model_order is None:
        model_order = [SWOMNI(), SWACE(), SWSWIFTEnsemble()]
        logging.warning("No model order specified, using default order: OMNI, ACE, SWIFT ensemble")

    data_out = [pd.DataFrame()]

    for model in model_order:

        if isinstance(model, SWOMNI):
            print(f"Reading omni from {start_time} to {end_time}")
            data_one_model = [model.read(start_time, end_time, cadence_min=1, download=download)]

            for i, _ in enumerate(data_one_model):
                data_one_model[i].loc[synthetic_now_time:end_time, "kp"] = np.nan
            model_label = "omni"
            logging.info(f"Setting NaNs in OMNI from {synthetic_now_time} to {end_time}")

        if isinstance(model, SWACE):
            print(f"Reading ACE from {start_time} to {end_time}")
            data_one_model = [model.read(start_time, end_time, download=download)]
            data_one_model[0].bfill(inplace=True)
            for i, _ in enumerate(data_one_model):
                data_one_model[i].loc[synthetic_now_time:end_time, "kp"] = np.nan
            model_label = "ace"
            logging.info(f"Setting NaNs in ACE from {synthetic_now_time} to {end_time}")


        if isinstance(model, SWSWIFTEnsemble):
            print("Reading PAGER SWIFT ensemble...")

            # we are trying to read the most recent file; it this fails, we go one step back (1 day) and see if this file is present

            target_time = synthetic_now_time

            data_one_model = []

            while target_time > (synthetic_now_time - timedelta(days=3)):

                # target_time = target_time.replace(hour=0, minute=0, second=0)

                data_one_model = model.read(target_time, end_time)

                if len(data_one_model) > 0:
                    # interpolate to common index

                    for ie, _ in enumerate(data_one_model):

                        df_common_index = pd.DataFrame(
                            index=pd.date_range(
                                datetime(target_time.year, target_time.month, target_time.day),
                                datetime(end_time.year, end_time.month, end_time.day, 23, 59, 59),
                                freq=timedelta(minutes=1),
                                tz="UTC",
                            )
                        )
                        df_common_index.index.name = data_one_model[ie].index.name

                        for colname, col in data_one_model[ie].items():
                            if col.dtype == "object":
                                # this is the filename column
                                df_common_index[colname] = col.iloc[0]
                            else:
                                df_common_index[colname] = np.interp(df_common_index.index, data_one_model[ie].index, col)

                        data_one_model[ie] = df_common_index
                        data_one_model[ie] = data_one_model[ie].truncate(
                            before=start_time - timedelta(minutes=0.999999), after=end_time + timedelta(minutes=0.999999)
                        )

                    break

                target_time -= timedelta(days=1)

            model_label = "swift"

            # num_ens_members = len(data_one_model)

            # if reduce_ensemble == "mean":

            #     kp_mean_ensembles = []

            #     for it, _ in enumerate(data_one_model[0].index):
            #         data_curr_time = []
            #         for ie in range(num_ens_members):
            #             data_curr_time.append(data_one_model[ie].loc[data_one_model[ie].index[it], "kp"])

            #         kp_mean_ensembles.append(np.mean(data_curr_time))

            #     data_one_model = [pd.DataFrame(index=data_one_model[0].index, data={"kp": kp_mean_ensembles})]

            # elif reduce_ensemble is None:
            #     data_out = data_out * num_ens_members

        any_nans_found = False
        # we making it a list in case of ensemble memblers
        for i, _ in enumerate(data_one_model):
            data_one_model[i]["model"] = model_label
            data_one_model[i].loc[data_one_model[i].isna().any(axis=1), "model"] = None
            data_out[i] = data_out[i].combine_first(data_one_model[i])

            if data_out[i].isna().any(axis=1).sum() > 0:
                any_nans_found = True

            logging.info(f"Found {data_out[i].isna().sum()} NaNs in {model_label}")

        # if no NaNs are present anymore, we don't have to read backups
        if not any_nans_found:
            break

    if len(data_out) == 1:
        data_out = data_out[0]

    return data_out
