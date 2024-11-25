from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import logging
from typing import List, Type

from data_management.io.kp import KpEnsemble, KpNiemegk, KpOMNI, KpSWPC


def read_kp_from_multiple_models(
    start_time: datetime,
    end_time: datetime,
    model_order: List[Type] = None,
    reduce_ensemble=None,
    synthetic_now_time: datetime = datetime.now(timezone.utc),
    download=False,
):

    if model_order is None:
        model_order = [KpOMNI(), KpNiemegk(), KpEnsemble(), KpSWPC()]
        logging.warning("No model order specified, using default order: OMNI, Niemegk, Ensemble, SWPC")

    data_out = [pd.DataFrame()]

    for model in model_order:

        if isinstance(model, KpOMNI):
            print(f"Reading omni from {start_time} to {end_time}")
            data_one_model = [model.read(start_time, end_time, download=download)]
            model_label = "omni"

        if isinstance(model, KpNiemegk):
            print(f"Reading niemegk from {start_time} to {end_time}")
            data_one_model = [model.read(start_time, end_time, download=download)]
            model_label = "niemegk"

        # Forecasting models are called with synthetic now time
        if isinstance(model, KpSWPC):
            print(f"Reading swpc from {synthetic_now_time} to {end_time}")
            data_one_model = [model.read(synthetic_now_time.replace(hour=0, minute=0, second=0), end_time, download=download)]
            model_label = "swpc"

        if isinstance(model, KpEnsemble):

            # we are trying to read the most recent file; it this fails, we go one step back (1 hour) and see if this file is present

            target_time = synthetic_now_time
            data_one_model = model.read(target_time, end_time)

            while len(data_one_model) == 0 and target_time > (synthetic_now_time - timedelta(days=3)):
                target_time -= timedelta(hours=1)

                # ONLY READ MIDNIGHT FILE FOR NOW; OTHER FILES BREAK
                target_time = target_time.replace(hour=0, minute=0, second=0)

                data_one_model = model.read(target_time, end_time)
            
            print(f"Reading PAGER Kp ensemble from {target_time} to {end_time}")
            model_label = "ensemble"

            num_ens_members = len(data_one_model)

            if reduce_ensemble == "mean":

                kp_mean_ensembles = []

                for it, _ in enumerate(data_one_model[0].index):

                    data_curr_time = []
                    for ie in range(num_ens_members):
                        data_curr_time.append(data_one_model[ie].loc[data_one_model[ie].index[it], "kp"])

                    kp_mean_ensembles.append(np.mean(data_curr_time))

                data_one_model = [pd.DataFrame(index=data_one_model[0].index, data={"kp": kp_mean_ensembles})]

            elif reduce_ensemble is None:
                data_out = data_out * num_ens_members

        any_nans_found = False
        # we making it a list in case of ensemble members
        for i, _ in enumerate(data_one_model):
            data_one_model[i]["model"] = model_label
            data_one_model[i].loc[data_one_model[i]["kp"].isna(), "model"] = None
            data_out[i] = data_out[i].combine_first(data_one_model[i])

            if data_out[i]["kp"].isna().sum() > 0:
                any_nans_found = True

        logging.info(f"Found {data_out[0]['kp'].isna().sum()} NaNs in {model_label}")
        # if no NaNs are present anymore, we don't have to read backups
        if not any_nans_found:
            break

    if len(data_out) == 1:
        data_out = data_out[0]

    return data_out
