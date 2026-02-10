# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from swvo.io.dst import DSTOMNI, DSTWDC
from swvo.io.exceptions import ModelError
from swvo.io.utils import any_nans, construct_updated_data_frame

logger = logging.getLogger(__name__)

DSTModel = DSTOMNI | DSTWDC

logging.captureWarnings(True)


def read_dst_from_multiple_models(
    start_time: datetime,
    end_time: datetime,
    model_order: Sequence[DSTModel] | None = None,
    historical_data_cutoff_time: datetime | None = None,
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
    model_order : Sequence or None, optional
        Order in which data will be read from the models. Defaults to [OMNI, WDC].
    historical_data_cutoff_time : datetime or None, optional
        Time representing "now". After this time, no data will be taken from
        historical models (OMNI, WDC). Defaults to None.
    download : bool, optional
        Flag indicating whether new data should be downloaded. Defaults to False.

    Returns
    -------
    :class:`pandas.DataFrame`
        A data frame containing data for the requested period.
    """
    if historical_data_cutoff_time is None:
        historical_data_cutoff_time = min(datetime.now(timezone.utc), end_time)

    if model_order is None:
        model_order = [DSTOMNI(), DSTWDC()]
        logger.warning("No model order specified, using default order: OMNI, WDC")

    data_out = pd.DataFrame()

    for model in model_order:
        if not isinstance(model, DSTModel):
            raise ModelError(f"Unknown or incompatible model: {type(model).__name__}")
        logger.info(f"Reading {model.LABEL} from {start_time} to {end_time}")
        data_one_model = model.read(start_time, end_time, download=download)

        index_range = pd.date_range(start=historical_data_cutoff_time, end=end_time, freq="h")
        data_one_model = data_one_model.reindex(data_one_model.index.union(index_range))

        data_one_model.loc[data_one_model.index > historical_data_cutoff_time, "dst"] = np.nan
        data_one_model = data_one_model.fillna({"file_name": np.nan})
        logger.info(f"Setting NaNs in {model.LABEL} from {historical_data_cutoff_time} to {end_time}")

        data_out = construct_updated_data_frame(data_out, data_one_model, model.LABEL)

        # if no NaNs are present anymore, we don't have to read backups
        if not any_nans(data_out):
            break

    if len(data_out) == 1:
        data_out = data_out[0]

    return data_out
