# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PlasmaspherePredictionReader:
    """Reads one of the available PAGER plasmasphere density prediction.

    Parameters
    ----------
    folder : str
        The folder where the plasmasphere prediction files are stored.

    Raises
    ------
    FileNotFoundError
        If the data folder does not exist.
    RuntimeError
        If the source of data requested is not among the available ones.
    """

    ENV_VAR_NAME = "PLASMASPHERE_OUTPUT_DIR"
    LABEL = "plasmsphere"

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            if self.ENV_VAR_NAME not in os.environ:
                raise ValueError(f"Necessary environment variable {self.ENV_VAR_NAME} not set!")

            data_dir = os.environ.get(self.ENV_VAR_NAME)  # ty: ignore[invalid-assignment]

        self.data_dir: Path = Path(data_dir)  # ty:ignore[invalid-argument-type]

        logger.info(f"Plasmasphere data directory: {self.data_dir}")

        if not self.data_dir.exists():
            msg = f"Plasmasphere directory {self.data_dir} does not exist! Impossible to retrive data!"
            logger.error(msg)
            raise FileNotFoundError(msg)

    def read(self, requested_date: datetime | None = None) -> pd.DataFrame | None:
        """
        Reads one of the available PAGER plasmasphere density prediction.

        Parameters
        ----------
        requested_date : datetime.datetime or None
            Date of plasma density prediction thar we want to read up to hour precision.

        Raises
        ------
        RuntimeError
            if the sources of data requested is not among the available ones.

        Returns
        -------
        pd.DataFrame or None
            pandas.DataFrame with L, MLT, density and date as columns
        """

        if requested_date is None:
            requested_date = datetime.now(timezone.utc).replace(microsecond=0, minute=0, second=0)

        requested_date = requested_date.replace(minute=0, second=0, microsecond=0)

        file_name = f"plasmasphere_density_{requested_date.year}{str(requested_date.month).zfill(2)}{str(requested_date.day).zfill(2)}T{str(requested_date.hour).zfill(2)}00.csv"

        file_path = os.path.join(self.data_dir, file_name)
        logger.info(f"Looking for file {file_path} for date {requested_date}")
        if not os.path.isfile(file_path):
            msg = f"No suitable files ({file_path}) found in the folder {self.data_dir} for the requested date {requested_date}"
            logger.warning(msg)
            return None

        data = pd.read_csv(file_path, parse_dates=["date"])
        data["t"] = data["date"]
        data.drop(labels=["date"], axis=1, inplace=True)
        return data
