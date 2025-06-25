# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
#
# SPDX-License-Identifier: Apache-2.0

import os
import logging
import pandas as pd
import datetime as dt

from data_management.io.base_file_reader import BaseReader


class PlasmasphereCombinedInputsReader(BaseReader):

    def __init__(self, folder):
        super().__init__()
        self.data_folder = folder
        self._check_data_folder()
        self.file = None
        self.requested_date = None

    def _check_data_folder(self):
        if not os.path.exists(self.data_folder):
            msg = "Data folder {} for WP3 plasma output not found...impossible to retrieve data.".format(
                self.data_folder)
            logging.error(msg)
            raise FileNotFoundError(msg)

    @staticmethod
    def _read_single_file(folder, date, source):

        if source == "kp":
            file_name = "kp_{}-{}-{}-{}-{}.csv".format(date.year,
                                                       str(date.month).zfill(2),
                                                       str(date.day).zfill(2),
                                                       str(date.hour).zfill(2),
                                                       str(date.minute).zfill(2))
        if source == "solar_wind":
            file_name = "solar_wind_{}-{}-{}-{}-{}.csv".format(date.year,
                                                       str(date.month).zfill(2),
                                                       str(date.day).zfill(2),
                                                       str(date.hour).zfill(2),
                                                       str(date.minute).zfill(2))


        file_path = os.path.join(folder, file_name)

        if not os.path.isfile(file_path):
            msg = "No suitable files found in the folder {} for the requested date {}".format(folder, date)
            logging.warning(msg)
            return None

        if source == "solar_wind":
            data = pd.read_csv(file_path, parse_dates=["date"])
            data["t"] = data["date"]
            data.drop(labels=["date"], axis=1, inplace=True)
        if source == "kp":
            data = pd.read_csv(file_path, parse_dates=["t"])

        return data

    def read(self, source, requested_date=None) -> pd.DataFrame:

        if requested_date is None:
            requested_date = dt.datetime.utcnow().replace(microsecond=0, minute=0, second=0)

        if source == "kp":
            requested_date = requested_date.replace(minute=0, second=0, microsecond=0)
            return self._read_single_file(os.path.join(self.data_folder, "inputs/kp"),
                                          requested_date, "kp")
        elif source == "solar_wind":
            requested_date = requested_date.replace(minute=0, second=0, microsecond=0)
            return self._read_single_file(os.path.join(self.data_folder, "inputs/solar_wind"),
                                          requested_date, "solar_wind")
        else:
            msg = "Combined input {} requested not available...".format(source)
            logging.error(msg)
            raise RuntimeError(msg)
