import datetime as dt
import logging
import os

import pandas as pd

from data_management.io.base_file_reader import BaseReader


class PlasmasphereCombinedInputsReader(BaseReader):
    def __init__(self, wp3_output_folder, sub_folder="GFZ_PLASMA"):
        super().__init__()
        self.data_folder = os.path.join(wp3_output_folder, sub_folder)
        self._check_data_folder()
        self.file = None
        self.requested_date = None

    def _check_data_folder(self):
        if not os.path.exists(self.data_folder):
            msg = f"Data folder {self.data_folder} for WP3 plasma output not found...impossible to retrieve data."
            logging.error(msg)
            raise FileNotFoundError(msg)

    @staticmethod
    def _read_single_file(folder, date, source):
        if source == "kp":
            file_name = f"kp_{date.year}-{str(date.month).zfill(2)}-{str(date.day).zfill(2)}-{str(date.hour).zfill(2)}-{str(date.minute).zfill(2)}.csv"
        if source == "solar_wind":
            file_name = f"solar_wind_{date.year}-{str(date.month).zfill(2)}-{str(date.day).zfill(2)}-{str(date.hour).zfill(2)}-{str(date.minute).zfill(2)}.csv"

        file_path = os.path.join(folder, file_name)

        if not os.path.isfile(file_path):
            msg = f"No suitable files found in the folder {folder} for the requested date {date}"
            logging.warning(msg)
            return None

        if source == "solar_wind":
            data = pd.read_csv(file_path, parse_dates=["date", "date_of_run"])
            data["t"] = data["date"]
            data.drop(labels=["date"], axis=1, inplace=True)
        if source == "kp":
            data = pd.read_csv(file_path, parse_dates=["t", "date_of_run"])

        return data

    def read(self, source, requested_date=None) -> pd.DataFrame:
        if requested_date is None:
            requested_date = dt.datetime.utcnow().replace(microsecond=0, minute=0, second=0)

        if source == "kp":
            requested_date = requested_date.replace(minute=0, second=0, microsecond=0)
            return self._read_single_file(os.path.join(self.data_folder, "inputs/kp"), requested_date, "kp")
        if source == "solar_wind":
            requested_date = requested_date.replace(minute=0, second=0, microsecond=0)
            return self._read_single_file(
                os.path.join(self.data_folder, "inputs/solar_wind"), requested_date, "solar_wind"
            )
        msg = f"Combined input {source} requested not available..."
        logging.error(msg)
        raise RuntimeError(msg)
