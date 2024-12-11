import os
import logging
import pandas as pd
import datetime as dt

from data_management.io.base_file_reader import BaseReader


class PlasmaspherePredictionReader(BaseReader):

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
    def _read_single_file(folder, date):
        """
        It reads the plasmasphere prediction from the specified folder.

        :param folder: folder where we look for the plasmasphere prediction
        :type folder: str
        :param date:
        :type date:
        :return: the plasmasphere prediction for the requested date
        :rtype: pandas.DataFrame
        """

        file_name = "plasmasphere_density_{}-{}-{}-{}-{}.csv".format(date.year,
                                                                     str(date.month).zfill(2),
                                                                     str(date.day).zfill(2),
                                                                     str(date.hour).zfill(2),
                                                                     str(date.minute).zfill(2))

        file_path = os.path.join(folder, file_name)

        if not os.path.isfile(file_path):
            msg = "No suitable files found in the folder {} for the requested date {}".format(folder, date)
            logging.warning(msg)
            return None

        data = pd.read_csv(file_path, parse_dates=["date"])
        data["t"] = data["date"]
        data.drop(labels=["date"], axis=1, inplace=True)
        return data

    def read(self, source, requested_date=None) -> pd.DataFrame:
        """
        Reads one of the available PAGER plasmasphere density prediction.

        :param source: The source of plasmasphere density product requested. Available only "gfz_plasma".
        :type source: str
        :param requested_date: Date of plasma density prediction thar we want to read up to hour precision.
        :type requested_date: datetime.datetime or None

        :raises: RuntimeError if the sources of data requested is not among the available ones.

        :return: pandas.DataFrame with L, MLT, density and date as columns
        """

        if requested_date is None:
            requested_date = dt.datetime.utcnow().replace(microsecond=0, minute=0, second=0)

        if source == "gfz_plasma":
            requested_date = requested_date.replace(minute=0, second=0, microsecond=0)
            return self._read_single_file(self.data_folder, requested_date)
        else:
            msg = "Source {} requested for reading plasmasphere prediction not available...".format(source)
            logging.error(msg)
            raise RuntimeError(msg)
