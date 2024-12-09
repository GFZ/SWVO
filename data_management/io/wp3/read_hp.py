import datetime as dt
import glob
import logging
import os

import pandas as pd

from data_management.io.base_file_reader import BaseReader


class HpReader(BaseReader):
    """
    Reader class for Hp products from WP3 PAGER project. It reads Hp60/Hp30 forecasts.
    """

    def __init__(self, data_folder, index):
        """
        :param data_folder: The path where data is contained.
        :type data_folder: str
        :param index: it can be either hp30 or hp60
        :type index: str
        """
        super().__init__()
        self.data_folder = data_folder
        self._check_data_folder()
        self.index = index
        self._check_index()

    def _check_data_folder(self):
        if not os.path.exists(self.data_folder):
            msg = "Data folder not found...impossible to retrieve data."
            logging.error(msg)
            raise FileNotFoundError(msg)

    def _check_index(self):
        if self.index not in ["hp30", "hp60"]:
            raise RuntimeError(f"requested {self.index} index does not exist.." ".")

    @staticmethod
    def _correct_column_name_for_files_generated_before_2023(df):
        return df.rename(columns={"kp": "hp60"})

    def read(self, requested_date=None, model_name="HP60-FULL-SW-SWAMI-PAGER", header=False) -> tuple:
        """
        This function reads one of the available PAGER Hp forecast products.

        :param requested_date: Requested data for data to read. If None it reads data from the latest file produced.
        :type requested_date: datetime.datetime or None
        :param model_name:
        :type model_name: str or None
        :param header:
        :type header: bool
        :raises: RuntimeError: This exception is raised if the sources of data requested is not among
                 the available ones.

        :return: tuple of data in pandas.DataFrame format and datetime.datetime of the date extracted from the file.
        """
        file_to_read = None
        if requested_date is None:
            requested_date = dt.datetime.utcnow().replace(microsecond=0, minute=0, second=0)

        date_found = None
        files = sorted(glob.glob(self.data_folder + "/*"))
        for file in files:
            date = file.split("/")[-1]
            date = date.split(".")[0]
            target_date = requested_date
            date = dt.datetime.strptime(date.split("_")[-1], "%Y%m%dT%H%M%S")
            if target_date == date:
                if model_name not in file:
                    continue
                file_to_read = file
                date_found = date

        if file_to_read is None:
            logging.error(
                f"File not found in folder {self.data_folder} for the date {requested_date} and " f"model name {model_name}"
            )
            return None, None

        if not header:
            df = pd.read_csv(file_to_read, names=["t", self.index])
        else:
            df = pd.read_csv(file_to_read)
            if "kp" in df.columns:
                df = HpReader._correct_column_name_for_files_generated_before_2023(df)

        df["t"] = pd.to_datetime(df["t"])
        df.index = df["t"]
        df.drop(labels=["t"], axis=1, inplace=True)

        return df, date_found


class HpEnsembleReader(HpReader):
    def __init__(self, data_folder, index):
        """
        :param data_folder: The path to data folder of the index output
        :type data_folder: str
        :param index: it can be either hp30 or hp60
        :type index: str
        """
        super().__init__(data_folder, index)

    def _read_ensemble_files(self, folder, requested_date=None, header=False, model_name=None) -> (list, str):
        if requested_date is None:
            requested_date = dt.datetime.utcnow().replace(microsecond=0, minute=0, second=0)

        str_date = requested_date.strftime("%Y%m%dT%H%M%S")
        file_list = sorted(glob.glob(folder + "/*" + model_name + "_*" + str_date + "*ensemble*.csv"))

        data = []
        for file in file_list:
            if not header:
                df = pd.read_csv(file, names=["t", self.index])
            else:
                df = pd.read_csv(file)
            df["t"] = pd.to_datetime(df["t"])
            df.index = df["t"]
            df.drop(labels=["t"], axis=1, inplace=True)
            data.append(df)

        if len(data) == 0:
            msg = f"No ensemble file found for requested date {requested_date}"
            logging.warning(msg)
            return None, None
        return data, requested_date

    def read(self, model_name, requested_date=None, header=False, *args) -> (list, str):
        data, data_timestamp = self._read_ensemble_files(
            self.data_folder, requested_date, header=header, model_name=model_name
        )
        return data, data_timestamp
