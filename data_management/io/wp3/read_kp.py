import datetime as dt
import glob
import logging
import os

import pandas as pd

from data_management.io.base_file_reader import BaseReader


class KPReader(BaseReader):
    """
    Reader class for Kp products from WP3 PAGER project. It reads data from different sources of Kp data,
    e.g. SWPC forecast, NIEMEGK nowcast and L1 real time based forecasts.
    """

    def __init__(self, wp3_output_folder):
        """
        :param wp3_output_folder: The path to data outputs for WP3. It needs to contain sub-folders with individual
                                  products (e.g. SWPC, NIEMEGK).
        :type wp3_output_folder: str
        """
        super().__init__()
        self.data_folder = wp3_output_folder
        self._check_data_folder()

    def _check_data_folder(self):
        if not os.path.exists(self.data_folder):
            msg = "Data folder for WP3 KP output not found...impossible to retrieve data."
            logging.error(msg)
            raise FileNotFoundError(msg)

    @staticmethod
    def _read_single_file(folder, requested_date=None, header=False, model_name=None) -> tuple:
        """
        Reads a single file product with Kp data from PAGER.

        :param folder: The folder path of the file to read.
        :type folder: str
        :param requested_date: Requested date for data to read.
        :type requested_date: datetime.datetime
        :param header: True if the file product contains a header, False if not. (In the future we will uniform
                       the output formats and remove this parameter).
        :type header: bool

        :return: Tuple of data in pandas.DataFrame format and datetime.datetime of the date extracted from the file.
        """
        file_to_read = None
        if requested_date is None:
            requested_date = dt.datetime.utcnow().replace(microsecond=0, minute=0, second=0)

        date_found = None
        files = sorted(glob.glob(folder + "/*"))
        for file in files:

            date = file.split("/")[-1]
            date = date.split(".")[0]
            try:
                date = dt.datetime.strptime(date.split("_")[-1], "%Y%m%d")
                hours = False
            except ValueError:
                date = dt.datetime.strptime(date.split("_")[-1], "%Y%m%dT%H%M%S")
                hours = True

            if (not hours) and (requested_date.replace(hour=0) == date):
                file_to_read = file
                date_found = date
                break
                
            if hours and (requested_date == date):
                if (model_name is not None) and (model_name not in file):
                    break
                else:
                    file_to_read = file
                    date_found = date
                    break

        try:
            if not header:
                # TODO Attention, this is not valid for hp or other indexes
                df = pd.read_csv(file_to_read, names=["t", "kp"])
            else:
                df = pd.read_csv(file_to_read)
            df["t"] = pd.to_datetime(df["t"])
            df.index = df["t"]
            df.drop(labels=["t"], axis=1, inplace=True)
            return df, date_found
        except FileNotFoundError:
            logging.error("File not found in folder {}...".format(folder))
            return None, None
        except ValueError:
            logging.error("No file found for requested date {}".format(requested_date))
            return None, None

    def read(self, source, requested_date=None, model_name=None, header=False) -> tuple:
        """
        This function reads one of the available PAGER non-ensemble Kp forecast products.

        :param source: The source of Kp product requested. Choose among "niemegk", "swpc"  and "l1"
        :type source: str
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

        if source == "niemegk":
            data, data_timestamp = self._read_single_file(os.path.join(self.data_folder, "NIEMEGK"), requested_date,
                                                          header=header)
        elif source == "swpc":
            data, data_timestamp = self._read_single_file(os.path.join(self.data_folder, "SWPC"), requested_date,
                                                          header=header)
        elif source == "l1":
            assert model_name is not None
            data, data_timestamp = self._read_single_file(os.path.join(self.data_folder, "L1_FORECAST"), requested_date,
                                                          model_name=model_name, header=header)
        else:
            msg = "Source {} requested for reading Kp not available...".format(source)
            logging.error(msg)
            raise RuntimeError(msg)

        return data, data_timestamp


class KPEnsembleReader(KPReader):
    """
    Reader class for Kp ensemble forecast from WP3 PAGER project. It is tailored to swift ensemble forecast
    based Kp forecast.
    """

    def __init__(self, wp3_output_folder, sub_folder="SWIFT_ENSEMBLE"):
        """
        :param wp3_output_folder: The path to data outputs for WP3
        :type wp3_output_folder: str
        :param sub_folder: Sub-folder with data from ensemble forecast
        :type sub_folder: str
        """
        super().__init__(os.path.join(wp3_output_folder, sub_folder))

    @staticmethod
    def _read_ensemble_files(folder, requested_date=None, header=False, model_name=None) -> (list, str):
        if requested_date is None:
            requested_date = dt.datetime.utcnow().replace(microsecond=0, minute=0, second=0)

        str_date = requested_date.strftime("%Y%m%dT%H%M%S")
        file_list = sorted(glob.glob(folder + "/*" + model_name + "_*" + str_date + "*ensemble*.csv"))

        data = []
        for file in file_list:
            if not header:
                df = pd.read_csv(file, names=["t", "kp"])
            else:
                df = pd.read_csv(file)
            df["t"] = pd.to_datetime(df["t"])
            df.index = df["t"]
            df.drop(labels=["t"], axis=1, inplace=True)
            data.append(df)

        if len(data) == 0:
            msg = "No Kp ensemble file found for requested date {}".format(requested_date)
            logging.warning(msg)
            return None, None
        else:
            return data, requested_date

    def read(self, model_name, requested_date=None, header=False, *args) -> (list, str):
        data, data_timestamp = self._read_ensemble_files(self.data_folder, requested_date,
                                                         header=header, model_name=model_name)
        return data, data_timestamp
